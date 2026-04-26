//! The core agent loop — the heart of ragent's agentic behaviour.
//!
//! # Phase 2 (PLAN.md)
//! Goal: "Agent autonomously uses tools to accomplish multi-step tasks."
//!
//! # How the Loop Works
//!
//! ```text
//!   User message
//!       │
//!       ▼
//!   ┌──────────────────────────────────────┐
//!   │  1. Add user message to conversation │
//!   │  2. Send conversation + tool defs    │──► LLM
//!   │     to LLM                           │
//!   │  3. Receive response                 │◄── LLM
//!   │  4. If response has tool_calls:      │
//!   │     a. Execute each tool             │
//!   │     b. Append tool results           │
//!   │     c. Go to step 2                  │
//!   │  5. Else: return text response       │
//!   └──────────────────────────────────────┘
//!       │
//!       ▼
//!   Final text response shown to user
//! ```
//!
//! # Safety Limits
//!
//! The loop has a **max iterations** limit to prevent infinite tool-call cycles.
//! If the LLM keeps requesting tools beyond this limit, the loop stops and
//! returns whatever the LLM last said (or an error message).
//!
//! # Design Decision: Generic over `LlmBackend`
//!
//! `AgentLoop<B: LlmBackend>` uses a generic parameter rather than a trait
//! object (`Box<dyn LlmBackend>`) because `LlmBackend` has `async fn` which
//! isn't object-safe. For Phase 2 this is fine — we only have `OllamaBackend`.
//! The generic compiles to static dispatch (zero overhead, inlined calls).

use anyhow::{Context, Result};
use tracing::{debug, error, info, instrument, warn};

use crate::agent::message::{Conversation, FunctionCall, Message, Role, ToolCall};
use crate::llm::traits::LlmBackend;
use crate::tools::ToolRegistry;

// ---------------------------------------------------------------------------
// AgentLoop
// ---------------------------------------------------------------------------

/// The agentic loop that orchestrates LLM ↔ tool interactions.
///
/// # Type Parameter
/// * `B` — The LLM backend (e.g. `OllamaBackend`). Using a generic keeps
///   this zero-cost (static dispatch) while still being swappable.
pub struct AgentLoop<B: LlmBackend> {
    /// The LLM backend used for chat completions.
    backend: B,

    /// Registry of tools the agent can use.
    registry: ToolRegistry,

    /// The conversation history (system prompt + all messages).
    conversation: Conversation,

    /// Maximum number of LLM round-trips per `process_message` call.
    /// Each round-trip = one LLM call that might produce tool calls.
    /// This prevents infinite loops if the LLM keeps calling tools endlessly.
    max_iterations: usize,
}

impl<B: LlmBackend> AgentLoop<B> {
    /// Create a new agent loop.
    ///
    /// # Arguments
    /// * `backend` — The LLM backend to use for inference.
    /// * `registry` — The set of tools available to the agent.
    /// * `system_prompt` — Instructions that tell the LLM how to behave.
    /// * `max_iterations` — Safety limit on tool-call loop iterations.
    pub fn new(
        backend: B,
        registry: ToolRegistry,
        system_prompt: &str,
        max_iterations: usize,
    ) -> Self {
        info!(
            tools = ?registry.names(),
            max_iterations = max_iterations,
            "Agent loop initialised"
        );

        Self {
            backend,
            registry,
            conversation: Conversation::with_system_prompt(system_prompt),
            max_iterations,
        }
    }

    /// Process a user message through the full agent loop.
    ///
    /// This is the main entry point. It:
    /// 1. Adds the user message to the conversation
    /// 2. Calls the LLM (with tool definitions)
    /// 3. If the LLM requests tool calls, executes them and loops
    /// 4. Returns the final text response
    ///
    /// # Returns
    /// The assistant's final text response after all tool calls are resolved.
    #[instrument(skip(self), fields(input_len = user_input.len()))]
    pub async fn process_message(&mut self, user_input: &str) -> Result<String> {
        // Step 1: Add the user's message to conversation history.
        self.conversation.add_user_message(user_input);

        let tool_defs = self.registry.tool_definitions();
        let has_tools = !tool_defs.is_empty();

        // Step 2-4: Loop until the LLM responds with plain text (no tool calls).
        for iteration in 1..=self.max_iterations {
            info!(iteration = iteration, "Agent loop iteration");

            // Call the LLM with the full conversation + tool definitions.
            let response = self
                .backend
                .chat(
                    self.conversation.messages(),
                    if has_tools { &tool_defs } else { &[] },
                )
                .await
                .context("LLM chat request failed")?;

            // Check if the LLM wants to call tools.
            //
            // Some models (especially smaller ones) understand tool calling
            // conceptually but return the tool call as a JSON blob in the
            // `content` field instead of using the structured `tool_calls`
            // format. We detect this and extract the tool call so the agent
            // loop works regardless of how the model formats its intent.
            let tool_calls = match &response.tool_calls {
                Some(calls) if !calls.is_empty() => Some(calls.clone()),
                _ => {
                    // Fallback: try to parse a tool call from the content text.
                    match try_parse_tool_call_from_content(&response.content, &self.registry) {
                        Some(parsed_calls) => {
                            info!(
                                count = parsed_calls.len(),
                                "Extracted tool call(s) from content text (model used text instead of structured format)"
                            );
                            Some(parsed_calls)
                        }
                        None => None,
                    }
                }
            };

            // Always add the assistant's response to the conversation.
            // This is important even when there are tool calls — the LLM
            // needs to see its own tool-call message in the history.
            self.conversation.push(response.clone());

            match tool_calls {
                Some(ref calls) if !calls.is_empty() => {
                    // The LLM wants to use tools. Execute each one and feed
                    // the results back into the conversation.
                    info!(
                        tool_count = calls.len(),
                        iteration = iteration,
                        "LLM requested tool calls"
                    );

                    for call in calls {
                        let tool_name = &call.function.name;
                        let tool_args = &call.function.arguments;

                        info!(
                            tool = %tool_name,
                            args = %serde_json::to_string(tool_args).unwrap_or_default(),
                            "Executing tool"
                        );

                        let tool_result: crate::tools::ToolResult =
                            match self.registry.get(tool_name) {
                                Some(tool) => {
                                    // Execute the tool, catching any infrastructure errors.
                                    let fut = tool.execute(tool_args.clone());
                                    match fut.await {
                                        Ok(result) => {
                                            if result.is_error {
                                                warn!(
                                                    tool = %tool_name,
                                                    error = %result.content,
                                                    "Tool returned error"
                                                );
                                            } else {
                                                debug!(
                                                    tool = %tool_name,
                                                    output_len = result.content.len(),
                                                    "Tool executed successfully"
                                                );
                                            }
                                            result
                                        }
                                        Err(e) => {
                                            // Infrastructure error (not a tool-level error).
                                            // Convert to a tool result so the LLM can see it.
                                            error!(
                                                tool = %tool_name,
                                                error = %e,
                                                "Tool execution failed with infrastructure error"
                                            );
                                            crate::tools::ToolResult::error(format!(
                                                "Tool execution failed: {e}"
                                            ))
                                        }
                                    }
                                }
                                None => {
                                    // LLM called a tool that doesn't exist. This happens
                                    // when the LLM hallucinates tool names. Feed the error
                                    // back so it can correct itself.
                                    warn!(
                                        tool = %tool_name,
                                        available = ?self.registry.names(),
                                        "LLM called unknown tool"
                                    );
                                    crate::tools::ToolResult::error(format!(
                                        "Unknown tool '{tool_name}'. Available tools: {}",
                                        self.registry
                                            .names()
                                            .iter()
                                            .map(|s: &String| s.as_str())
                                            .collect::<Vec<&str>>()
                                            .join(", ")
                                    ))
                                }
                            };

                        // Add the tool result to the conversation as a Tool message.
                        // The LLM will see this in the next iteration and can decide
                        // what to do next (call another tool, or respond with text).
                        let tool_message = Message {
                            role: Role::Tool,
                            content: tool_result.content,
                            tool_calls: None,
                            // Use the tool call ID if the LLM provided one (OpenAI format).
                            // Ollama doesn't use IDs, but we propagate them for compatibility.
                            tool_call_id: call.id.clone(),
                        };

                        self.conversation.push(tool_message);
                    }

                    // Continue the loop — send the tool results back to the LLM.
                    continue;
                }

                _ => {
                    // No tool calls — the LLM responded with plain text.
                    // This is the final answer.
                    let final_text = response.content.clone();

                    info!(
                        iterations = iteration,
                        response_len = final_text.len(),
                        "Agent loop completed"
                    );

                    return Ok(final_text);
                }
            }
        }

        // If we get here, we hit the max iterations limit.
        warn!(
            max_iterations = self.max_iterations,
            "Agent loop hit max iterations safety limit"
        );

        // Return whatever the LLM last said, plus a warning.
        let last_content = self
            .conversation
            .messages()
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        if last_content.is_empty() {
            Ok(format!(
                "[ragent: Agent loop stopped after {} iterations without a final response. \
                 The model may be stuck in a tool-calling cycle.]",
                self.max_iterations
            ))
        } else {
            Ok(format!(
                "{last_content}\n\n[ragent: Response may be incomplete — \
                 agent loop hit the {limit}-iteration safety limit.]",
                limit = self.max_iterations
            ))
        }
    }

    /// Get a reference to the conversation history.
    ///
    /// Useful for inspecting what happened during the agent loop, or for
    /// displaying the full conversation in a UI.
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// Clear the conversation history and start fresh.
    ///
    /// Keeps the system prompt (re-creates it from the original).
    /// Tools and backend remain the same.
    pub fn clear_conversation(&mut self, system_prompt: &str) {
        self.conversation = Conversation::with_system_prompt(system_prompt);
        info!("Conversation cleared");
    }
}

// ---------------------------------------------------------------------------
// Fallback tool-call parser (free functions, outside the impl block)
// ---------------------------------------------------------------------------

/// Try to extract tool calls from the LLM's content text.
///
/// # Why this exists
///
/// Some models (especially smaller ones like 7B) understand they should call
/// tools but output the calls as JSON code blocks in `content` instead of
/// using the structured `tool_calls` field. Worse, they often embed
/// multiple tool calls interleaved with explanatory prose.
///
/// This function scans the entire content for all JSON code blocks and
/// bare JSON objects, extracts any that look like tool calls with a name
/// matching a registered tool, and returns them all. This avoids false
/// positives because we only match on known tool names.
fn try_parse_tool_call_from_content(
    content: &str,
    registry: &ToolRegistry,
) -> Option<Vec<ToolCall>> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut all_calls: Vec<ToolCall> = Vec::new();

    // Strategy 1: Extract ALL ```json ... ``` code blocks from the content.
    // This handles the common case where the model writes prose around
    // multiple tool calls, each in its own fenced code block.
    for json_str in extract_all_json_codeblocks(trimmed) {
        if let Some(call) = parse_single_tool_call(json_str, registry) {
            all_calls.push(call);
        } else if let Some(mut calls) = parse_tool_call_array(json_str, registry) {
            all_calls.append(&mut calls);
        }
    }

    if !all_calls.is_empty() {
        return Some(all_calls);
    }

    // Strategy 2: Scan the text for bare JSON objects embedded in prose.
    // Some models (especially on later iterations) mix tool calls inline
    // with explanation text, e.g.:
    //   "Let's check the compiler. {"name": "shell_exec", "arguments": {...}}"
    // We find all `{...}` substrings and check if any are valid tool calls.
    for json_str in extract_embedded_json_objects(trimmed) {
        if let Some(call) = parse_single_tool_call(json_str, registry) {
            all_calls.push(call);
        }
    }

    if !all_calls.is_empty() {
        return Some(all_calls);
    }

    None
}

/// Extract ALL JSON strings from markdown code blocks in the text.
///
/// Finds every `` ```json ... ``` `` and `` ``` ... ``` `` pair and returns
/// the trimmed content of each. This is the key improvement over the previous
/// version which only found the first block.
fn extract_all_json_codeblocks(text: &str) -> Vec<&str> {
    let mut results = Vec::new();
    let mut search_from = 0;

    while search_from < text.len() {
        let remaining = &text[search_from..];

        // Find the next opening fence: ```json or ```
        let (_fence_start, content_start) = if let Some(pos) = remaining.find("```json") {
            (pos, pos + "```json".len())
        } else if let Some(pos) = remaining.find("```") {
            (pos, pos + "```".len())
        } else {
            break;
        };

        // Skip past any whitespace/newline after the opening fence
        let after_open = &remaining[content_start..];

        // Find the closing ``` fence
        let close_pos = match after_open.find("```") {
            Some(p) => p,
            None => break, // unclosed fence, stop
        };

        let json_str = after_open[..close_pos].trim();
        if !json_str.is_empty() {
            results.push(json_str);
        }

        // Advance past the closing fence to look for more blocks
        search_from += content_start + close_pos + "```".len();
    }

    results
}

/// Scan text for JSON objects `{...}` embedded anywhere in prose.
///
/// Uses brace-depth counting to find balanced `{...}` substrings, then
/// returns only those that look like valid JSON (parse successfully).
/// This handles the case where a model writes:
///   "Let's check. {"name": "shell_exec", "arguments": {"command": "rustc --version"}}"
fn extract_embedded_json_objects(text: &str) -> Vec<&str> {
    let mut results = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0;

    while i < chars.len() {
        let (byte_start, ch) = chars[i];
        if ch == '{' {
            // Count braces to find the matching closing brace.
            let mut depth = 0;
            let mut in_string = false;
            let mut escape_next = false;
            let mut end_byte = None;

            for &(byte_pos, c) in &chars[i..] {
                if escape_next {
                    escape_next = false;
                    continue;
                }
                if c == '\\' && in_string {
                    escape_next = true;
                    continue;
                }
                if c == '"' {
                    in_string = !in_string;
                    continue;
                }
                if in_string {
                    continue;
                }
                if c == '{' {
                    depth += 1;
                } else if c == '}' {
                    depth -= 1;
                    if depth == 0 {
                        end_byte = Some(byte_pos);
                        break;
                    }
                }
            }

            if let Some(end) = end_byte {
                let candidate = &text[byte_start..=end];
                // Only accept if it parses as valid JSON and has a "name" field
                // (quick pre-check to avoid expensive parsing of random braces).
                if candidate.contains("\"name\"")
                    && serde_json::from_str::<serde_json::Value>(candidate).is_ok()
                {
                    results.push(candidate);
                    // Skip past this object to avoid overlapping matches.
                    i = chars
                        .iter()
                        .position(|&(b, _)| b > end)
                        .unwrap_or(chars.len());
                    continue;
                }
            }
        }
        i += 1;
    }

    results
}

/// Parse a single tool call JSON object: `{"name": "...", "arguments": {...}}`
/// Only succeeds if the name matches a registered tool (prevents false positives).
fn parse_single_tool_call(json_str: &str, registry: &ToolRegistry) -> Option<ToolCall> {
    let value: serde_json::Value = serde_json::from_str(json_str).ok()?;

    // Format 1: {"name": "tool_name", "arguments": {...}}
    if let Some(name) = value.get("name").and_then(|n| n.as_str()) {
        if registry.get(name).is_some() {
            let arguments = value
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            return Some(ToolCall {
                id: None,
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            });
        }
    }

    // Format 2: {"function": {"name": "tool_name", "arguments": {...}}}
    // (OpenAI-style, some models use this)
    if let Some(func) = value.get("function") {
        if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
            if registry.get(name).is_some() {
                let arguments = func
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                return Some(ToolCall {
                    id: value.get("id").and_then(|i| i.as_str()).map(String::from),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments,
                    },
                });
            }
        }
    }

    None
}

/// Parse a JSON array of tool calls: `[{"name": "...", ...}, ...]`
fn parse_tool_call_array(json_str: &str, registry: &ToolRegistry) -> Option<Vec<ToolCall>> {
    let arr = serde_json::from_str::<Vec<serde_json::Value>>(json_str).ok()?;
    let calls: Vec<ToolCall> = arr
        .iter()
        .filter_map(|v| {
            let name = v.get("name")?.as_str()?;
            if registry.get(name).is_none() {
                return None;
            }
            let arguments = v.get("arguments").cloned().unwrap_or(serde_json::json!({}));
            Some(ToolCall {
                id: None,
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            })
        })
        .collect();
    if calls.is_empty() { None } else { Some(calls) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::message::ToolDef;
    use crate::tools::traits::{Tool, ToolResult};
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -----------------------------------------------------------------------
    // Mock LLM Backend
    // -----------------------------------------------------------------------

    /// A mock LLM backend that returns pre-configured responses.
    ///
    /// This is essential for deterministic testing of the agent loop without
    /// needing a running Ollama instance (as recommended by PLAN_CRITIQUE.md).
    struct MockBackend {
        /// Queue of responses to return, in order. Each call to `chat()`
        /// pops the next response from the front.
        responses: std::sync::Mutex<Vec<Message>>,
    }

    impl MockBackend {
        fn new(responses: Vec<Message>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
            }
        }
    }

    #[allow(async_fn_in_trait)]
    impl LlmBackend for MockBackend {
        async fn chat(&self, _messages: &[Message], _tools: &[ToolDef]) -> Result<Message> {
            let mut queue = self.responses.lock().unwrap();
            if queue.is_empty() {
                anyhow::bail!("MockBackend: no more responses in queue");
            }
            Ok(queue.remove(0))
        }
    }

    // -----------------------------------------------------------------------
    // Mock Tool
    // -----------------------------------------------------------------------

    /// A mock tool that records calls and returns a fixed output.
    struct MockTool {
        tool_name: &'static str,
        output: &'static str,
        call_count: Arc<AtomicUsize>,
    }

    impl MockTool {
        fn new(name: &'static str, output: &'static str) -> (Self, Arc<AtomicUsize>) {
            let counter = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    tool_name: name,
                    output,
                    call_count: counter.clone(),
                },
                counter,
            )
        }
    }

    impl Tool for MockTool {
        fn name(&self) -> &str {
            self.tool_name
        }

        fn description(&self) -> &str {
            "A mock tool for testing"
        }

        fn parameters_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        }

        fn execute(
            &self,
            _args: Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let output = self.output.to_string();
            Box::pin(async move { Ok(ToolResult::success(output)) })
        }
    }

    // -----------------------------------------------------------------------
    // Helper: build a text-only assistant response
    // -----------------------------------------------------------------------

    fn text_response(content: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Build an assistant response that requests a tool call.
    fn tool_call_response(tool_name: &str, args: Value) -> Message {
        Message {
            role: Role::Assistant,
            content: String::new(),
            tool_calls: Some(vec![crate::agent::message::ToolCall {
                id: None,
                function: crate::agent::message::FunctionCall {
                    name: tool_name.to_string(),
                    arguments: args,
                },
            }]),
            tool_call_id: None,
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_simple_text_response() {
        // LLM responds with plain text, no tools involved.
        let backend = MockBackend::new(vec![text_response("Hello from the LLM!")]);
        let registry = ToolRegistry::new();
        let mut agent = AgentLoop::new(backend, registry, "You are helpful.", 10);

        let result = agent.process_message("Hi").await.unwrap();
        assert_eq!(result, "Hello from the LLM!");
    }

    #[tokio::test]
    async fn test_single_tool_call_then_text() {
        // LLM calls a tool once, then responds with text.
        let backend = MockBackend::new(vec![
            tool_call_response("mock_tool", serde_json::json!({})),
            text_response("Done! The tool returned its output."),
        ]);

        let (mock_tool, call_count) = MockTool::new("mock_tool", "tool output here");
        let mut registry = ToolRegistry::new();
        registry.register(mock_tool);

        let mut agent = AgentLoop::new(backend, registry, "You are helpful.", 10);
        let result = agent.process_message("Do something").await.unwrap();

        assert_eq!(result, "Done! The tool returned its output.");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_multiple_tool_calls_then_text() {
        // LLM calls tools twice (two iterations), then responds.
        let backend = MockBackend::new(vec![
            tool_call_response("tool_a", serde_json::json!({})),
            tool_call_response("tool_b", serde_json::json!({})),
            text_response("All done."),
        ]);

        let (tool_a, count_a) = MockTool::new("tool_a", "output a");
        let (tool_b, count_b) = MockTool::new("tool_b", "output b");

        let mut registry = ToolRegistry::new();
        registry.register(tool_a);
        registry.register(tool_b);

        let mut agent = AgentLoop::new(backend, registry, "You are helpful.", 10);
        let result = agent.process_message("Do two things").await.unwrap();

        assert_eq!(result, "All done.");
        assert_eq!(count_a.load(Ordering::SeqCst), 1);
        assert_eq!(count_b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_unknown_tool_name() {
        // LLM calls a tool that doesn't exist. The error should be fed back.
        let backend = MockBackend::new(vec![
            tool_call_response("nonexistent_tool", serde_json::json!({})),
            text_response("Sorry, that tool doesn't exist."),
        ]);

        let registry = ToolRegistry::new(); // empty — no tools registered
        let mut agent = AgentLoop::new(backend, registry, "You are helpful.", 10);

        let result = agent.process_message("Use a tool").await.unwrap();
        assert_eq!(result, "Sorry, that tool doesn't exist.");

        // Verify the conversation includes the error message fed back.
        let messages = agent.conversation().messages();
        let tool_msg = messages
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("Should have a tool result message");
        assert!(tool_msg.content.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_max_iterations_safety_limit() {
        // LLM keeps requesting tools forever. The loop should stop.
        let mut responses = Vec::new();
        for _ in 0..25 {
            responses.push(tool_call_response("mock_tool", serde_json::json!({})));
        }
        let backend = MockBackend::new(responses);

        let (mock_tool, call_count) = MockTool::new("mock_tool", "output");
        let mut registry = ToolRegistry::new();
        registry.register(mock_tool);

        let mut agent = AgentLoop::new(backend, registry, "You are helpful.", 5);
        let result = agent.process_message("Loop forever").await.unwrap();

        // Should hit the safety limit. The last assistant message has empty
        // content (tool-call-only responses), so we get the "tool-calling cycle" branch.
        assert!(
            result.contains("iterations") || result.contains("tool-calling cycle"),
            "Expected iteration limit message, got: {result}"
        );
        // Should have made exactly max_iterations tool calls
        assert_eq!(call_count.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn test_conversation_history_preserved() {
        let backend = MockBackend::new(vec![
            text_response("First response."),
            text_response("Second response."),
        ]);

        let registry = ToolRegistry::new();
        let mut agent = AgentLoop::new(backend, registry, "System prompt.", 10);

        agent.process_message("First message").await.unwrap();
        agent.process_message("Second message").await.unwrap();

        let messages = agent.conversation().messages();
        // System + user1 + assistant1 + user2 + assistant2 = 5
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[3].role, Role::User);
        assert_eq!(messages[4].role, Role::Assistant);
    }

    #[tokio::test]
    async fn test_clear_conversation() {
        let backend = MockBackend::new(vec![text_response("Hello.")]);

        let registry = ToolRegistry::new();
        let mut agent = AgentLoop::new(backend, registry, "System prompt.", 10);

        agent.process_message("Hi").await.unwrap();
        assert_eq!(agent.conversation().messages().len(), 3); // system + user + assistant

        agent.clear_conversation("New system prompt.");
        assert_eq!(agent.conversation().messages().len(), 1); // just the new system prompt
        assert_eq!(agent.conversation().messages()[0].role, Role::System);
        assert_eq!(
            agent.conversation().messages()[0].content,
            "New system prompt."
        );
    }

    // -----------------------------------------------------------------------
    // Fallback tool-call parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_tool_call_from_json_codeblock() {
        let (mock_tool, _) = MockTool::new("file_write", "ok");
        let mut registry = ToolRegistry::new();
        registry.register(mock_tool);

        let content = r#"```json
{
  "name": "file_write",
  "arguments": {
    "path": "main.rs",
    "content": "fn main() {}"
  }
}
```"#;

        let result = try_parse_tool_call_from_content(content, &registry);
        assert!(result.is_some(), "Should parse tool call from code block");
        let calls = result.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "file_write");
        assert_eq!(calls[0].function.arguments["path"], "main.rs");
    }

    #[test]
    fn test_parse_tool_call_from_bare_json() {
        let (mock_tool, _) = MockTool::new("shell_exec", "ok");
        let mut registry = ToolRegistry::new();
        registry.register(mock_tool);

        let content = r#"{"name": "shell_exec", "arguments": {"command": "cargo build"}}"#;

        let result = try_parse_tool_call_from_content(content, &registry);
        assert!(result.is_some());
        let calls = result.unwrap();
        assert_eq!(calls[0].function.name, "shell_exec");
        assert_eq!(calls[0].function.arguments["command"], "cargo build");
    }

    /// This is the exact pattern qwen2.5-coder:7b produced: a bare JSON tool
    /// call embedded in the middle of a prose sentence with no code fences.
    #[test]
    fn test_parse_tool_call_embedded_in_prose() {
        let (mock_tool, _) = MockTool::new("shell_exec", "ok");
        let mut registry = ToolRegistry::new();
        registry.register(mock_tool);

        let content = r#"It seems there was an issue running the Rust compiler. Let's make sure Rust is installed and properly configured on your system.

{"name": "shell_exec", "arguments": {"command": "rustc --version"}}"#;

        let result = try_parse_tool_call_from_content(content, &registry);
        assert!(
            result.is_some(),
            "Should extract tool call embedded in prose without code fences"
        );
        let calls = result.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "shell_exec");
        assert_eq!(calls[0].function.arguments["command"], "rustc --version");
    }

    /// Multiple bare JSON tool calls in prose, no fences at all.
    #[test]
    fn test_parse_multiple_bare_tool_calls_in_prose() {
        let (tool_w, _) = MockTool::new("file_write", "ok");
        let (tool_s, _) = MockTool::new("shell_exec", "ok");
        let mut registry = ToolRegistry::new();
        registry.register(tool_w);
        registry.register(tool_s);

        let content = r#"First, let me write the file. {"name": "file_write", "arguments": {"path": "main.rs", "content": "fn main() {}"}} Now let's compile it. {"name": "shell_exec", "arguments": {"command": "rustc main.rs"}}"#;

        let result = try_parse_tool_call_from_content(content, &registry);
        assert!(result.is_some());
        let calls = result.unwrap();
        assert_eq!(calls.len(), 2, "Should find both tool calls");
        assert_eq!(calls[0].function.name, "file_write");
        assert_eq!(calls[1].function.name, "shell_exec");
    }

    #[test]
    fn test_extract_embedded_json_objects() {
        // extract_embedded_json_objects requires a "name" field to avoid
        // false positives on arbitrary JSON in prose.
        let text =
            r#"Some text {"name": "alpha", "value": 1} more text {"name": "beta", "value": 2} end"#;
        let objects = extract_embedded_json_objects(text);
        assert_eq!(objects.len(), 2);
        assert!(objects[0].contains("alpha"));
        assert!(objects[1].contains("beta"));
    }

    #[test]
    fn test_extract_embedded_json_nested_braces() {
        let text = r#"Here: {"name": "file_write", "arguments": {"path": "test.rs", "content": "fn main() {}"}} done"#;
        let objects = extract_embedded_json_objects(text);
        assert_eq!(objects.len(), 1);
        assert!(objects[0].contains("file_write"));
    }

    #[test]
    fn test_parse_tool_call_ignores_unknown_tool() {
        let registry = ToolRegistry::new(); // empty — no tools registered

        let content = r#"{"name": "unknown_tool", "arguments": {}}"#;

        let result = try_parse_tool_call_from_content(content, &registry);
        assert!(
            result.is_none(),
            "Should not match JSON with unregistered tool name"
        );
    }

    #[test]
    fn test_parse_tool_call_ignores_plain_text() {
        let (mock_tool, _) = MockTool::new("file_read", "ok");
        let mut registry = ToolRegistry::new();
        registry.register(mock_tool);

        let content = "Here's how you can read a file using the file_read tool.";

        let result = try_parse_tool_call_from_content(content, &registry);
        assert!(result.is_none(), "Should not match plain English text");
    }

    #[test]
    fn test_parse_tool_call_openai_format() {
        let (mock_tool, _) = MockTool::new("file_read", "ok");
        let mut registry = ToolRegistry::new();
        registry.register(mock_tool);

        let content = r#"{"function": {"name": "file_read", "arguments": {"path": "Cargo.toml"}}}"#;

        let result = try_parse_tool_call_from_content(content, &registry);
        assert!(result.is_some(), "Should parse OpenAI-style format");
        let calls = result.unwrap();
        assert_eq!(calls[0].function.name, "file_read");
    }

    /// This is the key test: a model that embeds MULTIPLE tool calls in prose,
    /// each in its own ```json code block with explanatory text between them.
    /// This is exactly what qwen2.5-coder:7b does.
    #[test]
    fn test_parse_multiple_tool_calls_in_prose() {
        let (tool_write, _) = MockTool::new("file_write", "ok");
        let (tool_shell, _) = MockTool::new("shell_exec", "ok");
        let mut registry = ToolRegistry::new();
        registry.register(tool_write);
        registry.register(tool_shell);

        let content = r#"Sure! Let's create the file.

```json
{
  "name": "file_write",
  "arguments": {
    "path": "main.rs",
    "content": "fn main() {\n    println!(\"Hello, world!\");\n}"
  }
}
```

Now let's compile and run it.

```json
{
  "name": "shell_exec",
  "arguments": {
    "command": "rustc main.rs && ./main"
  }
}
```

This will compile and run the program."#;

        let result = try_parse_tool_call_from_content(content, &registry);
        assert!(
            result.is_some(),
            "Should extract both tool calls from prose"
        );
        let calls = result.unwrap();
        assert_eq!(calls.len(), 2, "Should find exactly 2 tool calls");
        assert_eq!(calls[0].function.name, "file_write");
        assert_eq!(calls[0].function.arguments["path"], "main.rs");
        assert_eq!(calls[1].function.name, "shell_exec");
        assert_eq!(
            calls[1].function.arguments["command"],
            "rustc main.rs && ./main"
        );
    }

    /// Test that code blocks containing non-tool JSON are ignored.
    #[test]
    fn test_parse_ignores_non_tool_json_in_codeblocks() {
        let (mock_tool, _) = MockTool::new("file_write", "ok");
        let mut registry = ToolRegistry::new();
        registry.register(mock_tool);

        let content = r#"Here's an example config:

```json
{
  "database": "postgres",
  "port": 5432
}
```

That's not a tool call."#;

        let result = try_parse_tool_call_from_content(content, &registry);
        assert!(
            result.is_none(),
            "Should not match JSON that isn't a tool call"
        );
    }

    #[test]
    fn test_extract_all_json_codeblocks() {
        let text = r#"Some text
```json
{"first": true}
```
middle text
```json
{"second": true}
```
end text"#;

        let blocks = extract_all_json_codeblocks(text);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].contains("first"));
        assert!(blocks[1].contains("second"));
    }

    #[test]
    fn test_extract_all_json_codeblocks_none() {
        let text = "Just plain text with no code blocks at all.";
        let blocks = extract_all_json_codeblocks(text);
        assert!(blocks.is_empty());
    }

    #[tokio::test]
    async fn test_text_fallback_tool_call_executed() {
        // Simulate a model that outputs tool calls as text instead of
        // using the structured format. The agent loop should detect this,
        // execute the tool, and loop back to the LLM.
        let backend = MockBackend::new(vec![
            // First response: model outputs a tool call as text
            Message {
                role: Role::Assistant,
                content: r#"```json
{"name": "mock_tool", "arguments": {}}
```"#
                    .to_string(),
                tool_calls: None, // no structured tool_calls!
                tool_call_id: None,
            },
            // Second response: model gives a final text answer
            text_response("Done! I used the tool."),
        ]);

        let (mock_tool, call_count) = MockTool::new("mock_tool", "tool output here");
        let mut registry = ToolRegistry::new();
        registry.register(mock_tool);

        let mut agent = AgentLoop::new(backend, registry, "You are helpful.", 10);
        let result = agent.process_message("Do something").await.unwrap();

        assert_eq!(result, "Done! I used the tool.");
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "Tool should have been called once via fallback parsing"
        );
    }

    /// End-to-end test: model embeds two tool calls in prose (like qwen2.5-coder:7b).
    /// Both tools should be executed, results fed back, then the model responds.
    #[tokio::test]
    async fn test_multiple_text_fallback_tools_executed() {
        let backend = MockBackend::new(vec![
            // First response: model outputs TWO tool calls in prose
            Message {
                role: Role::Assistant,
                content: r#"Let me write the file and compile it.

```json
{"name": "tool_a", "arguments": {}}
```

Now compile:

```json
{"name": "tool_b", "arguments": {}}
```"#
                    .to_string(),
                tool_calls: None,
                tool_call_id: None,
            },
            // Second response: model gives a final text answer
            text_response("Both tools ran successfully!"),
        ]);

        let (tool_a, count_a) = MockTool::new("tool_a", "wrote the file");
        let (tool_b, count_b) = MockTool::new("tool_b", "compiled ok");
        let mut registry = ToolRegistry::new();
        registry.register(tool_a);
        registry.register(tool_b);

        let mut agent = AgentLoop::new(backend, registry, "You are helpful.", 10);
        let result = agent.process_message("Write and compile").await.unwrap();

        assert_eq!(result, "Both tools ran successfully!");
        assert_eq!(count_a.load(Ordering::SeqCst), 1, "tool_a should run once");
        assert_eq!(count_b.load(Ordering::SeqCst), 1, "tool_b should run once");
    }
}
