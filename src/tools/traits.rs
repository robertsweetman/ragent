//! The `Tool` trait and `ToolResult` type — the foundation of ragent's tool system.
//!
//! # Phase 2 (PLAN.md)
//! Goal: "Agent autonomously uses tools to accomplish multi-step tasks."
//!
//! # Design Decision: Why `Pin<Box<dyn Future>>` instead of `async fn`?
//!
//! We need `Box<dyn Tool>` in the `ToolRegistry` so that different tool types
//! (file ops, shell, web fetch, etc.) can live in the same collection. This
//! requires the trait to be **object-safe**.
//!
//! `async fn` in traits is NOT object-safe because each implementation returns
//! a different anonymous `Future` type, and `dyn Trait` needs a single concrete
//! type for each method's return.
//!
//! The standard solution is to return a **boxed, type-erased future**:
//!
//! ```text
//!   async fn execute(...)          →  compiler generates an opaque Future type per impl
//!   fn execute(...) -> Pin<Box<    →  all impls return the same type: a heap-allocated,
//!     dyn Future<...> + Send       →  Send-safe (works across tokio tasks),
//!     + '_                         →  that borrows from &self
//!   >>
//! ```
//!
//! This is exactly what the `async_trait` proc macro does under the hood. We do
//! it manually here so you can see the mechanism. Each tool implementation wraps
//! its async block in `Box::pin(async move { ... })`.
//!
//! The `'_` lifetime is an anonymous lifetime that captures `&self`, allowing
//! the future to borrow data from the tool struct (e.g. config, working directory).

use anyhow::Result;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

// ---------------------------------------------------------------------------
// ToolResult
// ---------------------------------------------------------------------------

/// The outcome of executing a tool.
///
/// This is intentionally **not** a `Result` type. Both success and failure are
/// represented as `ToolResult` values that get fed back to the LLM as text.
/// The LLM can then decide how to proceed (retry, try a different approach,
/// or report the error to the user).
///
/// The `Result<ToolResult>` return type on `Tool::execute` means:
/// - `Ok(ToolResult { is_error: false, .. })` → tool succeeded, content is the output
/// - `Ok(ToolResult { is_error: true, .. })`  → tool failed gracefully, content explains why
/// - `Err(...)`                               → infrastructure failure (e.g. timeout, panic) —
///                                              the agent loop handles this, not the LLM
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The text content returned by the tool. For successful operations this is
    /// the output (file contents, command output, etc.). For errors, this is a
    /// human-readable error message that the LLM can understand and act on.
    pub content: String,

    /// Whether this result represents an error condition.
    /// When `true`, the agent loop will still feed the content back to the LLM
    /// (so it can recover), but may also log it as a warning.
    pub is_error: bool,
}

impl ToolResult {
    /// Create a successful tool result.
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// Create an error tool result.
    ///
    /// The error message should be descriptive enough for the LLM to understand
    /// what went wrong and potentially try a different approach.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: message.into(),
            is_error: true,
        }
    }
}

// For convenience: display the content directly
impl std::fmt::Display for ToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_error {
            write!(f, "[ERROR] {}", self.content)
        } else {
            write!(f, "{}", self.content)
        }
    }
}

// ---------------------------------------------------------------------------
// Tool trait
// ---------------------------------------------------------------------------

/// A tool that the LLM can invoke to interact with the outside world.
///
/// Each tool has a name, description, and JSON Schema for its parameters.
/// The LLM sees these definitions and can choose to call tools by name,
/// passing arguments that match the schema.
///
/// # Implementing a Tool
///
/// ```rust,ignore
/// use ragent::tools::traits::{Tool, ToolResult};
/// use serde_json::Value;
/// use std::pin::Pin;
/// use std::future::Future;
/// use anyhow::Result;
///
/// pub struct MyTool;
///
/// impl Tool for MyTool {
///     fn name(&self) -> &str { "my_tool" }
///
///     fn description(&self) -> &str { "Does something useful" }
///
///     fn parameters_schema(&self) -> Value {
///         serde_json::json!({
///             "type": "object",
///             "properties": {
///                 "input": { "type": "string", "description": "The input" }
///             },
///             "required": ["input"]
///         })
///     }
///
///     fn execute(&self, args: Value) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
///         Box::pin(async move {
///             let input = args["input"].as_str().unwrap_or_default();
///             Ok(ToolResult::success(format!("Processed: {input}")))
///         })
///     }
/// }
/// ```
///
/// # Object Safety
///
/// This trait is object-safe: you can use `Box<dyn Tool>` in the `ToolRegistry`.
/// The `execute` method returns a boxed future instead of using `async fn` to
/// make this possible. See the module-level docs for the full explanation.
pub trait Tool: Send + Sync {
    /// The unique name of this tool, used by the LLM to invoke it.
    ///
    /// Should be lowercase, snake_case (e.g. `"file_read"`, `"shell_exec"`).
    /// Must be unique within a `ToolRegistry`.
    fn name(&self) -> &str;

    /// A human-readable description of what this tool does.
    ///
    /// This is sent to the LLM so it knows when and why to use the tool.
    /// Be specific — the better the description, the better the LLM's
    /// tool-use decisions.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's parameters.
    ///
    /// Must be a valid JSON Schema object (type "object" with "properties").
    /// This is sent to the LLM as part of the tool definition so it knows
    /// what arguments to pass.
    ///
    /// Example:
    /// ```json
    /// {
    ///   "type": "object",
    ///   "properties": {
    ///     "path": { "type": "string", "description": "File path to read" }
    ///   },
    ///   "required": ["path"]
    /// }
    /// ```
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given arguments.
    ///
    /// # Arguments
    /// * `args` — A JSON value matching the schema from `parameters_schema()`.
    ///            The agent loop extracts this from the LLM's tool call.
    ///
    /// # Returns
    /// * `Ok(ToolResult)` — Tool completed (successfully or with a handled error).
    ///   The result content is fed back to the LLM.
    /// * `Err(...)` — Infrastructure failure. The agent loop will handle this
    ///   (e.g. log it, possibly abort the loop).
    ///
    /// # Why `Pin<Box<dyn Future>>` instead of `async fn`?
    ///
    /// This signature makes the trait object-safe so we can store `Box<dyn Tool>`
    /// in the registry. Each implementation wraps its async block in
    /// `Box::pin(async move { ... })`. See the module docs for the full story.
    fn execute(&self, args: Value)
    -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("file contents here");
        assert!(!result.is_error);
        assert_eq!(result.content, "file contents here");
        assert_eq!(format!("{result}"), "file contents here");
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("file not found: foo.txt");
        assert!(result.is_error);
        assert_eq!(result.content, "file not found: foo.txt");
        assert_eq!(format!("{result}"), "[ERROR] file not found: foo.txt");
    }

    /// A minimal tool implementation to verify the trait is object-safe.
    struct DummyTool;

    impl Tool for DummyTool {
        fn name(&self) -> &str {
            "dummy"
        }

        fn description(&self) -> &str {
            "A dummy tool for testing"
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
            Box::pin(async move { Ok(ToolResult::success("dummy output")) })
        }
    }

    #[test]
    fn test_trait_is_object_safe() {
        // This compiles only if Tool is object-safe — the whole point of
        // using Pin<Box<dyn Future>> instead of async fn.
        let tool: Box<dyn Tool> = Box::new(DummyTool);
        assert_eq!(tool.name(), "dummy");
        assert_eq!(tool.description(), "A dummy tool for testing");
    }

    #[tokio::test]
    async fn test_dummy_tool_execute() {
        let tool = DummyTool;
        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content, "dummy output");
    }

    #[tokio::test]
    async fn test_dyn_tool_execute() {
        // Verify we can execute through a trait object (Box<dyn Tool>)
        let tool: Box<dyn Tool> = Box::new(DummyTool);
        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert_eq!(result.content, "dummy output");
    }
}
