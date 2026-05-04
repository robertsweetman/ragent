//! OpenAI-compatible LLM backend.
//!
//! Speaks the `/v1/chat/completions` API that is shared by OpenAI, LM Studio,
//! Jan, Mistral, and many other inference servers. Use this backend when you
//! want to leverage GPU acceleration on Windows via LM Studio + DirectML, or
//! when targeting a remote OpenAI/Anthropic API.
//!
//! # Key differences from the Ollama backend
//!
//! | Detail | Ollama | OpenAI |
//! |--------|--------|--------|
//! | Endpoint | `/api/chat` | `/v1/chat/completions` |
//! | Response wrapper | `message` | `choices[0].message` |
//! | Tool call args | JSON **object** | JSON **string** (must be parsed) |
//! | Streaming format | raw NDJSON | SSE (`data: {...}`, ends with `data: [DONE]`) |
//! | Streaming tool args | whole object in one chunk | fragments accumulated per `index` |

use crate::agent::message::{FunctionCall, Message, Role, ToolCall, ToolDef};
use crate::config::OpenAiConfig;
use crate::llm::traits::LlmBackend;
use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, instrument, warn};

// ---------------------------------------------------------------------------
// OpenAI wire message format
// ---------------------------------------------------------------------------
//
// The internal `Message` type is shaped for Ollama. The OpenAI wire format
// differs in three important ways that cause HTTP 400 errors if ignored:
//
//   1. `arguments` must be a JSON *string* (e.g. `"{\"path\":\"foo.rs\"}"`).
//      Ollama and our internal type store it as a `serde_json::Value` object.
//
//   2. Each tool call in the assistant message must carry `"type": "function"`.
//      Ollama omits this field entirely.
//
//   3. `content` must be `null` (not `""`) when the assistant message only
//      contains tool calls. Some servers are strict about this.
//
// We solve this with a small set of wire-only types and a conversion function.
// These types are never stored — they exist solely for serialisation.

/// OpenAI-wire representation of a single conversation message.
///
/// Uses serde's internal tagging (`#[serde(tag = "role")]`) so each variant
/// serialises to `{"role": "...", ...rest}` — exactly what the API expects.
#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
enum WireMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        /// `None` serialises as `null`, which OpenAI requires when tool_calls
        /// are present. An empty string `""` is rejected by strict servers.
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<WireToolCall>>,
    },
    Tool {
        content: String,
        /// Must match the `id` from the assistant's tool_call that triggered
        /// this result. OpenAI validates this pairing; Ollama doesn't use IDs.
        tool_call_id: String,
    },
}

/// A tool call as it appears in an assistant message on the wire.
#[derive(Debug, Serialize)]
struct WireToolCall {
    id: String,
    /// OpenAI always requires `"type": "function"`. Ollama omits this field.
    #[serde(rename = "type")]
    call_type: &'static str,
    function: WireFunction,
}

/// Function name + arguments in the wire format.
#[derive(Debug, Serialize)]
struct WireFunction {
    name: String,
    /// OpenAI expects a JSON-encoded *string*, e.g. `"{\"path\":\"foo.rs\"}"`,
    /// not a JSON object. We convert with `serde_json::Value::to_string()`.
    arguments: String,
}

/// Convert our internal `Message` slice into OpenAI-compatible wire messages.
///
/// This is the translation layer between ragent's Ollama-shaped internal
/// format and the stricter OpenAI wire format. Called once per chat request.
fn to_wire_messages(messages: &[Message]) -> Vec<WireMessage> {
    messages
        .iter()
        .enumerate()
        .map(|(_, msg)| match msg.role {
            Role::System => WireMessage::System {
                content: msg.content.clone(),
            },
            Role::User => WireMessage::User {
                content: msg.content.clone(),
            },
            Role::Assistant => {
                let tool_calls = msg.tool_calls.as_ref().map(|tcs| {
                    tcs.iter()
                        .enumerate()
                        .map(|(i, tc)| WireToolCall {
                            // Fall back to a generated ID if the model didn't
                            // provide one (Ollama never does).
                            id: tc.id.clone().unwrap_or_else(|| format!("call_{i}")),
                            call_type: "function",
                            function: WireFunction {
                                name: tc.function.name.clone(),
                                // Convert Value -> JSON string as OpenAI requires.
                                arguments: tc.function.arguments.to_string(),
                            },
                        })
                        .collect()
                });
                WireMessage::Assistant {
                    // Use None (→ null) rather than Some("") when content is
                    // empty and tool calls are present — OpenAI is strict here.
                    content: if msg.content.is_empty() {
                        None
                    } else {
                        Some(msg.content.clone())
                    },
                    tool_calls,
                }
            }
            Role::Tool => WireMessage::Tool {
                content: msg.content.clone(),
                tool_call_id: msg.tool_call_id.clone().unwrap_or_default(),
            },
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// POST /v1/chat/completions request body.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    /// Pre-converted wire messages (see `to_wire_messages`).
    messages: Vec<WireMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDef]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

/// Non-streaming response.
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ResponseChoice>,
}

#[derive(Debug, Deserialize)]
struct ResponseChoice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

/// A tool call in OpenAI non-streaming format.
///
/// Note that `arguments` is a **JSON string**, not a JSON object.
/// We parse it into `serde_json::Value` when converting to our internal type.
#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    #[serde(default)]
    id: Option<String>,
    function: OpenAiFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunction {
    name: String,
    /// A JSON-encoded string, e.g. `"{\"path\":\"src/main.rs\"}"`.
    #[serde(default)]
    arguments: String,
}

// ---------------------------------------------------------------------------
// Streaming types
// ---------------------------------------------------------------------------

/// A single SSE chunk: `data: <json>\n\n`
#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Delta,
    #[allow(dead_code)]
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    /// Tool call fragments arrive per `index`. Each chunk adds a fragment of
    /// the `arguments` string. We accumulate them and parse at the end.
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct ToolCallDelta {
    /// Which tool call slot this fragment belongs to.
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    /// Fragment of the JSON-encoded arguments string.
    #[serde(default)]
    arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool-call accumulator (streaming)
// ---------------------------------------------------------------------------

/// Accumulates the fragments of a single tool call during streaming.
///
/// In the OpenAI streaming format, the `id` and `name` arrive in the first
/// chunk for that index, and then subsequent chunks keep appending to
/// `arguments` until `finish_reason` signals completion.
#[derive(Default)]
struct ToolCallBuilder {
    id: Option<String>,
    name: String,
    /// Concatenated JSON-string fragments for the arguments.
    arguments: String,
}

impl ToolCallBuilder {
    /// Convert to our internal `ToolCall`, parsing the accumulated argument
    /// string into a `serde_json::Value`.
    fn finish(self) -> ToolCall {
        let arguments =
            serde_json::from_str::<serde_json::Value>(&self.arguments).unwrap_or_else(|_| {
                // If parsing fails (model produced malformed JSON), wrap in an
                // object with a `raw` key so the agent can still see something.
                serde_json::json!({ "raw": self.arguments })
            });

        ToolCall {
            id: self.id,
            function: FunctionCall {
                name: self.name,
                arguments,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAiBackend
// ---------------------------------------------------------------------------

/// An LLM backend that speaks the OpenAI `/v1/chat/completions` API.
///
/// Compatible with LM Studio, Jan, OpenAI, Mistral, and other
/// OpenAI-compatible inference servers.
pub struct OpenAiBackend {
    client: Client,
    config: OpenAiConfig,
}

impl OpenAiBackend {
    /// Create a new backend from config.
    ///
    /// Does NOT check connectivity — call `check_connection()` for that.
    pub fn new(config: OpenAiConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        info!(
            url = %config.url,
            model = %config.model,
            request_timeout_secs = config.request_timeout_secs,
            "Created OpenAI-compatible backend"
        );

        Self { client, config }
    }

    /// Check that the server is reachable by hitting `/v1/models`.
    ///
    /// Most OpenAI-compatible servers (including LM Studio) implement this
    /// endpoint. A 200 means the server is up; anything else is treated as an
    /// error. Returns a descriptive message if the server is unreachable.
    pub async fn check_connection(&self) -> Result<()> {
        let url = format!("{}/models", self.config.url);
        debug!(url = %url, "Checking OpenAI-compatible backend connection");

        let mut req = self.client.get(&url);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                info!(url = %url, "OpenAI-compatible backend is reachable");
                Ok(())
            }
            Ok(resp) => {
                anyhow::bail!(
                    "OpenAI-compatible server at {} returned HTTP {}.\n  \
                     Is LM Studio running and is the API server enabled?",
                    self.config.url,
                    resp.status()
                )
            }
            Err(_) => {
                anyhow::bail!(
                    "Cannot connect to OpenAI-compatible server at {}.\n  \
                     If using LM Studio: open it, load a model, then enable\n  \
                     the local server under Developer → Local Server.",
                    self.config.url
                )
            }
        }
    }

    /// Get the configured model name.
    pub fn model(&self) -> &str {
        &self.config.model
    }
}

impl LlmBackend for OpenAiBackend {
    /// Send a chat completion request, optionally streaming tokens.
    ///
    /// The OpenAI streaming format uses Server-Sent Events (SSE):
    /// each line looks like `data: <json>` and the stream ends with
    /// `data: [DONE]`. We strip the prefix and parse each JSON object.
    ///
    /// Tool call arguments arrive as JSON-string **fragments** across multiple
    /// chunks (one fragment per `index` slot). We accumulate all fragments,
    /// then parse the complete string into a `serde_json::Value` at the end.
    #[instrument(skip(self, messages, tools, on_token), fields(model = %self.config.model, msg_count = messages.len()))]
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        on_token: Option<&dyn Fn(&str)>,
    ) -> Result<Message> {
        let url = format!("{}/chat/completions", self.config.url);

        let request = ChatRequest {
            model: &self.config.model,
            messages: to_wire_messages(messages),
            stream: on_token.is_some(),
            tools: if tools.is_empty() { None } else { Some(tools) },
            temperature: self.config.temperature,
        };

        debug!(
            request = %serde_json::to_string_pretty(&request).unwrap_or_default(),
            "Sending chat request to OpenAI-compatible backend"
        );

        let mut req = self.client.post(&url).json(&request);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }

        let response = req
            .send()
            .await
            .context("Failed to send request to OpenAI backend")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            warn!(status = %status, body = %body, "OpenAI backend returned error");
            anyhow::bail!("OpenAI backend returned HTTP {}: {}", status, body);
        }

        // -----------------------------------------------------------------------
        // Branch: streaming vs non-streaming
        // -----------------------------------------------------------------------

        if let Some(cb) = on_token {
            // --- Streaming path ---
            //
            // OpenAI SSE format:
            //   data: {"choices":[{"delta":{"content":"Hello"},...}]}
            //   data: {"choices":[{"delta":{"content":"!"},...}]}
            //   data: [DONE]
            //
            // Tool call arguments come as accumulated string fragments across
            // multiple chunks, keyed by `index`. We collect them all, then
            // parse each builder's argument string into JSON at the end.
            let mut full_content = String::new();
            let mut builders: Vec<ToolCallBuilder> = Vec::new();
            let mut line_buffer = String::new();
            let chunk_timeout = Duration::from_secs(self.config.request_timeout_secs);

            let mut byte_stream = response.bytes_stream();
            'stream: loop {
                let maybe_chunk = tokio::time::timeout(chunk_timeout, byte_stream.next())
                    .await
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "Streaming timed out: no data from OpenAI backend for {} seconds.",
                            chunk_timeout.as_secs()
                        )
                    })?;

                let chunk_result = match maybe_chunk {
                    Some(r) => r,
                    None => break,
                };

                let bytes = chunk_result.context("Error reading stream chunk")?;
                let text =
                    std::str::from_utf8(&bytes).context("Stream chunk is not valid UTF-8")?;
                line_buffer.push_str(text);

                while let Some(nl) = line_buffer.find('\n') {
                    let line = line_buffer[..nl].trim().to_string();
                    line_buffer = line_buffer[nl + 1..].to_string();

                    if line.is_empty() {
                        continue; // SSE uses blank lines as event separators
                    }

                    // Strip the SSE "data: " prefix.
                    let json_str = if let Some(rest) = line.strip_prefix("data: ") {
                        if rest == "[DONE]" {
                            break 'stream; // clean end-of-stream marker
                        }
                        rest
                    } else {
                        line.as_str() // some servers omit the prefix
                    };

                    match serde_json::from_str::<StreamChunk>(json_str) {
                        Ok(chunk) => {
                            for choice in &chunk.choices {
                                // Content token
                                if let Some(content) = &choice.delta.content {
                                    if !content.is_empty() {
                                        cb(content);
                                        full_content.push_str(content);
                                    }
                                }

                                // Tool call fragments — accumulate per index
                                if let Some(tc_deltas) = &choice.delta.tool_calls {
                                    for delta in tc_deltas {
                                        // Grow the builders vec to fit this index.
                                        while builders.len() <= delta.index {
                                            builders.push(ToolCallBuilder::default());
                                        }
                                        let b = &mut builders[delta.index];

                                        if let Some(id) = &delta.id {
                                            b.id = Some(id.clone());
                                        }
                                        if let Some(func) = &delta.function {
                                            if let Some(name) = &func.name {
                                                b.name.push_str(name);
                                            }
                                            if let Some(args) = &func.arguments {
                                                b.arguments.push_str(args);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            debug!(line = %json_str, error = %e, "Failed to parse SSE chunk — skipping");
                        }
                    }
                }
            }

            let tool_calls = if builders.is_empty() {
                None
            } else {
                Some(builders.into_iter().map(|b| b.finish()).collect())
            };

            Ok(Message {
                role: Role::Assistant,
                content: full_content,
                tool_calls,
                tool_call_id: None,
            })
        } else {
            // --- Non-streaming path ---
            let body_text = response
                .text()
                .await
                .context("Failed to read OpenAI backend response body")?;

            debug!(response = %body_text, "Received response from OpenAI backend");

            let chat_response: ChatResponse =
                serde_json::from_str(&body_text).context("Failed to parse OpenAI chat response")?;

            let choice = chat_response
                .choices
                .into_iter()
                .next()
                .context("OpenAI response contained no choices")?;

            let tool_calls = choice.message.tool_calls.map(|calls| {
                calls
                    .into_iter()
                    .map(|c| {
                        // OpenAI tool call arguments are a JSON *string*.
                        // Parse them into a Value so the agent loop can use them.
                        let arguments =
                            serde_json::from_str::<serde_json::Value>(&c.function.arguments)
                                .unwrap_or_else(
                                    |_| serde_json::json!({ "raw": c.function.arguments }),
                                );
                        ToolCall {
                            id: c.id,
                            function: FunctionCall {
                                name: c.function.name,
                                arguments,
                            },
                        }
                    })
                    .collect()
            });

            Ok(Message {
                role: Role::Assistant,
                content: choice.message.content.unwrap_or_default(),
                tool_calls,
                tool_call_id: None,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_builder_valid_json() {
        let mut b = ToolCallBuilder::default();
        b.id = Some("call_123".to_string());
        b.name = "file_read".to_string();
        b.arguments = r#"{"path":"src/main.rs"}"#.to_string();

        let tc = b.finish();
        assert_eq!(tc.function.name, "file_read");
        assert_eq!(tc.function.arguments["path"], "src/main.rs");
        assert_eq!(tc.id.unwrap(), "call_123");
    }

    #[test]
    fn test_tool_call_builder_invalid_json_falls_back() {
        let mut b = ToolCallBuilder::default();
        b.name = "shell_exec".to_string();
        b.arguments = "not valid json{{".to_string();

        let tc = b.finish();
        // Falls back to wrapping in a `raw` key rather than panicking.
        assert_eq!(tc.function.arguments["raw"], "not valid json{{");
    }

    #[test]
    fn test_stream_chunk_content_deserialization() {
        let json = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":" Hello"},"finish_reason":null}]}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some(" Hello"));
    }

    #[test]
    fn test_stream_chunk_tool_call_deserialization() {
        let json = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"file_read","arguments":""}}]},"finish_reason":null}]}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();
        let tc = &chunk.choices[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.index, 0);
        assert_eq!(tc.id.as_deref(), Some("call_abc"));
        assert_eq!(
            tc.function.as_ref().unwrap().name.as_deref(),
            Some("file_read")
        );
    }

    #[test]
    fn test_non_streaming_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here are the files.",
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }]
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("Here are the files.")
        );
    }
}
