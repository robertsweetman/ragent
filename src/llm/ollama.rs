//! Ollama backend — talks to a local Ollama server via its REST API.
//!
//! Uses raw `reqwest` (not ollama-rs) for full transparency. Every HTTP
//! request and response is logged via `tracing` so you can see exactly
//! what's flowing between ragent and the LLM.
//!
//! Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md

use crate::agent::message::{Message, ToolDef};
use crate::config::OllamaConfig;
use crate::llm::traits::LlmBackend;
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

/// Ollama-specific errors with actionable messages.
///
/// We use `thiserror` here (rather than `anyhow`) because these errors
/// carry structured information that the caller (main.rs) uses to show
/// helpful messages to the user (e.g. "run `ollama pull <model>`").
#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error(
        "Cannot connect to Ollama at {url}. Is Ollama running?\n  -> Start it with: ollama serve"
    )]
    ConnectionFailed { url: String },

    #[error("Ollama is not installed or not in PATH.\n  -> Install from: https://ollama.com")]
    NotInstalled,

    #[error("Model '{model}' is not available locally.\n  -> Pull it with: ollama pull {model}")]
    ModelNotFound { model: String },

    #[error(
        "Model '{model}' does not support tool/function calling.\n  \
         Tool calling is required for the agent loop (Phase 2+).\n  \
         Switch to a model that supports tools, for example:\n    \
         ollama pull qwen2.5-coder:32b\n    \
         ollama pull qwen3-coder:30b\n    \
         ollama pull mistral:7b\n  \
         Then run ragent with: ragent --model <model_name>"
    )]
    ToolsNotSupported { model: String },

    #[error("Ollama returned an error: {message}")]
    ApiError { message: String },
}

// ---------------------------------------------------------------------------
// Ollama API request/response types (private — not part of our public API)
// ---------------------------------------------------------------------------

/// Request body for POST /api/chat
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatOptions>,
}

/// Optional model parameters sent to Ollama.
#[derive(Debug, Serialize)]
struct ChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<u32>,
}

/// Response body from POST /api/chat (non-streaming).
#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: Message,
    #[allow(dead_code)]
    done: bool,
    #[allow(dead_code)]
    #[serde(default)]
    total_duration: Option<u64>,
    #[allow(dead_code)]
    #[serde(default)]
    eval_count: Option<u64>,
    #[allow(dead_code)]
    #[serde(default)]
    eval_duration: Option<u64>,
}

/// Response from GET /api/tags (list local models).
#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

/// A single model entry from the tags endpoint.
#[derive(Debug, Deserialize)]
struct ModelInfo {
    name: String,
    #[allow(dead_code)]
    #[serde(default)]
    size: Option<u64>,
}

// ---------------------------------------------------------------------------
// OllamaBackend implementation
// ---------------------------------------------------------------------------

/// The Ollama LLM backend.
///
/// Holds a `reqwest::Client` (which manages a connection pool internally)
/// and the Ollama configuration (URL, model, temperature, etc.).
pub struct OllamaBackend {
    client: Client,
    config: OllamaConfig,
}

impl OllamaBackend {
    /// Create a new Ollama backend from configuration.
    ///
    /// This does NOT check if Ollama is running — call `check_connection()` for that.
    /// The separation lets the caller decide when/how to handle connectivity issues.
    pub fn new(config: OllamaConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 min timeout for slow models
            .build()
            .expect("Failed to create HTTP client");

        info!(
            url = %config.url,
            model = %config.model,
            "Created Ollama backend"
        );

        Self { client, config }
    }

    /// Check if Ollama is reachable at the configured URL.
    ///
    /// Ollama responds with `"Ollama is running"` on `GET /`.
    pub async fn check_connection(&self) -> Result<(), OllamaError> {
        let url = &self.config.url;
        debug!(url = %url, "Checking Ollama connection");

        match self.client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!(url = %url, "Ollama is reachable");
                Ok(())
            }
            Ok(resp) => Err(OllamaError::ApiError {
                message: format!("Unexpected status {} from {}", resp.status(), url),
            }),
            Err(_) => Err(OllamaError::ConnectionFailed { url: url.clone() }),
        }
    }

    /// List models available in the local Ollama instance.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.config.url);
        debug!(url = %url, "Listing Ollama models");

        let resp: TagsResponse = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to list Ollama models")?
            .json()
            .await
            .context("Failed to parse Ollama tags response")?;

        let names: Vec<String> = resp.models.into_iter().map(|m| m.name).collect();
        debug!(models = ?names, "Available models");
        Ok(names)
    }

    /// Check if the configured model is available locally.
    pub async fn check_model(&self) -> Result<(), OllamaError> {
        let models = self
            .list_models()
            .await
            .map_err(|_| OllamaError::ConnectionFailed {
                url: self.config.url.clone(),
            })?;

        let target = &self.config.model;
        let found = models.iter().any(|m| {
            m == target
                || m.starts_with(&format!("{}:", target))
                || target.starts_with(&format!("{}:", m.split(':').next().unwrap_or("")))
        });

        if found {
            info!(model = %target, "Model is available");
            Ok(())
        } else {
            warn!(model = %target, available = ?models, "Requested model not found");
            Err(OllamaError::ModelNotFound {
                model: target.clone(),
            })
        }
    }

    /// Get the configured model name.
    pub fn model(&self) -> &str {
        &self.config.model
    }
}

impl LlmBackend for OllamaBackend {
    #[instrument(skip(self, messages, tools), fields(model = %self.config.model, msg_count = messages.len()))]
    async fn chat(&self, messages: &[Message], tools: &[ToolDef]) -> Result<Message> {
        let url = format!("{}/api/chat", self.config.url);

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages: messages.to_vec(),
            stream: false,
            tools: if tools.is_empty() {
                None
            } else {
                Some(tools.to_vec())
            },
            options: Some(ChatOptions {
                temperature: self.config.temperature,
                num_ctx: self.config.context_window,
            }),
        };

        // Log full request at debug level — core goal: full transparency
        debug!(
            request = %serde_json::to_string_pretty(&request).unwrap_or_default(),
            "Sending chat request to Ollama"
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Ollama")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            warn!(status = %status, body = %body, "Ollama returned error");

            // Detect the specific "does not support tools" error so we can
            // give the user an actionable message instead of a raw HTTP dump.
            // Ollama returns this as a 400 when you send tool definitions to
            // a model that wasn't trained for function calling (e.g. deepseek-r1).
            if body.contains("does not support tools") {
                return Err(OllamaError::ToolsNotSupported {
                    model: self.config.model.clone(),
                }
                .into());
            }

            anyhow::bail!("Ollama returned HTTP {}: {}", status, body);
        }

        let body_text = response
            .text()
            .await
            .context("Failed to read Ollama response body")?;

        // Log full response at debug level
        debug!(response = %body_text, "Received response from Ollama");

        let chat_response: ChatResponse =
            serde_json::from_str(&body_text).context("Failed to parse Ollama chat response")?;

        // Log generation speed if timing info is available
        if let Some(eval_count) = chat_response.eval_count {
            if let Some(eval_duration) = chat_response.eval_duration {
                let tokens_per_sec = if eval_duration > 0 {
                    (eval_count as f64) / (eval_duration as f64 / 1_000_000_000.0)
                } else {
                    0.0
                };
                info!(
                    eval_tokens = eval_count,
                    tokens_per_sec = format!("{:.1}", tokens_per_sec),
                    "Generation complete"
                );
            }
        }

        Ok(chat_response.message)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::message::Role;

    #[test]
    fn test_chat_request_serialization() {
        let request = ChatRequest {
            model: "deepseek-r1:8b".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: "Hello".to_string(),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            tools: None,
            options: Some(ChatOptions {
                temperature: Some(0.7),
                num_ctx: None,
            }),
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["model"], "deepseek-r1:8b");
        assert_eq!(json["stream"], false);
        assert!(json.get("tools").is_none()); // skip_serializing_if works
        assert_eq!(json["messages"][0]["role"], "user");
    }

    #[test]
    fn test_chat_response_deserialization() {
        let json = r#"{
            "model": "deepseek-r1:8b",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help?"
            },
            "done": true,
            "total_duration": 1234567890,
            "eval_count": 42,
            "eval_duration": 500000000
        }"#;

        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message.role, Role::Assistant);
        assert_eq!(resp.message.content, "Hello! How can I help?");
        assert!(resp.done);
    }

    #[test]
    fn test_tags_response_deserialization() {
        let json = r#"{
            "models": [
                {"name": "deepseek-r1:8b", "size": 4000000000},
                {"name": "llama3.2:latest", "size": 2000000000}
            ]
        }"#;

        let resp: TagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 2);
        assert_eq!(resp.models[0].name, "deepseek-r1:8b");
    }
}
