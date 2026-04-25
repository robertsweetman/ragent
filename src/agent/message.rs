//! Core message types for LLM conversations.
//!
//! These types serialize/deserialize to match the Ollama `/api/chat` JSON format.
//! See: https://github.com/ollama/ollama/blob/main/docs/api.md#chat

use serde::{Deserialize, Serialize};

/// The role of a message participant in a conversation.
///
/// Maps to the `role` field in chat API messages.
/// Uses `#[serde(rename_all = "lowercase")]` so `Role::System` serializes as `"system"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instructions that guide the LLM's behaviour.
    System,
    /// Messages from the human user.
    User,
    /// Messages from the LLM.
    Assistant,
    /// Results returned from tool execution (Phase 2+).
    Tool,
}

/// A single message in a conversation.
///
/// This struct maps directly to the message objects in Ollama's chat API.
/// The `content` field uses `String` (not `Option<String>`) because Ollama
/// always includes content (empty string `""` when the assistant makes tool calls).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Who sent this message.
    pub role: Role,

    /// The text content of the message. Empty string for tool-call-only responses.
    #[serde(default)]
    pub content: String,

    /// Tool calls requested by the assistant (Phase 2+).
    /// Present only when the LLM wants to invoke tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// The ID of the tool call this message is responding to (Phase 2+).
    /// Present only on messages with `role: Tool`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool invocation requested by the LLM.
///
/// Ollama's format doesn't include an `id` field (unlike OpenAI), so we make it optional
/// for forward compatibility with OpenAI-format backends in Phase 4.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Optional unique identifier for this tool call (used by OpenAI format, not Ollama).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// The function to call and its arguments.
    pub function: FunctionCall,
}

/// The function name and arguments within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function/tool to execute.
    pub name: String,

    /// Arguments as a JSON value. Ollama sends this as an object `{...}`,
    /// while OpenAI sends it as a JSON string. Using `Value` handles both.
    pub arguments: serde_json::Value,
}

/// Definition of a tool that the LLM can call.
///
/// Sent to the LLM alongside messages so it knows what tools are available.
/// Not used in Phase 1 (simple chat), but defined now so the `LlmBackend` trait
/// signature is stable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// Always `"function"` for now.
    #[serde(rename = "type")]
    pub tool_type: String,

    /// The function definition (name, description, parameters schema).
    pub function: FunctionDef,
}

/// A function definition within a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    /// The name the LLM uses to invoke this tool.
    pub name: String,

    /// Human-readable description of what this tool does.
    pub description: String,

    /// JSON Schema describing the function's parameters.
    pub parameters: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Conversation helper
// ---------------------------------------------------------------------------

/// An ordered sequence of messages forming a conversation.
///
/// Wraps a `Vec<Message>` with convenience methods for the common patterns
/// in an agent loop (adding messages, accessing history, clearing).
#[derive(Debug, Clone, Default)]
pub struct Conversation {
    messages: Vec<Message>,
}

impl Conversation {
    /// Create an empty conversation.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Create a conversation that starts with a system prompt.
    ///
    /// The system prompt tells the LLM how to behave (e.g. "You are a helpful assistant").
    pub fn with_system_prompt(prompt: &str) -> Self {
        let mut conv = Self::new();
        conv.messages.push(Message {
            role: Role::System,
            content: prompt.to_string(),
            tool_calls: None,
            tool_call_id: None,
        });
        conv
    }

    /// Append a message to the conversation.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get all messages in order.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Add a user message (convenience method).
    pub fn add_user_message(&mut self, content: &str) {
        self.push(Message {
            role: Role::User,
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Add an assistant message (convenience method).
    pub fn add_assistant_message(&mut self, content: &str) {
        self.push(Message {
            role: Role::Assistant,
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    /// Clear all messages (reset the conversation).
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Number of messages in the conversation.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Display implementations for nice REPL output
// ---------------------------------------------------------------------------

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.role, self.content)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serialization() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), r#""system""#);
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), r#""user""#);
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            r#""assistant""#
        );
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), r#""tool""#);
    }

    #[test]
    fn test_role_deserialization() {
        let role: Role = serde_json::from_str(r#""assistant""#).unwrap();
        assert_eq!(role, Role::Assistant);
    }

    #[test]
    fn test_message_serialization_simple() {
        let msg = Message {
            role: Role::User,
            content: "Hello!".to_string(),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "Hello!");
        assert!(json.get("tool_calls").is_none());
        assert!(json.get("tool_call_id").is_none());
    }

    #[test]
    fn test_message_deserialization_from_ollama_response() {
        let json = r#"{
            "role": "assistant",
            "content": "Hello! How can I help you today?"
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "Hello! How can I help you today?");
        assert!(msg.tool_calls.is_none());
    }

    #[test]
    fn test_message_deserialization_with_tool_calls() {
        let json = r#"{
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "London"}
                    }
                }
            ]
        }"#;
        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content, "");
        let tool_calls = msg.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert!(tool_calls[0].id.is_none());
    }

    #[test]
    fn test_conversation_with_system_prompt() {
        let conv = Conversation::with_system_prompt("You are helpful.");
        assert_eq!(conv.len(), 1);
        assert_eq!(conv.messages()[0].role, Role::System);
        assert_eq!(conv.messages()[0].content, "You are helpful.");
    }

    #[test]
    fn test_conversation_add_messages() {
        let mut conv = Conversation::new();
        assert!(conv.is_empty());
        conv.add_user_message("Hi");
        conv.add_assistant_message("Hello!");
        assert_eq!(conv.len(), 2);
        assert_eq!(conv.messages()[0].role, Role::User);
        assert_eq!(conv.messages()[1].role, Role::Assistant);
    }

    #[test]
    fn test_conversation_clear() {
        let mut conv = Conversation::with_system_prompt("test");
        conv.add_user_message("hello");
        assert_eq!(conv.len(), 2);
        conv.clear();
        assert!(conv.is_empty());
    }

    #[test]
    fn test_tool_def_serialization() {
        let tool = ToolDef {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "read_file".to_string(),
                description: "Read a file from disk".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file path to read"
                        }
                    },
                    "required": ["path"]
                }),
            },
        };
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "read_file");
    }
}
