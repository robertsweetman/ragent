//! Agent module — core agent types, conversation management, and the agent loop.
//!
//! This module contains:
//! - `message` — Message types that mirror the Ollama/OpenAI chat API format,
//!   plus a `Conversation` helper for managing ordered message history.
//! - `agent_loop` — The core agentic loop that cycles between LLM calls and
//!   tool execution until the LLM produces a final text response.

pub mod agent_loop;
pub mod message;

