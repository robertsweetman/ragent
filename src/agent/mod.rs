//! Agent module — core agent types and conversation management.
//!
//! This module contains the message types that mirror the Ollama/OpenAI chat API
//! format, plus a `Conversation` helper for managing ordered message history.

pub mod message;

// Note: loop.rs exists but is not wired up until Phase 2 (tool system).
// Rust won't let us write `mod loop;` because `loop` is a keyword.
// We'll use `mod agent_loop;` and rename the file in Phase 2.
