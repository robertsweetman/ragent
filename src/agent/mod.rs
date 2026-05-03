//! Agent module — core agent types, conversation management, and the agent loop.
//!
//! This module contains:
//! - `message` — Message types that mirror the Ollama/OpenAI chat API format,
//!   plus a `Conversation` helper for managing ordered message history.
//! - `agent_loop` — The core agentic loop that cycles between LLM calls and
//!   tool execution until the LLM produces a final text response.
//! - `skill` — Named bundles of (system prompt + tool allowlist) that configure
//!   the agent's behaviour for a specific task domain (chat, code, etc.).
//! - `project_context` — Workspace introspection: directory tree + key files,
//!   formatted as Markdown and appended to the skill's system prompt so the
//!   LLM starts every session with a mental model of the project.

pub mod agent_loop;
pub mod message;
pub mod project_context;
pub mod skill;

// Re-export the most commonly used types so callers can write
// `use ragent::agent::{Skill, ProjectContext}` without reaching into submodules.
pub use project_context::ProjectContext;
pub use skill::Skill;
