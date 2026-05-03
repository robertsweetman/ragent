//! Tool system — traits, registry, and built-in tool implementations.
//!
//! # Phase 2 (PLAN.md)
//! Goal: "Agent autonomously uses tools to accomplish multi-step tasks."
//!
//! This module provides:
//! - `traits` — The `Tool` trait and `ToolResult` type
//! - `registry` — `ToolRegistry` for collecting and looking up tools by name
//! - `file_ops` — File read, write, and search (grep) tools
//! - `shell` — Shell command execution tool
//!
//! # Architecture
//!
//! Tools are the bridge between the LLM's intent and real-world actions.
//! The LLM outputs a tool call (name + JSON arguments), the agent loop
//! looks up the tool in the registry, executes it, and feeds the result
//! back to the LLM as a message.
//!
//! ```text
//!   LLM response          ToolRegistry            Tool impl
//!   ─────────────         ────────────            ─────────
//!   tool_calls: [     →   registry.get("name")  → tool.execute(args)
//!     { name, args }      returns &dyn Tool        returns ToolResult
//!   ]                                              ↓
//!                                                fed back to LLM
//! ```

pub mod file_ops;
pub mod file_patch;
pub mod registry;
pub mod shell;
pub mod traits;

// Re-export the most commonly used types so callers can write
// `use ragent::tools::{Tool, ToolResult, ToolRegistry}` without
// reaching into submodules.
pub use file_patch::FilePatchTool;
pub use registry::ToolRegistry;
pub use traits::{Tool, ToolResult};
