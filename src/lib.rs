//! ragent — a local-only agentic AI framework.
//!
//! This is the library root that re-exports all public modules.
//! Keeping a `lib.rs` separate from `main.rs` means the library can
//! be tested independently and potentially used as a dependency later.

pub mod agent;
pub mod config;
pub mod llm;
pub mod tools;
