//! LLM backend module — abstractions and implementations for language model inference.
//!
//! This module defines the `LlmBackend` trait that all LLM providers implement,
//! plus the Ollama implementation that talks to a local Ollama server via REST API.

pub mod ollama;
pub mod traits;

pub use ollama::OllamaBackend;
pub use traits::LlmBackend;
