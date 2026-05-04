//! LLM backend module — abstractions and implementations for language model inference.
//!
//! This module defines the `LlmBackend` trait that all LLM providers implement,
//! plus the Ollama implementation that talks to a local Ollama server via REST API,
//! and an OpenAI-compatible backend for LM Studio, Jan, or remote OpenAI/Anthropic.

pub mod ollama;
pub mod openai;
pub mod traits;

pub use ollama::OllamaBackend;
pub use openai::OpenAiBackend;
pub use traits::LlmBackend;
