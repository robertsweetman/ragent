//! The `LlmBackend` trait — the abstraction that makes LLM providers swappable.
//!
//! # Design Decision: Why a trait?
//!
//! We want to swap between Ollama, llama.cpp, or other backends without changing
//! the agent loop. A trait gives us that polymorphism. In Phase 1 we only have
//! `OllamaBackend`, but the trait is here from the start so the interface is stable.
//!
//! # Object Safety Note
//!
//! `async fn` in traits (stable since Rust 1.75 / edition 2024) works great with
//! generics but is NOT object-safe (you can't write `Box<dyn LlmBackend>`).
//! For Phase 1 this is fine — we use the concrete type directly. If we need dynamic
//! dispatch later, we can use the `trait_variant` crate or manual boxing.

use crate::agent::message::{Message, ToolDef};
use anyhow::Result;

/// Trait that all LLM backend implementations must satisfy.
///
/// The `Send + Sync` bounds are required because the backend will be used
/// across async task boundaries in tokio.
///
/// # Why `#[allow(async_fn_in_trait)]`?
///
/// The `async_fn_in_trait` lint warns that the `Future` returned by `async fn`
/// in a public trait has no guaranteed `Send` bound, which matters if callers
/// use `tokio::spawn` (requires `Send`). We suppress it because:
///
/// 1. This trait is only used within our own crate (not a public API for others).
/// 2. We call it via generics (`impl LlmBackend`), not `dyn LlmBackend`, so the
///    compiler can see the concrete type and infer `Send` from the implementation.
/// 3. Our sole implementation (`OllamaBackend`) uses only `Send`-safe types.
///
/// If we ever need `dyn LlmBackend` (dynamic dispatch), we'd switch to the
/// `trait_variant` crate or manually return `Pin<Box<dyn Future + Send>>`.
#[allow(async_fn_in_trait)]
pub trait LlmBackend: Send + Sync {
    /// Send a conversation to the LLM and get a response.
    ///
    /// # Arguments
    /// * `messages` — The conversation history (system + user + assistant messages).
    /// * `tools` — Tool definitions the LLM can choose to call. Pass `&[]` for no tools.
    /// * `on_token` — Optional callback invoked with each text token as it streams in.
    ///   When `Some`, the backend switches to streaming mode (NDJSON). When `None`,
    ///   the backend uses a single blocking response (keeps existing test behaviour).
    ///
    /// # Returns
    /// The assistant's full response as a `Message` after all tokens/tool-calls arrive.
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        on_token: Option<&dyn Fn(&str)>,
    ) -> Result<Message>;

    // Streaming is handled via the `on_token` callback on `chat` rather than
    // a separate method, keeping the trait surface small and the call sites simple.
}
