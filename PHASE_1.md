A focused reading list mapped to each Phase 1 task:

---

### 1. Local LLM Setup — Model Choice

Your plan mentions DeepSeek first. For a coding-focused local model in 2026, the top picks are:

- **Qwen3-Coder** (30B if your VRAM allows, 7B for lighter rigs) — currently one of the strongest open coding models, excellent at tool calling
- **DeepSeek-Coder-V2** / **DeepSeek-V3** — very strong at code, but the larger variants are heavy
- **Codestral** (Mistral's coding model) — good alternative

**Where to read:**
- Ollama model library: **https://ollama.com/library** — browse available models, check VRAM requirements, and pull commands
- Ollama docs for install + quickstart: **https://ollama.com** (one binary, `ollama pull qwen2.5-coder:32b`, done)
- Each model's card on Ollama shows parameter counts, quantisation options, and supported features (critically: **tool calling support** — not all models handle it well)

**Key thing to verify before committing to a model**: does it support Ollama's tool/function calling format? Qwen2.5-Coder does. This matters for Phase 2 but you'll want to pick a model now that grows with you.

---

### 2. Message Types (`agent/message.rs`)

Your plan defines `Message`, `Role`, `Conversation`. These mirror the **OpenAI Chat Completions API format**, which Ollama also adopts.

**Where to read:**
- **Ollama REST API docs**: https://github.com/ollama/ollama/blob/main/docs/api.md — specifically the `/api/chat` endpoint. This is your ground truth for what JSON you'll send and receive
- **OpenAI Chat Completions reference** (for the message shape your types should mirror): https://platform.openai.com/docs/api-reference/chat — look at the `messages` array structure, `role` enum, `tool_calls` field, and `tool_call_id`. Your Rust types should serialize/deserialize to match this shape
- For the Rust side: **serde docs** https://serde.rs — specifically `#[serde(rename_all = "lowercase")]` for the `Role` enum, `#[serde(skip_serializing_if = "Option::is_none")]` for optional fields like `tool_calls`

---

### 3. LLM Backend Trait + Ollama Implementation (`llm/traits.rs`, `llm/ollama.rs`)

**Where to read:**
- **Ollama `/api/chat` endpoint** (same link above) — study the request body shape (model, messages, stream, tools) and the response body. Pay attention to the streaming vs non-streaming variants
- **reqwest docs**: https://docs.rs/reqwest — focus on the async client, `.post()`, `.json()`, `.send()`, and `.json::<T>()` response deserialization. For streaming later: `.bytes_stream()`
- **Rust async traits**: as of Rust 2024 edition, `async fn` in traits is stable. Read https://blog.rust-lang.org/2023/12/21/async-fn-rpit-in-traits.html — this directly impacts how you define `LlmBackend`
- **tracing crate**: https://docs.rs/tracing — look at `#[instrument]`, `tracing::info!`, `tracing::debug!` for logging all HTTP payloads. Also https://docs.rs/tracing-subscriber for setting up the subscriber in `main.rs`

---

### 4. CLI REPL (`main.rs`)

**Where to read:**
- **clap derive docs**: https://docs.rs/clap — the derive tutorial is excellent. You need `#[derive(Parser)]` for args like `--model`, `--ollama-url`, `--log-level`
- **tokio docs**: https://docs.rs/tokio — mainly `#[tokio::main]` and `tokio::io::BufReader` + `tokio::io::AsyncBufReadExt` for async stdin reading
- **colored crate**: https://docs.rs/colored — for distinguishing user/assistant/system output

---

### Suggested Reading Order

1. **Ollama install + pull a model** (get it running, poke `/api/chat` with `curl` manually)
2. **Ollama `/api/chat` API docs** (understand the JSON shapes — this dictates your Rust types)
3. **serde docs** (so you know how to map those JSON shapes to Rust structs)
4. **reqwest docs** (so you know how to make the HTTP calls)
5. **tracing docs** (so you can see everything flowing)
6. **clap derive docs** (wire it all into a CLI)

Start with step 1 — get Ollama running and manually `curl` a chat request. Seeing the raw JSON before writing any Rust types will make the `message.rs` design click immediately.