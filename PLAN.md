# Plan: ragent — Local Rust Agent Framework

## TL;DR

Build a local-only agentic framework in Rust ("ragent") that runs your own LLM (via Ollama initially), executes tools autonomously, and integrates with ZED/VSCode editors. The framework prioritises full debuggability, swappable LLM backends, and controlled web access — no data leaves your network unless you explicitly allow it.

## Architecture Overview

```text
  ZED / VSCode / CLI / Web UI
         │
         ▼
  ┌──────────────────────┐
  │   ragent API Server  │  (axum, OpenAI-compatible /v1/chat/completions)
  │   + CLI REPL         │
  └────────┬─────────────┘
           │
  ┌────────▼──────────────┐
  │    Agent Loop         │  (message → LLM → parse tool calls → execute → repeat)
  │  ┌─────────────────┐  │
  │  │ LLM Abstraction │  │  (trait: OllamaBackend, LlamaCppBackend, etc.)
  │  └─────────────────┘  │
  │  ┌─────────────────┐  │
  │  │  Tool Registry  │  │  (trait: Tool, per-project tool sets)
  │  └─────────────────┘  │
  │  ┌─────────────────┐  │
  │  │  Skill/Profile  │  │  (system prompt + tool set + config per role)
  │  └─────────────────┘  │
  └───────────────────────┘
           │
  ┌────────▼──────────────┐
  │   Ollama (local)      │  (localhost:11434, model inference)
  └───────────────────────┘
```

## Project Structure (single crate, grows into workspace later)

```text
ragent/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI entry point (clap + REPL)
│   ├── lib.rs               # Library root, re-exports
│   ├── agent/
│   │   ├── mod.rs
│   │   ├── loop.rs          # Core agent loop
│   │   ├── message.rs       # Message/conversation types
│   │   └── skill.rs         # Skill definitions
│   ├── llm/
│   │   ├── mod.rs
│   │   ├── traits.rs        # LlmBackend trait
│   │   └── ollama.rs        # Ollama implementation
│   ├── tools/
│   │   ├── mod.rs
│   │   ├── traits.rs        # Tool trait + ToolResult
│   │   ├── registry.rs      # Tool registry
│   │   ├── file_ops.rs      # File read/write/search
│   │   ├── shell.rs         # Shell command execution
│   │   └── web.rs           # Web fetch with domain allowlist
│   ├── server/
│   │   ├── mod.rs
│   │   └── openai_compat.rs # OpenAI-compatible API endpoints
│   └── config.rs            # Configuration (TOML)
├── config/
│   └── default.toml         # Default config
└── tests/
    └── integration/         # Integration tests
```

## Phases

### Phase 1: Hello Agent (Foundation)

**Goal**: Chat with a local LLM via CLI, see all traffic.

1. Initialise Rust project with Cargo workspace-ready structure
2. Add dependencies: `tokio`, `serde`/`serde_json`, `reqwest`, `tracing`/`tracing-subscriber`, `clap`
3. Define core message types in `agent/message.rs`:
   - `Message { role: Role, content: String, tool_calls: Option<Vec<ToolCall>>, tool_call_id: Option<String> }`
   - `Role { System, User, Assistant, Tool }`
   - `Conversation` (ordered `Vec<Message>` with helper methods)
4. Define `LlmBackend` trait in `llm/traits.rs`:
   - `async fn chat(&self, messages: &[Message], tools: &[ToolDef]) -> Result<Message>`
   - `async fn chat_stream(&self, messages: &[Message], tools: &[ToolDef]) -> Result<impl Stream<Item = ...>>`
5. Implement `OllamaBackend` in `llm/ollama.rs`:
   - Uses `reqwest` directly (not ollama-rs) for full transparency and learning
   - POST to `/api/chat` with tool definitions
   - Full request/response logging via `tracing`
   - Configurable model, temperature, context window
6. Build CLI REPL in `main.rs`:
   - `clap` for args (model name, Ollama URL, log level)
   - Simple stdin loop: read input → send to LLM → print response
   - Coloured output distinguishing user/assistant/system
7. **Verification**: Run `cargo build`, start Ollama with a model (e.g. `llama3.2` but use `DeepSeek` first), chat via CLI, inspect logs showing full HTTP payloads

### Phase 2: Tool System (Core Agentic Loop)

**Goal**: Agent autonomously uses tools to accomplish multi-step tasks.
*Depends on Phase 1.*

1. Define `Tool` trait in `tools/traits.rs`:
   - `fn name(&self) -> &str`
   - `fn description(&self) -> &str`
   - `fn parameters(&self) -> serde_json::Value` (JSON Schema)
   - `async fn execute(&self, args: serde_json::Value) -> Result<ToolResult>`
   - `ToolResult { content: String, is_error: bool }`
2. Build `ToolRegistry` in `tools/registry.rs`:
   - Register/lookup tools by name
   - Generate OpenAI-format tool definitions for LLM
3. Implement the agent loop in `agent/loop.rs`:
   - Send messages + tool defs to LLM
   - If response contains `tool_calls` → execute each tool → append tool results → re-send
   - Loop until LLM responds with plain text (no tool calls)
   - Configurable max iterations (safety limit)
   - Full logging of each loop iteration
4. Implement `FileReadTool`, `FileWriteTool`, `FileSearchTool` (grep) in `tools/file_ops.rs`
5. Implement `ShellExecTool` in `tools/shell.rs`:
   - Runs commands via `tokio::process::Command`
   - Timeout + output size limits
   - Configurable working directory
6. Wire tools into CLI: agent loop replaces simple chat
7. **Verification**: Ask agent "create a file called hello.rs with a hello world program, compile it, and run it" — watch it use file_write → shell_exec → observe the full tool call chain in logs

### Phase 3: Code Agent Skill (Primary Use Case)

**Goal**: Agent writes, tests, and iterates on code effectively.
*Depends on Phase 2.*

1. Define `Skill` struct in `agent/skill.rs`:
   - System prompt template
   - Allowed tool set
   - Project context (files to auto-include)
   - Config overrides (model, temperature)
2. Build a "code-writer" skill:
   - System prompt tuned for code generation
   - All file + shell tools enabled
   - Project-aware context loading (reads key files)
3. Add `ProjectContext` loader:
   - Reads project structure (tree)
   - Loads key files (Cargo.toml, README, etc.)
   - Configurable via per-project TOML
4. Add test execution tool:
   - Runs `cargo test` / `npm test` / etc. based on project type
   - Parses output for pass/fail
5. **Verification**: Point ragent at a Rust project, ask it to add a function with tests — verify it reads existing code, writes new code, runs tests, fixes failures

### Phase 4: API Server + Editor Integration

**Goal**: Use ragent from ZED (and VSCode).
*Depends on Phase 2; can run in parallel with Phase 3.*

1. Build axum HTTP server in `server/`:
   - OpenAI-compatible `POST /v1/chat/completions` endpoint
   - Accepts standard OpenAI chat format
   - Internally runs agent loop (tool use is invisible to the editor)
   - SSE streaming for token-by-token output
   - `GET /v1/models` endpoint listing available skills/models
2. Add server mode to CLI: `ragent serve --port 8080`
3. Configure ZED to use ragent:
   - Settings: `"language_models": { "openai_compatible": { "ragent": { "api_url": "http://localhost:8080/v1" } } }`
   - ragent appears as an LLM provider in ZED's Agent Panel
4. Configure VSCode (if desired):
   - Same OpenAI-compatible endpoint works with extensions like "Continue"
5. **Verification**: Start ragent server, configure ZED, open a project, use ZED's Agent Panel to ask ragent to write code — see it work end-to-end

### Phase 5: Controlled Web Research

**Goal**: Agent researches online within project-defined boundaries.
*Depends on Phase 2.*

1. Implement `WebFetchTool` in `tools/web.rs`:
   - HTTP GET with configurable allowed domains per project
   - HTML → text extraction (using `scraper` crate)
   - robots.txt respect
   - Rate limiting
2. Add per-project source configuration:
   - `ragent.toml` per project: `allowed_domains = ["docs.rs", "doc.rust-lang.org", ...]`
   - Default deny — only whitelisted domains
3. Add search tool (optional):
   - Integration with SearXNG (self-hosted search) or DuckDuckGo HTML
   - Results filtered to allowed domains
4. **Verification**: Ask agent to look up the docs for a specific Rust crate — verify it only accesses allowed domains, extracts relevant content

### Phase 6: Multi-Agent Swarm (Future)

**Goal**: Multiple specialised agents collaborate.
*Depends on Phases 3 & 4.*

1. Agent-as-tool pattern: wrap an Agent instance as a Tool
2. Orchestrator agent that delegates to specialist agents based on task analysis
3. Shared context/memory between agents in a session
4. Skill library: code-writer, code-reviewer, researcher, planner
5. **Verification**: Give orchestrator a complex task (e.g. "research best practices for X, then implement it") — watch it delegate between researcher and code-writer agents

## Technology Stack

| Purpose | Crate | Why |
| --------- | ------- | ----- |
| Async runtime | `tokio` | Only real choice, battle-tested |
| HTTP client | `reqwest` | Standard, full-featured |
| HTTP server | `axum` | Ergonomic, tower middleware, fits tokio |
| Serialisation | `serde` + `serde_json` | Universal |
| CLI | `clap` (derive) | De facto standard |
| Logging | `tracing` + `tracing-subscriber` | Structured, async-aware |
| Config | `toml` + `serde` | Simple, human-readable |
| HTML parsing | `scraper` | CSS selectors, lightweight |
| Terminal colours | `colored` or `crossterm` | REPL output |

## Key Design Decisions

1. **Build from scratch, not using Rig**: We want full control + deep understanding. Rig (Rust agent framework) is good reference material but too opaque for our goals. We build the agent loop ourselves.
2. **Raw reqwest over ollama-rs**: Directly call Ollama REST API with reqwest for full transparency. Every HTTP request/response is logged and inspectable.
3. **OpenAI-compatible API surface**: ragent server speaks OpenAI format so ZED, VSCode, and any OpenAI-compatible client can connect without custom extensions.
4. **Single crate initially**: Easier to learn and navigate. Can split into workspace crates later when boundaries are clear.
5. **Ollama first, llama.cpp later**: Ollama provides a clean REST API and is trivial to set up. Direct llama.cpp integration (via `llama-cpp-2` or `candle`) can be added as another `LlmBackend` implementation later.
6. **GPL v3 licence**: Already set in repo.

## Deployment Target

- Primary: Linux host (or WSL) on local network
- `ragent serve --bind 0.0.0.0:8080` to make accessible from other machines on the same network
- Firewall: block all outbound except Ollama port + whitelisted domains
- No cloud dependencies, no telemetry, no phone-home

## Scope Boundaries

**In scope**: Core agent loop, tool system, Ollama backend, CLI, API server, ZED integration, controlled web access, code-writing skill, multi-agent orchestration

**Out of scope (for now)**: GUI/Web UI, TUI (ratatui), MCP server implementation, voice interface, persistent memory/RAG (vector DB), image/multimodal support

Persistent memory example can be found here [mempalace](https://github.com/milla-jovovich/mempalace) pretty useful although implemented in Python
