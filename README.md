# ragent

A local-only agentic AI framework written in Rust.

Run your own LLM (via [Ollama](https://ollama.com)), execute tools autonomously, and integrate with code editors like [ZED](https://zed.dev) and VS Code — all without sending data to the cloud.

## Goals

- **Full local control**: Swap the underlying LLM, inspect every request/response, keep all data on your network
- **Agentic tool use**: Autonomous multi-step tool execution (file ops, shell, web research) driven by a local model
- **Editor integration**: OpenAI-compatible API server so ZED, VS Code, or any compatible client can use ragent as an LLM provider
- **Debuggability**: Structured logging of the entire agent loop — see exactly what the model asked for and what tools returned
- **Extensible skills**: Define specialised agent profiles (system prompt + tool set) per task type

## Status

🚧 **Early development** — see [PLAN.md](PLAN.md) for the full roadmap.

## Prerequisites

- [Rust](https://rustup.rs/) (stable toolchain)
- [Ollama](https://ollama.com) running locally with at least one model pulled (e.g. `ollama pull llama3.2`)

## Quick Start

```bash
# Clone
git clone https://github.com/<you>/ragent.git && cd ragent

# Build
cargo build

# Run (once Phase 1 is complete)
cargo run -- --model llama3.2
```

## Licence

GPL-3.0 — see [LICENSE.md](LICENSE.md).
