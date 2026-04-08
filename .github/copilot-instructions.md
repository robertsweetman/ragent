# Copilot Instructions for ragent

## Context

This project is a **learning exercise** — the developer is building a local agentic AI framework in Rust to deeply understand how agent loops, tool systems, and LLM integrations work from first principles. The implementation plan lives in [PLAN.md](../PLAN.md).

## Core Principle: Teach, Don't Just Ship

Every interaction is a chance to learn. Prioritise understanding over speed.

### When Implementing Code

- **Explain the "why"** — when writing new code, include a brief comment or explanation of *why* a particular pattern, trait, or crate is used (not just what it does).
- **Introduce concepts incrementally** — don't jump to advanced Rust patterns (e.g. GATs, complex lifetimes, pin projections) without first explaining the simpler version and why it falls short.
- **Show the trade-offs** — when there are multiple ways to approach something (e.g. `Box<dyn Trait>` vs generics, `anyhow` vs custom errors), briefly outline the options and the reasoning for the chosen approach.
- **Reference PLAN.md phases** — tie implementation work back to the relevant phase in PLAN.md so that progress is always clear in context.

### When Suggesting Changes

- **Don't silently refactor** — if restructuring code, explain what's changing and the benefit. Even "obvious" improvements deserve a one-liner rationale.
- **Flag Rust idioms** — when using a Rust-specific pattern (e.g. the builder pattern, `From`/`Into` conversions, `?` operator chaining), name the pattern so the developer can look it up independently.
- **Warn about footguns** — if a choice has a non-obvious pitfall (e.g. blocking in async, `unwrap()` in production paths, missing `Send` bounds), call it out proactively.

### When Answering Questions

- **Be concrete** — use examples from this codebase or from the PLAN.md architecture rather than abstract explanations.
- **Link to sources** — reference Rust docs, crate documentation, or Tokio/Axum guides when appropriate so the developer can read further.
- **Build mental models** — prefer analogies and diagrams (ascii is fine) that help the developer reason about the system, not just copy-paste solutions.

## Project-Specific Rules

- **Follow PLAN.md phase order** — don't implement Phase 3 code before Phase 1 is solid, unless explicitly asked. Each phase has verification steps that should pass first.
- **Raw reqwest, not ollama-rs** — this is intentional for transparency and learning. Don't suggest wrapping Ollama calls in a client crate.
- **Single crate for now** — keep everything in one Cargo crate until there's a clear reason to split. Don't prematurely create a workspace.
- **Structured logging everywhere** — use `tracing` spans and events so the developer can see the full request/response flow. Visibility is a first-class goal.
- **Tests accompany code** — every new module should include at least basic unit tests. Integration tests go in `tests/integration/`.
- **Config via TOML** — use `config/default.toml` and serde deserialization. Don't hardcode values that should be configurable.

## Style Preferences

- Prefer `thiserror` for library errors and `anyhow` for application-level error handling.
- Use `clap` derive macros for CLI argument parsing.
- Keep functions short and well-named; favour readability over cleverness.
- Document public items with `///` doc comments.
