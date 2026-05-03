//! Skill system — named bundles of system prompt + tool allowlist.
//!
//! A [`Skill`] pairs a base system prompt template with a set of allowed tools.
//! The caller constructs the final prompt via [`Skill::build_system_prompt`] (optionally
//! enriching it with [`ProjectContext`][crate::agent::ProjectContext] output) and then
//! hands the resulting string to [`AgentLoop::new`][crate::agent::agent_loop::AgentLoop].
//!
//! # Design note — Why a "Skill" abstraction?
//!
//! Different tasks need different instructions. A chat assistant doesn't need
//! a paragraph about `cargo --target-dir`; a code-writer does. Bundling the
//! prompt *and* the tool allowlist together means callers get a consistent,
//! self-describing configuration object rather than a bag of separate flags.
//!
//! # Phase reference
//! Introduced in Phase 3 of PLAN.md: "Skill & ProjectContext wiring".

use tracing::debug;

// ---------------------------------------------------------------------------
// Skill struct
// ---------------------------------------------------------------------------

/// A named bundle that generates the system prompt and declares which tools it uses.
///
/// # Example
/// ```ignore
/// let skill = Skill::code_writer();
/// let prompt = skill.build_system_prompt(Some(&ctx.format_for_prompt()));
/// let agent  = AgentLoop::new(backend, registry, &prompt, 20);
/// ```
#[derive(Debug, Clone)]
pub struct Skill {
    /// Unique slug used to select the skill from the CLI (e.g. `"chat"`, `"code"`).
    pub name: String,

    /// Short, human-readable description shown in the REPL header.
    pub description: String,

    /// Base system prompt template.
    ///
    /// Call [`build_system_prompt`][Skill::build_system_prompt] rather than reading
    /// this field directly so that project context is appended correctly.
    pub system_prompt: String,

    /// Tool names this skill is allowed to invoke.
    ///
    /// An **empty** `Vec` means "all registered tools are allowed" — the skill
    /// imposes no restriction. A non-empty list acts as an allowlist.
    pub allowed_tools: Vec<String>,

    /// Optional override for the Ollama model name.
    ///
    /// When `Some`, the agent loop should prefer this model over the global
    /// config value. `None` means "use whatever the config says".
    pub model_override: Option<String>,

    /// Optional override for the sampling temperature (0.0 – 2.0).
    ///
    /// Useful for skills that need deterministic output (temperature → 0)
    /// or creative generation (temperature → 1+).
    pub temperature_override: Option<f32>,
}

// ---------------------------------------------------------------------------
// Core methods
// ---------------------------------------------------------------------------

impl Skill {
    /// Returns the full system prompt, optionally enriched with project context.
    ///
    /// When `project_context` is `Some`, its contents are appended after the
    /// base prompt separated by a blank line.  This lets the LLM see both the
    /// behavioural instructions *and* the workspace structure without any extra
    /// token budget tricks.
    ///
    /// # Pattern: Strategy / template method
    /// The prompt "template" lives in `system_prompt`; this method fills in
    /// the optional slot at the end.  Callers never concatenate strings
    /// manually.
    pub fn build_system_prompt(&self, project_context: Option<&str>) -> String {
        match project_context {
            None => {
                debug!(skill = %self.name, "Building system prompt (no project context)");
                self.system_prompt.clone()
            }
            Some(ctx) => {
                debug!(skill = %self.name, "Building system prompt with project context");
                // Two newlines produce a blank separator line — keeps the two
                // sections visually distinct in log output.
                format!("{}\n\n{}", self.system_prompt, ctx)
            }
        }
    }

    /// Returns `true` if `tool_name` is permitted by this skill.
    ///
    /// The rule: **empty `allowed_tools` ⇒ every tool is allowed.**
    /// A non-empty list acts as an explicit allowlist.
    ///
    /// This is used by the agent loop (or a wrapping layer) to filter the tool
    /// definitions sent to the LLM before each inference call.
    pub fn allows_tool(&self, tool_name: &str) -> bool {
        if self.allowed_tools.is_empty() {
            // Empty list → no restriction → all tools allowed.
            true
        } else {
            self.allowed_tools.iter().any(|t| t == tool_name)
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in constructors
// ---------------------------------------------------------------------------

impl Skill {
    /// General-purpose chat skill.
    ///
    /// Uses a simple, friendly system prompt and imposes no tool restrictions —
    /// the LLM may use any registered tool.  Good default for interactive
    /// sessions without a specific coding goal.
    pub fn chat() -> Self {
        Self {
            name: "chat".to_string(),
            description: "General-purpose helpful assistant".to_string(),
            system_prompt: CHAT_PROMPT.to_string(),
            allowed_tools: vec![],
            model_override: None,
            temperature_override: None,
        }
    }

    /// Expert software-engineering skill, tuned for code tasks.
    ///
    /// Provides detailed workflow guidance (read-before-edit, surgical patches,
    /// test-after-write) and Windows-specific `cargo --target-dir` advice so
    /// the LLM doesn't silently create broken build artefacts.
    ///
    /// All tools are allowed — the prompt itself guides *how* to use them.
    pub fn code_writer() -> Self {
        Self {
            name: "code".to_string(),
            description: "Expert software engineer and coding assistant".to_string(),
            system_prompt: CODE_WRITER_PROMPT.to_string(),
            allowed_tools: vec![],
            model_override: None,
            temperature_override: None,
        }
    }

    /// Look up a built-in skill by its slug.
    ///
    /// Returns `None` if the name is not recognised.
    ///
    /// Currently registered slugs: `"chat"`, `"code"`.
    pub fn builtin(name: &str) -> Option<Self> {
        match name {
            "chat" => Some(Self::chat()),
            "code" => Some(Self::code_writer()),
            _ => {
                debug!(skill_name = %name, "Unknown built-in skill requested");
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// System prompt constants
// ---------------------------------------------------------------------------

/// System prompt for the [`Skill::chat`] skill.
///
/// Deliberately concise — we don't want to waste context window on
/// code-specific instructions when the user just wants a conversation.
const CHAT_PROMPT: &str = "\
You are a helpful AI assistant running locally via ragent.
You have access to tools for reading files, writing files, patching files, \
searching files, and executing shell commands.
Use tools when appropriate to accomplish tasks. Be concise and helpful.";

/// System prompt for the [`Skill::code_writer`] skill.
///
/// Kept as a raw string (`r#"..."#`) so that embedded double-quotes and
/// Windows backslash paths don't require escaping.  This prompt encodes the
/// full workflow the LLM must follow to be a reliable coding assistant.
///
/// # Why so detailed?
/// LLMs hallucinate file contents and paths far less often when given an
/// explicit read-before-edit contract.  The `file_patch` disambiguation rule
/// ("if it matches N times, add more context") prevents a common failure mode
/// where the model sends a patch that matches multiple locations and corrupts
/// the file.
const CODE_WRITER_PROMPT: &str = r#"You are an expert software engineer and coding assistant running locally via ragent.

## Workflow (follow this carefully)

1. **Read before you modify** — always use `file_read` to understand existing code before changing it. Never assume the contents of a file.
2. **Find before you assume** — use `file_search` to locate relevant files rather than guessing paths.
3. **Surgical edits for existing files** — use `file_patch` with a small, unique snippet of surrounding context (2–3 lines). The `old_text` must appear *exactly once* in the file.
4. **New files** — use `file_write` to create them (or to completely replace an existing file's content when a patch would be impractical).
5. **Test after every change** — run the test suite with `shell_exec` to verify correctness. Fix any failures before reporting success to the user.

## Tool guidance

- `file_read` — read files to understand existing code. Do this first, always.
- `file_patch` — surgical find/replace edits. Rules:
  - `old_text` must appear **exactly once** in the file. If it matches multiple times, add more surrounding lines until it is unique.
  - Include 2–3 lines of context around the changed lines so the match is unambiguous.
  - Prefer small, focused patches over large rewrites.
- `file_write` — create new files, or fully replace a file when a patch would be unwieldy.
- `file_search` — find files by name or content pattern before assuming their location.
- `shell_exec` — run arbitrary shell commands. For Rust projects **always** pass `--target-dir` to avoid Windows MAX_PATH issues:
  ```
  cargo build --target-dir C:\Temp\ragent-target
  cargo test  --target-dir C:\Temp\ragent-target
  ```

## Error recovery

- If `file_patch` reports "text appears N times", add more surrounding context to `old_text` until it is unique.
- If a shell command fails, read the error output carefully and try a different approach. Do not repeat the same failing command.
- If you are genuinely stuck after two attempts, explain clearly what you have tried and ask the user for clarification.

## Style

- Be methodical: state what you are about to do, do it, then report the result.
- Prefer small, focused changes over large rewrites. Smaller diffs are easier to review and less likely to introduce bugs.
- Write clear, idiomatic code in the target language. Follow the conventions of the existing codebase."#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Metadata ---

    #[test]
    fn test_chat_skill_metadata() {
        let skill = Skill::chat();
        assert_eq!(skill.name, "chat");
        assert!(
            !skill.description.is_empty(),
            "description should be non-empty"
        );
        // Empty allowed_tools means "all tools are allowed".
        assert!(
            skill.allowed_tools.is_empty(),
            "chat should not restrict tools"
        );
        // allows_tool must return true for any name when the allowlist is empty.
        assert!(skill.allows_tool("file_read"));
        assert!(skill.allows_tool("shell_exec"));
        assert!(skill.allows_tool("some_future_tool_that_doesnt_exist_yet"));
    }

    #[test]
    fn test_code_writer_metadata() {
        let skill = Skill::code_writer();
        assert_eq!(skill.name, "code");
        assert!(
            !skill.description.is_empty(),
            "description should be non-empty"
        );
        // The prompt must reference each of the three critical tools by name
        // so the LLM knows they exist.
        assert!(
            skill.system_prompt.contains("file_patch"),
            "code_writer prompt must mention file_patch"
        );
        assert!(
            skill.system_prompt.contains("file_read"),
            "code_writer prompt must mention file_read"
        );
        assert!(
            skill.system_prompt.contains("shell_exec"),
            "code_writer prompt must mention shell_exec"
        );
    }

    // --- allows_tool ---

    #[test]
    fn test_allows_tool_with_empty_list_permits_everything() {
        let skill = Skill::chat(); // allowed_tools is empty
        assert!(skill.allows_tool("anything"));
        assert!(skill.allows_tool(""));
    }

    #[test]
    fn test_allows_tool_with_explicit_list() {
        let skill = Skill {
            name: "restricted".to_string(),
            description: "test".to_string(),
            system_prompt: "test".to_string(),
            allowed_tools: vec!["file_read".to_string(), "file_write".to_string()],
            model_override: None,
            temperature_override: None,
        };
        assert!(skill.allows_tool("file_read"));
        assert!(skill.allows_tool("file_write"));
        assert!(
            !skill.allows_tool("shell_exec"),
            "shell_exec should be blocked"
        );
        assert!(
            !skill.allows_tool("file_patch"),
            "file_patch should be blocked"
        );
    }

    // --- build_system_prompt ---

    #[test]
    fn test_build_system_prompt_no_context() {
        let skill = Skill::chat();
        let prompt = skill.build_system_prompt(None);
        // Must be identical to the raw system_prompt — no extra content.
        assert_eq!(prompt, skill.system_prompt);
    }

    #[test]
    fn test_build_system_prompt_with_context() {
        let skill = Skill::chat();
        let ctx = "## Project Context\n**Type**: Rust (workspace)";
        let prompt = skill.build_system_prompt(Some(ctx));
        // Both the base prompt and the context must appear in the output.
        assert!(
            prompt.contains(&skill.system_prompt),
            "base prompt should be present in combined prompt"
        );
        assert!(
            prompt.contains(ctx),
            "project context should be appended to prompt"
        );
        // The context must come *after* the base prompt.
        let base_pos = prompt.find(&skill.system_prompt).unwrap();
        let ctx_pos = prompt.find(ctx).unwrap();
        assert!(
            ctx_pos > base_pos,
            "context should appear after the base prompt"
        );
    }

    // --- builtin ---

    #[test]
    fn test_builtin_chat() {
        let skill = Skill::builtin("chat");
        assert!(
            skill.is_some(),
            "Skill::builtin(\"chat\") should return Some"
        );
        assert_eq!(skill.unwrap().name, "chat");
    }

    #[test]
    fn test_builtin_code() {
        let skill = Skill::builtin("code");
        assert!(
            skill.is_some(),
            "Skill::builtin(\"code\") should return Some"
        );
        assert_eq!(skill.unwrap().name, "code");
    }

    #[test]
    fn test_builtin_unknown() {
        let skill = Skill::builtin("nonexistent");
        assert!(
            skill.is_none(),
            "Skill::builtin with an unknown name should return None"
        );
    }
}
