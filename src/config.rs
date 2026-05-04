//! Configuration loading and types.
//!
//! Loads settings from `config/default.toml` with optional CLI overrides.
//! Uses the simple approach of `toml` + `serde` deserialization.
//!
//! # Phase 2 additions
//! Added `AgentConfig` (max iterations, system prompt) and `ToolsConfig`
//! (shell timeout/limits, file sandbox/limits) to support the agent loop
//! and tool system.
//!
//! # Phase 3 additions
//! Added `SkillConfig` (which built-in skill to activate) and `ProjectConfig`
//! (project context auto-detection settings for injecting file tree + key files
//! into the system prompt). Also added `cargo_target_dir` to `ShellConfig`
//! to set `CARGO_TARGET_DIR` automatically and avoid Windows MAX_PATH issues.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Which LLM backend ragent should use.
///
/// Select with `backend = "openai"` in `config/default.toml`, or override
/// per-run with `--backend openai`. The `openai` variant works with any
/// OpenAI-compatible server: LM Studio, Jan, a remote OpenAI key, etc.
#[derive(Debug, Deserialize, Clone, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    #[default]
    Ollama,
    OpenAi,
}

/// Top-level application configuration.
///
/// Mirrors the structure of `config/default.toml`.
#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    /// Which LLM backend to use. Defaults to `"ollama"`.
    #[serde(default)]
    pub backend: BackendType,

    pub ollama: OllamaConfig,

    /// OpenAI-compatible backend (LM Studio, OpenAI, etc.).
    #[serde(default)]
    pub openai: OpenAiConfig,

    /// Agent loop configuration. Optional in the TOML — uses defaults if absent.
    #[serde(default)]
    pub agent: AgentConfig,

    /// Tool-specific configuration. Optional in the TOML — uses defaults if absent.
    #[serde(default)]
    pub tools: ToolsConfig,

    /// Skill selection. Optional — defaults to the "code" skill.
    #[serde(default)]
    pub skill: SkillConfig,

    /// Project context loading. Optional — auto-detection is on by default.
    #[serde(default)]
    pub project: ProjectConfig,
}

/// Configuration for the Ollama LLM backend.
#[derive(Debug, Deserialize, Clone)]
pub struct OllamaConfig {
    /// URL where Ollama is running (e.g. `http://localhost:11434`).
    pub url: String,

    /// Model name to use (must be pulled in Ollama, e.g. `qwen2.5-coder:7b`).
    pub model: String,

    /// LLM temperature (0.0 = deterministic, 1.0 = creative).
    /// `None` means use Ollama's default.
    pub temperature: Option<f32>,

    /// Context window size in tokens.
    /// `None` means use the model's default.
    pub context_window: Option<u32>,

    /// How long (in seconds) to wait for Ollama to respond before giving up.
    ///
    /// This covers the full round-trip: connecting + sending the request +
    /// waiting for Ollama to finish the **prefill** phase (processing all input
    /// tokens) and return the first response byte.
    ///
    /// On CPU, prefill time grows with model size × context length. A 7B model
    /// with a large project-context system prompt can easily take several minutes
    /// before generating its first token. Increase this if you see
    /// "operation timed out" errors. Defaults to 600 seconds (10 minutes).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
}

fn default_request_timeout_secs() -> u64 {
    600 // 10 minutes — generous for CPU inference with large contexts
}

// ---------------------------------------------------------------------------
// OpenAI-compatible backend configuration
// ---------------------------------------------------------------------------

/// Configuration for any OpenAI-compatible LLM backend.
///
/// Works with LM Studio (`http://localhost:1234/v1`), Jan, a remote OpenAI
/// API key, or any other server that speaks the `/v1/chat/completions` format.
#[derive(Debug, Deserialize, Clone)]
pub struct OpenAiConfig {
    /// Base URL of the OpenAI-compatible API, WITHOUT a trailing slash.
    /// LM Studio default: `http://localhost:1234/v1`
    #[serde(default = "default_openai_url")]
    pub url: String,

    /// Model identifier as shown in the server's model list.
    /// In LM Studio this is the filename/path shown in the model loader.
    #[serde(default = "default_openai_model")]
    pub model: String,

    /// API key. Optional for local servers (LM Studio accepts any value).
    /// Required for remote OpenAI / Anthropic / etc.
    pub api_key: Option<String>,

    /// LLM temperature (0.0 = deterministic, 1.0 = creative).
    pub temperature: Option<f32>,

    /// Request timeout in seconds (same semantics as `OllamaConfig::request_timeout_secs`).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
}

fn default_openai_url() -> String {
    "http://localhost:1234/v1".to_string()
}

fn default_openai_model() -> String {
    "local-model".to_string()
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            url: default_openai_url(),
            model: default_openai_model(),
            api_key: None,
            temperature: None,
            request_timeout_secs: default_request_timeout_secs(),
        }
    }
}

/// Configuration for the agent loop.
#[derive(Debug, Deserialize, Clone)]
pub struct AgentConfig {
    /// Maximum number of LLM round-trips (tool-call iterations) per user message.
    /// This is a safety limit to prevent infinite loops if the LLM keeps calling tools.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// System prompt that defines the agent's behaviour.
    /// Sent as the first message in every conversation.
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
}

fn default_max_iterations() -> usize {
    20
}

fn default_system_prompt() -> String {
    "You are a helpful AI assistant running locally via ragent. \
     You have access to tools for reading files, writing files, searching files, \
     and executing shell commands. Use them when appropriate to accomplish tasks. \
     Be concise and helpful."
        .to_string()
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: default_max_iterations(),
            system_prompt: default_system_prompt(),
        }
    }
}

/// Tool-specific configuration.
///
/// Each tool category has its own sub-section. All fields have sensible defaults
/// so the `[tools]` section can be omitted entirely from the TOML.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct ToolsConfig {
    /// Shell execution tool configuration.
    #[serde(default)]
    pub shell: ShellConfig,

    /// File operation tools configuration.
    #[serde(default)]
    pub file: FileConfig,
}

/// Configuration for the `ShellExecTool`.
#[derive(Debug, Deserialize, Clone)]
pub struct ShellConfig {
    /// Maximum time (in seconds) a shell command is allowed to run.
    #[serde(default = "default_shell_timeout_secs")]
    pub timeout_secs: u64,

    /// Maximum output size (in bytes) captured from shell commands.
    /// Output beyond this limit is truncated with a warning.
    #[serde(default = "default_shell_max_output_bytes")]
    pub max_output_bytes: usize,

    /// Working directory for shell commands.
    /// `None` means use the current directory.
    pub working_directory: Option<String>,

    /// Override for the `CARGO_TARGET_DIR` environment variable.
    ///
    /// When set, all shell commands run with `CARGO_TARGET_DIR` pointing to
    /// this path. This is critical on Windows where deeply nested `target/`
    /// directories easily exceed the legacy 260-character `MAX_PATH` limit.
    ///
    /// Automatically set to `{TEMP}/ragent-target` when a workspace is active
    /// (unless explicitly configured here). Set to `null` to disable.
    pub cargo_target_dir: Option<String>,
}

fn default_shell_timeout_secs() -> u64 {
    30
}

fn default_shell_max_output_bytes() -> usize {
    102_400 // 100 KB
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            timeout_secs: default_shell_timeout_secs(),
            max_output_bytes: default_shell_max_output_bytes(),
            working_directory: None,
            cargo_target_dir: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Skill configuration (Phase 3)
// ---------------------------------------------------------------------------

/// Configuration for skill selection.
///
/// A skill is a named bundle of (system prompt + tool allowlist) that tunes
/// the agent for a specific task domain. Built-in skills:
/// - `"chat"` — general-purpose assistant, all tools, simple prompt
/// - `"code"` — software development, all tools, detailed coding workflow prompt
#[derive(Debug, Deserialize, Clone)]
pub struct SkillConfig {
    /// Name of the built-in skill to activate.
    /// Can be overridden with the `--skill` CLI flag.
    #[serde(default = "default_skill_name")]
    pub name: String,
}

fn default_skill_name() -> String {
    "code".to_string()
}

impl Default for SkillConfig {
    fn default() -> Self {
        Self {
            name: default_skill_name(),
        }
    }
}

// ---------------------------------------------------------------------------
// Project context configuration (Phase 3)
// ---------------------------------------------------------------------------

/// Configuration for automatic project context injection.
///
/// When a workspace is set and `auto_detect` is true, ragent reads the project's
/// directory structure and key files (Cargo.toml, README, etc.) and injects them
/// into the system prompt. This gives the LLM a mental model of the project
/// before it starts working, reducing the need to explore manually.
#[derive(Debug, Deserialize, Clone)]
pub struct ProjectConfig {
    /// Whether to automatically detect and inject project context.
    /// Disable this if you want a clean prompt with no pre-loaded context.
    #[serde(default = "default_auto_detect")]
    pub auto_detect: bool,

    /// Maximum directory depth for the file tree (default: 4).
    /// Deeper directories are omitted from the tree.
    #[serde(default = "default_max_tree_depth")]
    pub max_tree_depth: usize,

    /// Maximum bytes to include from each key file (default: 16 KB).
    /// Files larger than this are truncated with a notice.
    #[serde(default = "default_max_key_file_bytes")]
    pub max_key_file_bytes: usize,
}

fn default_auto_detect() -> bool {
    true
}

fn default_max_tree_depth() -> usize {
    4
}

fn default_max_key_file_bytes() -> usize {
    16_384 // 16 KB
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            auto_detect: default_auto_detect(),
            max_tree_depth: default_max_tree_depth(),
            max_key_file_bytes: default_max_key_file_bytes(),
        }
    }
}

/// Configuration for file operation tools (`FileReadTool`, `FileWriteTool`, `FileSearchTool`).
#[derive(Debug, Deserialize, Clone)]
pub struct FileConfig {
    /// Maximum file size (in bytes) that the read tool will return.
    /// Files larger than this are rejected with an error suggesting a search instead.
    #[serde(default = "default_file_max_read_bytes")]
    pub max_read_bytes: usize,

    /// Maximum number of search results returned by the file search tool.
    #[serde(default = "default_file_max_search_results")]
    pub max_search_results: usize,

    /// Base directory that file tools are restricted to (sandbox).
    /// All file paths are resolved relative to this and must stay within it.
    /// `None` means use the current directory.
    pub sandbox_path: Option<String>,
}

fn default_file_max_read_bytes() -> usize {
    1_048_576 // 1 MB
}

fn default_file_max_search_results() -> usize {
    50
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            max_read_bytes: default_file_max_read_bytes(),
            max_search_results: default_file_max_search_results(),
            sandbox_path: None,
        }
    }
}

impl AppConfig {
    /// Load configuration from the default TOML file.
    ///
    /// Looks for `config/default.toml` relative to the current working directory.
    /// This is the simplest approach for Phase 1/2 — later we can add layered config
    /// (project-local ragent.toml, environment variables, etc.).
    pub fn load() -> Result<Self> {
        let config_path = Path::new("config/default.toml");

        if !config_path.exists() {
            anyhow::bail!(
                "Config file not found at '{}'. \
                 Make sure you're running from the ragent project root.",
                config_path.display()
            );
        }

        let content =
            std::fs::read_to_string(config_path).context("Failed to read config/default.toml")?;

        let config: AppConfig =
            toml::from_str(&content).context("Failed to parse config/default.toml")?;

        Ok(config)
    }

    /// Create config with explicit overrides (from CLI args).
    ///
    /// Starts from the file-based config and overrides individual fields
    /// when the user provides them via CLI flags.
    pub fn with_overrides(
        mut self,
        model: Option<String>,
        ollama_url: Option<String>,
        skill: Option<String>,
    ) -> Self {
        if let Some(m) = model {
            self.ollama.model = m;
        }
        if let Some(u) = ollama_url {
            self.ollama.url = u;
        }
        if let Some(s) = skill {
            self.skill.name = s;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_deserialization() {
        let toml_str = r#"
            [ollama]
            url = "http://localhost:11434"
            model = "test-model"
            temperature = 0.5
        "#;

        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.ollama.url, "http://localhost:11434");
        assert_eq!(config.ollama.model, "test-model");
        assert_eq!(config.ollama.temperature, Some(0.5));
        assert_eq!(config.ollama.context_window, None);
        // Agent and tools should get defaults
        assert_eq!(config.agent.max_iterations, 20);
        assert_eq!(config.tools.shell.timeout_secs, 30);
        assert_eq!(config.tools.file.max_read_bytes, 1_048_576);
    }

    #[test]
    fn test_config_with_agent_section() {
        let toml_str = r#"
            [ollama]
            url = "http://localhost:11434"
            model = "test-model"

            [agent]
            max_iterations = 10
            system_prompt = "Custom prompt"
        "#;

        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.agent.max_iterations, 10);
        assert_eq!(config.agent.system_prompt, "Custom prompt");
    }

    #[test]
    fn test_config_with_tools_section() {
        let toml_str = r#"
            [ollama]
            url = "http://localhost:11434"
            model = "test-model"

            [tools.shell]
            timeout_secs = 60
            max_output_bytes = 204800
            working_directory = "/tmp"

            [tools.file]
            max_read_bytes = 512000
            max_search_results = 100
            sandbox_path = "/home/user/projects"
        "#;

        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.tools.shell.timeout_secs, 60);
        assert_eq!(config.tools.shell.max_output_bytes, 204800);
        assert_eq!(
            config.tools.shell.working_directory,
            Some("/tmp".to_string())
        );
        assert_eq!(config.tools.file.max_read_bytes, 512000);
        assert_eq!(config.tools.file.max_search_results, 100);
        assert_eq!(
            config.tools.file.sandbox_path,
            Some("/home/user/projects".to_string())
        );
    }

    #[test]
    fn test_config_overrides() {
        let config = AppConfig {
            backend: BackendType::default(),
            ollama: OllamaConfig {
                url: "http://localhost:11434".to_string(),
                model: "default-model".to_string(),
                temperature: None,
                context_window: None,
                request_timeout_secs: default_request_timeout_secs(),
            },
            openai: OpenAiConfig::default(),
            agent: AgentConfig::default(),
            tools: ToolsConfig::default(),
            skill: SkillConfig::default(),
            project: ProjectConfig::default(),
        };

        let overridden = config.with_overrides(
            Some("custom-model".to_string()),
            Some("http://remote:11434".to_string()),
            None,
        );

        assert_eq!(overridden.ollama.model, "custom-model");
        assert_eq!(overridden.ollama.url, "http://remote:11434");
    }

    #[test]
    fn test_config_minimal_toml() {
        // Only required fields, no optional ones — everything else gets defaults.
        let toml_str = r#"
            [ollama]
            url = "http://localhost:11434"
            model = "llama3.2"
        "#;

        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.ollama.model, "llama3.2");
        assert!(config.ollama.temperature.is_none());
        // Defaults
        assert_eq!(config.agent.max_iterations, 20);
        assert!(!config.agent.system_prompt.is_empty());
        assert_eq!(config.tools.shell.timeout_secs, 30);
        assert_eq!(config.tools.file.max_read_bytes, 1_048_576);
        assert_eq!(config.tools.file.max_search_results, 50);
        assert!(config.tools.shell.working_directory.is_none());
        assert!(config.tools.file.sandbox_path.is_none());
    }

    #[test]
    fn test_default_agent_config() {
        let config = AgentConfig::default();
        assert_eq!(config.max_iterations, 20);
        assert!(config.system_prompt.contains("helpful"));
    }

    #[test]
    fn test_default_shell_config() {
        let config = ShellConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.max_output_bytes, 102_400);
        assert!(config.working_directory.is_none());
    }

    #[test]
    fn test_default_file_config() {
        let config = FileConfig::default();
        assert_eq!(config.max_read_bytes, 1_048_576);
        assert_eq!(config.max_search_results, 50);
        assert!(config.sandbox_path.is_none());
    }
}
