//! Configuration loading and types.
//!
//! Loads settings from `config/default.toml` with optional CLI overrides.
//! Uses the simple approach of `toml` + `serde` deserialization.
//!
//! # Phase 2 additions
//! Added `AgentConfig` (max iterations, system prompt) and `ToolsConfig`
//! (shell timeout/limits, file sandbox/limits) to support the agent loop
//! and tool system.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Top-level application configuration.
///
/// Mirrors the structure of `config/default.toml`.
#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub ollama: OllamaConfig,

    /// Agent loop configuration. Optional in the TOML — uses defaults if absent.
    #[serde(default)]
    pub agent: AgentConfig,

    /// Tool-specific configuration. Optional in the TOML — uses defaults if absent.
    #[serde(default)]
    pub tools: ToolsConfig,
}

/// Configuration for the Ollama LLM backend.
#[derive(Debug, Deserialize, Clone)]
pub struct OllamaConfig {
    /// URL where Ollama is running (e.g. `http://localhost:11434`).
    pub url: String,

    /// Model name to use (must be pulled in Ollama, e.g. `deepseek-r1:8b`).
    pub model: String,

    /// LLM temperature (0.0 = deterministic, 1.0 = creative).
    /// `None` means use Ollama's default.
    pub temperature: Option<f32>,

    /// Context window size in tokens.
    /// `None` means use the model's default.
    pub context_window: Option<u32>,
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
    pub fn with_overrides(mut self, model: Option<String>, ollama_url: Option<String>) -> Self {
        if let Some(m) = model {
            self.ollama.model = m;
        }
        if let Some(u) = ollama_url {
            self.ollama.url = u;
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
            ollama: OllamaConfig {
                url: "http://localhost:11434".to_string(),
                model: "default-model".to_string(),
                temperature: None,
                context_window: None,
            },
            agent: AgentConfig::default(),
            tools: ToolsConfig::default(),
        };

        let overridden = config.with_overrides(
            Some("custom-model".to_string()),
            Some("http://remote:11434".to_string()),
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
