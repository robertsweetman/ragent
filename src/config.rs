//! Configuration loading and types.
//!
//! Loads settings from `config/default.toml` with optional CLI overrides.
//! Uses the simple approach of `toml` + `serde` deserialization.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// Top-level application configuration.
///
/// Mirrors the structure of `config/default.toml`.
#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub ollama: OllamaConfig,
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

impl AppConfig {
    /// Load configuration from the default TOML file.
    ///
    /// Looks for `config/default.toml` relative to the current working directory.
    /// This is the simplest approach for Phase 1 — later we can add layered config
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
        let toml_str = r#"
            [ollama]
            url = "http://localhost:11434"
            model = "llama3.2"
        "#;

        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.ollama.model, "llama3.2");
        assert!(config.ollama.temperature.is_none());
    }
}
