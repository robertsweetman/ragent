//! Shell command execution tool — runs commands via the system shell.
//!
//! # Phase 2 (PLAN.md)
//!
//! The `ShellExecTool` lets the agent run arbitrary shell commands to compile
//! code, run tests, inspect the filesystem, install packages, etc. It's one
//! of the most powerful (and dangerous) tools, so it has several safety limits:
//!
//! - **Timeout**: Commands are killed after a configurable duration (default 30s).
//! - **Output size cap**: stdout + stderr are truncated to prevent the LLM's
//!   context window from being flooded by massive build logs.
//! - **Working directory**: Configurable per-session so the agent operates in
//!   the right project root.
//!
//! # Security Considerations (from PLAN_CRITIQUE.md)
//!
//! This tool intentionally does NOT sandbox commands — it runs them with the
//! same permissions as the ragent process. For Phase 2 this is acceptable
//! (local-only, single-user). Future phases could add:
//! - Command allowlist/blocklist
//! - Per-project sandboxing (e.g. `bwrap`, `firejail`, containers)
//! - Resource limits (cgroups on Linux)

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use tracing::{debug, info, instrument, warn};

use crate::tools::traits::{Tool, ToolResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the shell execution tool.
///
/// Loaded from `[tools.shell]` in `config/default.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct ShellConfig {
    /// Maximum time a command is allowed to run before being killed.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,

    /// Maximum number of bytes captured from command output (stdout + stderr).
    /// Output beyond this limit is truncated with a warning message appended.
    #[serde(default = "default_max_output_bytes")]
    pub max_output_bytes: usize,

    /// Working directory for commands. If `None`, uses the current directory.
    pub working_directory: Option<String>,
}

fn default_timeout_secs() -> u64 {
    30
}

fn default_max_output_bytes() -> usize {
    102_400 // 100 KB
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            timeout_secs: default_timeout_secs(),
            max_output_bytes: default_max_output_bytes(),
            working_directory: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ShellExecTool
// ---------------------------------------------------------------------------

/// Tool that executes shell commands and returns their output.
///
/// The command is run via the system shell (`sh -c` on Unix, `cmd /C` on
/// Windows) so shell features like pipes, redirects, and environment variable
/// expansion work as expected.
pub struct ShellExecTool {
    config: ShellConfig,
    /// Resolved working directory (absolute path).
    working_dir: PathBuf,
}

impl ShellExecTool {
    /// Create a new shell execution tool with the given configuration.
    ///
    /// The working directory is resolved to an absolute path at construction
    /// time. If the configured directory doesn't exist, a warning is logged
    /// and the current directory is used as a fallback.
    pub fn new(config: ShellConfig) -> Self {
        let working_dir = resolve_working_dir(&config.working_directory);

        info!(
            working_dir = %working_dir.display(),
            timeout_secs = config.timeout_secs,
            max_output_bytes = config.max_output_bytes,
            "ShellExecTool initialised"
        );

        Self {
            config,
            working_dir,
        }
    }

    /// Execute a command string with timeout and output limits.
    ///
    /// Returns the combined stdout + stderr as a string, truncated if it
    /// exceeds `max_output_bytes`.
    async fn run_command(&self, command: &str, working_dir: &Path) -> Result<ToolResult> {
        debug!(
            command = %command,
            working_dir = %working_dir.display(),
            "Executing shell command"
        );

        // Build the OS-appropriate shell invocation.
        // On Windows: cmd /C "command"
        // On Unix:    sh -c "command"
        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = tokio::process::Command::new("cmd");
            c.arg("/C").arg(command);
            c
        } else {
            let mut c = tokio::process::Command::new("sh");
            c.arg("-c").arg(command);
            c
        };

        cmd.current_dir(working_dir);

        // Capture both stdout and stderr so the LLM sees the full picture.
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        // Spawn the child process.
        let child = cmd.spawn().context("Failed to spawn shell command")?;

        // Wait for completion with timeout.
        let timeout = Duration::from_secs(self.config.timeout_secs);
        let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                return Ok(ToolResult::error(format!("Command failed to execute: {e}")));
            }
            Err(_) => {
                // Timeout expired. The child is dropped which sends SIGKILL on
                // Unix (or TerminateProcess on Windows) via tokio's Child Drop.
                warn!(
                    command = %command,
                    timeout_secs = self.config.timeout_secs,
                    "Shell command timed out"
                );
                return Ok(ToolResult::error(format!(
                    "Command timed out after {} seconds. \
                     Consider breaking it into smaller steps or increasing the timeout.",
                    self.config.timeout_secs
                )));
            }
        };

        // Combine stdout and stderr into a single output string.
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut combined = String::new();

        if !stdout.is_empty() {
            combined.push_str(&stdout);
        }

        if !stderr.is_empty() {
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str("[stderr]\n");
            combined.push_str(&stderr);
        }

        // Truncate if output is too large.
        let truncated = if combined.len() > self.config.max_output_bytes {
            let truncated_output = &combined[..self.config.max_output_bytes];
            format!(
                "{truncated_output}\n\n[OUTPUT TRUNCATED — {total} bytes total, \
                 showing first {limit} bytes]",
                total = combined.len(),
                limit = self.config.max_output_bytes,
            )
        } else {
            combined
        };

        let exit_code = output.status.code();
        let success = output.status.success();

        info!(
            command = %command,
            exit_code = ?exit_code,
            output_bytes = truncated.len(),
            success = success,
            "Shell command completed"
        );

        if success {
            if truncated.is_empty() {
                Ok(ToolResult::success("(command completed with no output)"))
            } else {
                Ok(ToolResult::success(truncated))
            }
        } else {
            let msg = format!(
                "Command exited with code {code}\n{output}",
                code = exit_code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "unknown (killed by signal)".to_string()),
                output = truncated,
            );
            Ok(ToolResult::error(msg))
        }
    }
}

impl Tool for ShellExecTool {
    fn name(&self) -> &str {
        "shell_exec"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output (stdout and stderr). \
         Commands run via the system shell (cmd on Windows, sh on Unix). \
         Use this to compile code, run tests, inspect the filesystem, \
         install packages, or perform any command-line task."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (e.g. 'cargo build', 'ls -la', 'cat file.txt')"
                },
                "working_directory": {
                    "type": "string",
                    "description": "Optional working directory for the command. If not provided, uses the default working directory."
                }
            },
            "required": ["command"]
        })
    }

    #[instrument(skip(self, args), fields(tool = "shell_exec"))]
    fn execute(
        &self,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
        Box::pin(async move {
            // Extract the command argument.
            let command = match args.get("command").and_then(|v| v.as_str()) {
                Some(cmd) if !cmd.trim().is_empty() => cmd,
                Some(_) => return Ok(ToolResult::error("Command cannot be empty.")),
                None => {
                    return Ok(ToolResult::error(
                        "Missing required parameter 'command'. \
                         Provide a shell command string to execute.",
                    ));
                }
            };

            // Determine working directory: per-call override > tool default.
            let working_dir = match args.get("working_directory").and_then(|v| v.as_str()) {
                Some(dir) => {
                    let path = PathBuf::from(dir);
                    if path.is_dir() {
                        path
                    } else {
                        return Ok(ToolResult::error(format!(
                            "Working directory does not exist: {dir}"
                        )));
                    }
                }
                None => self.working_dir.clone(),
            };

            self.run_command(command, &working_dir).await
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the working directory from the config, falling back to the current
/// directory if unset or invalid.
fn resolve_working_dir(configured: &Option<String>) -> PathBuf {
    if let Some(dir) = configured {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            match path.canonicalize() {
                Ok(abs) => return abs,
                Err(e) => {
                    warn!(
                        path = %path.display(),
                        error = %e,
                        "Configured working directory cannot be resolved, using current directory"
                    );
                }
            }
        } else {
            warn!(
                path = %path.display(),
                "Configured working directory does not exist, using current directory"
            );
        }
    }

    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tool() -> ShellExecTool {
        ShellExecTool::new(ShellConfig {
            timeout_secs: 5,
            max_output_bytes: 1024,
            working_directory: None,
        })
    }

    #[test]
    fn test_tool_metadata() {
        let tool = test_tool();
        assert_eq!(tool.name(), "shell_exec");
        assert!(!tool.description().is_empty());

        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["command"].is_object());
        assert_eq!(schema["required"][0], "command");
    }

    #[test]
    fn test_resolve_working_dir_none() {
        let dir = resolve_working_dir(&None);
        assert!(dir.is_dir());
    }

    #[test]
    fn test_resolve_working_dir_invalid() {
        let dir = resolve_working_dir(&Some("/nonexistent/path/xyz".to_string()));
        // Falls back to current dir
        assert!(dir.is_dir());
    }

    #[tokio::test]
    async fn test_execute_simple_command() {
        let tool = test_tool();
        let command = if cfg!(target_os = "windows") {
            "echo hello"
        } else {
            "echo hello"
        };
        let result = tool
            .execute(serde_json::json!({ "command": command }))
            .await
            .unwrap();
        assert!(!result.is_error, "Unexpected error: {}", result.content);
        assert!(
            result.content.contains("hello"),
            "Expected 'hello' in output, got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_execute_missing_command() {
        let tool = test_tool();
        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Missing required parameter"));
    }

    #[tokio::test]
    async fn test_execute_empty_command() {
        let tool = test_tool();
        let result = tool
            .execute(serde_json::json!({ "command": "  " }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("empty"));
    }

    #[tokio::test]
    async fn test_execute_failing_command() {
        let tool = test_tool();
        let command = if cfg!(target_os = "windows") {
            "cmd /C exit 1"
        } else {
            "exit 1"
        };
        let result = tool
            .execute(serde_json::json!({ "command": command }))
            .await
            .unwrap();
        assert!(result.is_error, "Expected error for failing command");
        assert!(result.content.contains("exited with code"));
    }

    #[tokio::test]
    async fn test_output_truncation() {
        let tool = ShellExecTool::new(ShellConfig {
            timeout_secs: 5,
            max_output_bytes: 20, // Very small limit
            working_directory: None,
        });

        let command = if cfg!(target_os = "windows") {
            "echo this is a long output string that should be truncated"
        } else {
            "echo 'this is a long output string that should be truncated'"
        };
        let result = tool
            .execute(serde_json::json!({ "command": command }))
            .await
            .unwrap();
        assert!(
            result.content.contains("TRUNCATED"),
            "Expected truncation notice in: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_timeout() {
        let tool = ShellExecTool::new(ShellConfig {
            timeout_secs: 1,
            max_output_bytes: 1024,
            working_directory: None,
        });

        let command = if cfg!(target_os = "windows") {
            "ping -n 10 127.0.0.1"
        } else {
            "sleep 10"
        };
        let result = tool
            .execute(serde_json::json!({ "command": command }))
            .await
            .unwrap();
        assert!(result.is_error, "Expected timeout error");
        assert!(
            result.content.contains("timed out"),
            "Expected 'timed out' in: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_invalid_working_directory() {
        let tool = test_tool();
        let result = tool
            .execute(serde_json::json!({
                "command": "echo hi",
                "working_directory": "/nonexistent/dir/xyz"
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("does not exist"));
    }
}
