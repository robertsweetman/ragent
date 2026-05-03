//! File operation tools — read, write, and search (grep) files.
//!
//! # Phase 2 (PLAN.md)
//! These are the first concrete tool implementations. They give the agent
//! the ability to inspect and modify files on disk, which is essential for
//! the code-writing use case in Phase 3.
//!
//! # Security: Path Validation
//!
//! All file tools validate paths against a **sandbox directory**. Every path
//! is resolved to its canonical (absolute) form and checked to ensure it
//! stays within the sandbox. This prevents directory traversal attacks where
//! the LLM might try `../../etc/passwd` or similar.
//!
//! The sandbox defaults to the current working directory but can be configured
//! via `config/default.toml` under `[tools.file]`.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;
use tracing::{debug, info, warn};
use walkdir::{DirEntry, WalkDir};

use crate::tools::traits::{Tool, ToolResult};

// ---------------------------------------------------------------------------
// Shared configuration for file tools
// ---------------------------------------------------------------------------

/// Configuration shared by all file tools.
///
/// Extracted from `[tools.file]` in the TOML config. Each file tool holds
/// an `Arc<FileToolsConfig>` (or a clone — it's small enough to clone).
#[derive(Debug, Clone)]
pub struct FileToolsConfig {
    /// Maximum file size (in bytes) that `FileReadTool` will return.
    /// Files larger than this are rejected with an error message telling
    /// the LLM the size and suggesting it read a specific range.
    pub max_read_bytes: usize,

    /// Maximum number of search results from `FileSearchTool`.
    pub max_search_results: usize,

    /// The sandbox directory. All file paths must resolve to a location
    /// within this directory. Prevents directory traversal.
    pub sandbox_path: PathBuf,
}

impl Default for FileToolsConfig {
    fn default() -> Self {
        Self {
            max_read_bytes: 1_048_576, // 1 MB
            max_search_results: 50,
            sandbox_path: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        }
    }
}

// ---------------------------------------------------------------------------
// Path validation helper
// ---------------------------------------------------------------------------

/// Resolve a user-provided path and validate it's within the sandbox.
///
/// # Why this matters
///
/// The LLM controls the `path` argument. Without validation, it could request
/// `../../etc/shadow` or `/root/.ssh/id_rsa`. This function:
/// 1. Resolves the path relative to the sandbox (if relative)
/// 2. Canonicalises both paths (resolves symlinks, `..`, etc.)
/// 3. Checks that the resolved path starts with the sandbox path
///
/// # Errors
///
/// Returns a `ToolResult::error` if the path escapes the sandbox or can't
/// be resolved.
pub(crate) fn validate_path(
    raw_path: &str,
    sandbox: &Path,
) -> std::result::Result<PathBuf, ToolResult> {
    if raw_path.is_empty() {
        return Err(ToolResult::error("Path cannot be empty."));
    }

    let candidate = if Path::new(raw_path).is_absolute() {
        PathBuf::from(raw_path)
    } else {
        sandbox.join(raw_path)
    };

    // For reading, the path must exist so we can canonicalize directly.
    // For writing, the path (or even its parent) may not exist yet —
    // the write tool creates parent dirs on demand. We walk up the
    // ancestor chain to find the first directory that *does* exist,
    // canonicalize that, then re-append the remaining components.
    // This ensures the sandbox check works even for deeply nested new paths.
    let resolved = if candidate.exists() {
        candidate
            .canonicalize()
            .map_err(|e| ToolResult::error(format!("Cannot resolve path '{}': {}", raw_path, e)))?
    } else {
        // Walk up from the candidate until we find an existing ancestor.
        // Collect the "tail" components that don't exist yet.
        let mut existing_ancestor = candidate.clone();
        let mut tail_components: Vec<std::ffi::OsString> = Vec::new();

        while !existing_ancestor.exists() {
            if let Some(file_name) = existing_ancestor.file_name() {
                tail_components.push(file_name.to_os_string());
            } else {
                // Reached a root or empty path without finding anything.
                break;
            }
            existing_ancestor = existing_ancestor
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf();
        }

        // Canonicalize the existing ancestor (resolves symlinks, .., etc.)
        let canon_base = if existing_ancestor.exists() {
            existing_ancestor.canonicalize().map_err(|e| {
                ToolResult::error(format!("Cannot resolve ancestor of '{}': {}", raw_path, e))
            })?
        } else {
            existing_ancestor
        };

        // Re-append the tail components (in reverse — we collected them bottom-up).
        let mut resolved = canon_base;
        for component in tail_components.into_iter().rev() {
            resolved = resolved.join(component);
        }
        resolved
    };

    // Canonicalize the sandbox too so comparison is apples-to-apples.
    let canon_sandbox = sandbox
        .canonicalize()
        .unwrap_or_else(|_| sandbox.to_path_buf());

    if !resolved.starts_with(&canon_sandbox) {
        warn!(
            path = %resolved.display(),
            sandbox = %canon_sandbox.display(),
            "Path escapes sandbox — access denied"
        );
        return Err(ToolResult::error(format!(
            "Access denied: path '{}' is outside the allowed directory '{}'.",
            raw_path,
            canon_sandbox.display()
        )));
    }

    Ok(resolved)
}

// ===========================================================================
// FileReadTool
// ===========================================================================

/// Reads the contents of a file and returns it as text.
///
/// The LLM uses this to inspect existing code, configs, docs, etc.
pub struct FileReadTool {
    config: FileToolsConfig,
}

impl FileReadTool {
    pub fn new(config: FileToolsConfig) -> Self {
        Self { config }
    }
}

impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at the given path. Returns the file contents as text. \
         Use this to inspect existing code, configuration files, documentation, etc."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read (relative to the working directory)."
                }
            },
            "required": ["path"]
        })
    }

    #[allow(clippy::manual_async_fn)]
    fn execute(
        &self,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
        Box::pin(async move {
            let raw_path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            let _span = tracing::info_span!("file_read", path = %raw_path).entered();

            let resolved = match validate_path(raw_path, &self.config.sandbox_path) {
                Ok(p) => p,
                Err(tool_err) => return Ok(tool_err),
            };

            if !resolved.exists() {
                return Ok(ToolResult::error(format!("File not found: '{}'", raw_path)));
            }

            if !resolved.is_file() {
                return Ok(ToolResult::error(format!(
                    "'{}' is not a file (it may be a directory).",
                    raw_path
                )));
            }

            // Check file size before reading to avoid OOM on huge files.
            let metadata = std::fs::metadata(&resolved).context("Failed to read file metadata")?;
            let size = metadata.len() as usize;

            if size > self.config.max_read_bytes {
                return Ok(ToolResult::error(format!(
                    "File '{}' is {} bytes, which exceeds the {} byte limit. \
                     Consider reading a specific section or using file_search to find relevant lines.",
                    raw_path, size, self.config.max_read_bytes
                )));
            }

            let content = std::fs::read_to_string(&resolved);
            match content {
                Ok(text) => {
                    info!(
                        path = %raw_path,
                        bytes = text.len(),
                        "File read successfully"
                    );
                    Ok(ToolResult::success(text))
                }
                Err(e) => {
                    // Might be a binary file or permission issue
                    Ok(ToolResult::error(format!(
                        "Failed to read '{}': {}. The file may be binary or you may lack permissions.",
                        raw_path, e
                    )))
                }
            }
        })
    }
}

// ===========================================================================
// FileWriteTool
// ===========================================================================

/// Writes content to a file, creating it (and parent directories) if needed.
///
/// The LLM uses this to create new files or overwrite existing ones.
pub struct FileWriteTool {
    config: FileToolsConfig,
}

impl FileWriteTool {
    pub fn new(config: FileToolsConfig) -> Self {
        Self { config }
    }
}

impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "file_write"
    }

    fn description(&self) -> &str {
        "Write content to a file at the given path. Creates the file and any parent directories \
         if they don't exist. Overwrites the file if it already exists. Use this to create new \
         source files, update configurations, write scripts, etc."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to (relative to the working directory)."
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write to the file."
                }
            },
            "required": ["path", "content"]
        })
    }

    #[allow(clippy::manual_async_fn)]
    fn execute(
        &self,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
        Box::pin(async move {
            let raw_path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let content = args
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            let _span = tracing::info_span!("file_write", path = %raw_path, bytes = content.len())
                .entered();

            let resolved = match validate_path(raw_path, &self.config.sandbox_path) {
                Ok(p) => p,
                Err(tool_err) => return Ok(tool_err),
            };

            // Create parent directories if they don't exist.
            if let Some(parent) = resolved.parent() {
                if !parent.exists() {
                    debug!(dir = %parent.display(), "Creating parent directories");
                    std::fs::create_dir_all(parent).map_err(|e| {
                        anyhow::anyhow!("Failed to create directories for '{}': {}", raw_path, e)
                    })?;
                }
            }

            match std::fs::write(&resolved, content) {
                Ok(()) => {
                    info!(
                        path = %raw_path,
                        bytes = content.len(),
                        "File written successfully"
                    );
                    Ok(ToolResult::success(format!(
                        "Successfully wrote {} bytes to '{}'.",
                        content.len(),
                        raw_path
                    )))
                }
                Err(e) => Ok(ToolResult::error(format!(
                    "Failed to write '{}': {}",
                    raw_path, e
                ))),
            }
        })
    }
}

// ===========================================================================
// FileSearchTool
// ===========================================================================

/// Searches for a regex pattern across files in a directory (like `grep -rn`).
///
/// The LLM uses this to find relevant code, function definitions, usages, etc.
/// without reading entire files.
pub struct FileSearchTool {
    config: FileToolsConfig,
}

impl FileSearchTool {
    pub fn new(config: FileToolsConfig) -> Self {
        Self { config }
    }
}

impl Tool for FileSearchTool {
    fn name(&self) -> &str {
        "file_search"
    }

    fn description(&self) -> &str {
        "Search for a regex pattern in files under a directory (like grep -rn). Returns matching \
         lines with file paths and line numbers. Use this to find function definitions, usages, \
         error messages, TODO comments, etc. \
         Note: build artifact directories (target/, node_modules/, dist/, build/) and hidden \
         directories (.git/, etc.) are automatically skipped."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for."
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in (relative to the working directory). Defaults to '.' (current directory)."
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob-like filter for filenames (e.g. '*.rs', '*.toml'). If omitted, all text files are searched."
                }
            },
            "required": ["pattern"]
        })
    }

    #[allow(clippy::manual_async_fn)]
    fn execute(
        &self,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
        Box::pin(async move {
            let pattern_str = args
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let search_path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");
            let file_pattern = args.get("file_pattern").and_then(|v| v.as_str());

            let _span = tracing::info_span!(
                "file_search",
                pattern = %pattern_str,
                path = %search_path
            )
            .entered();

            if pattern_str.is_empty() {
                return Ok(ToolResult::error("Search pattern cannot be empty."));
            }

            // Compile the regex. If the LLM sends invalid regex, report it
            // as a tool error so it can try again.
            let regex = match Regex::new(pattern_str) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(ToolResult::error(format!(
                        "Invalid regex pattern '{}': {}",
                        pattern_str, e
                    )));
                }
            };

            let resolved_dir = match validate_path(search_path, &self.config.sandbox_path) {
                Ok(p) => p,
                Err(tool_err) => return Ok(tool_err),
            };

            if !resolved_dir.exists() {
                return Ok(ToolResult::error(format!(
                    "Directory not found: '{}'",
                    search_path
                )));
            }

            if !resolved_dir.is_dir() {
                return Ok(ToolResult::error(format!(
                    "'{}' is not a directory.",
                    search_path
                )));
            }

            let mut matches: Vec<String> = Vec::new();
            let max_results = self.config.max_search_results;

            // Walk the directory tree, searching each text file.
            //
            // `filter_entry` prunes entire subtrees before descending — this is
            // what prevents the tool from grinding through `target/` (tens of
            // thousands of Rust build artefacts) or `node_modules/`. Without
            // this, searching a Rust workspace can hang for 30+ seconds.
            for entry in WalkDir::new(&resolved_dir)
                .follow_links(false)
                .into_iter()
                .filter_entry(|e| !is_excluded_search_dir(e))
                .filter_map(|e| e.ok())
            {
                if matches.len() >= max_results {
                    break;
                }

                let path = entry.path();
                if !path.is_file() {
                    continue;
                }

                // Apply file pattern filter if provided.
                if let Some(fp) = file_pattern {
                    let file_name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or_default();
                    if !simple_glob_match(fp, file_name) {
                        continue;
                    }
                }

                // Skip binary / very large files silently.
                let metadata = match std::fs::metadata(path) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if metadata.len() > self.config.max_read_bytes as u64 {
                    continue;
                }

                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => continue, // likely binary
                };

                // Make the displayed path relative to the search root for readability.
                let display_path = path
                    .strip_prefix(&resolved_dir)
                    .unwrap_or(path)
                    .display()
                    .to_string();
                // Normalise path separators to forward slashes for consistency.
                let display_path = display_path.replace('\\', "/");

                for (line_num, line) in content.lines().enumerate() {
                    if matches.len() >= max_results {
                        break;
                    }
                    if regex.is_match(line) {
                        matches.push(format!(
                            "{}:{}: {}",
                            display_path,
                            line_num + 1,
                            line.trim()
                        ));
                    }
                }
            }

            if matches.is_empty() {
                info!(
                    pattern = %pattern_str,
                    path = %search_path,
                    "No matches found"
                );
                Ok(ToolResult::success(format!(
                    "No matches found for pattern '{}' in '{}'.",
                    pattern_str, search_path
                )))
            } else {
                let truncated = matches.len() >= max_results;
                let mut output = matches.join("\n");
                if truncated {
                    output.push_str(&format!(
                        "\n\n(Results truncated at {} matches. Narrow your search pattern for more specific results.)",
                        max_results
                    ));
                }
                info!(
                    pattern = %pattern_str,
                    path = %search_path,
                    match_count = matches.len(),
                    truncated = truncated,
                    "Search complete"
                );
                Ok(ToolResult::success(output))
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` for directory entries that should be entirely skipped
/// during a file search walk.
///
/// Using this with `WalkDir::filter_entry` prunes the subtree *before*
/// descending, which is what makes the difference between an instant search
/// and a 30-second hang when the workspace contains a Rust `target/` directory
/// or a JavaScript `node_modules/`.
///
/// # What's excluded
/// - Hidden directories (leading `.`): `.git/`, `.cargo/`, `.vscode/`, etc.
/// - Rust build output: `target/`
/// - JS/TS package trees: `node_modules/`
/// - Python bytecode: `__pycache__/`
/// - Common frontend build outputs: `dist/`, `build/`, `.next/`, `.nuxt/`
fn is_excluded_search_dir(entry: &DirEntry) -> bool {
    // depth == 0 is the root of the walk — always include it regardless of name.
    // filter_entry is called on the root too, and on Windows TempDir creates
    // names like `.tmpXXXXXX` which would otherwise match our dot check.
    if entry.depth() == 0 || !entry.file_type().is_dir() {
        return false;
    }
    let name = entry.file_name().to_str().unwrap_or("");
    name.starts_with('.')
        || matches!(
            name,
            "target" | "node_modules" | "__pycache__" | "dist" | "build" | ".next" | ".nuxt"
        )
}

/// Very simple glob matching: supports `*` as a wildcard prefix/suffix.
///
/// This covers the common cases like `*.rs`, `Cargo.*`, `*.toml` without
/// pulling in a full glob crate. For Phase 1 tools this is sufficient.
///
/// Examples:
/// - `*.rs` matches `main.rs`, `lib.rs`
/// - `Cargo.*` matches `Cargo.toml`, `Cargo.lock`
/// - `test_*` matches `test_foo.rs`
/// - `exact.txt` matches only `exact.txt`
fn simple_glob_match(pattern: &str, name: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    // No wildcard — exact match (case-insensitive on Windows for convenience).
    if cfg!(windows) {
        pattern.eq_ignore_ascii_case(name)
    } else {
        pattern == name
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: create a temporary sandbox directory with some test files.
    fn setup_sandbox() -> TempDir {
        let dir = TempDir::new().unwrap();
        let base = dir.path();

        std::fs::write(base.join("hello.txt"), "Hello, world!\nSecond line.\n").unwrap();
        std::fs::write(
            base.join("code.rs"),
            "fn main() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();
        std::fs::create_dir_all(base.join("sub")).unwrap();
        std::fs::write(base.join("sub/nested.txt"), "Nested content here.\n").unwrap();

        dir
    }

    fn test_config(sandbox: &Path) -> FileToolsConfig {
        FileToolsConfig {
            max_read_bytes: 1_048_576,
            max_search_results: 50,
            sandbox_path: sandbox.to_path_buf(),
        }
    }

    // -----------------------------------------------------------------------
    // Path validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_path_relative() {
        let dir = setup_sandbox();
        let result = validate_path("hello.txt", dir.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_path_traversal_blocked() {
        let dir = setup_sandbox();
        let result = validate_path("../../../etc/passwd", dir.path());
        // Should either fail because it escapes sandbox, or because the path
        // doesn't exist. Either way, it's an Err(ToolResult).
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_empty() {
        let dir = setup_sandbox();
        let result = validate_path("", dir.path());
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // FileReadTool
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_file_read_success() {
        let dir = setup_sandbox();
        let tool = FileReadTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({ "path": "hello.txt" }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Hello, world!"));
    }

    #[tokio::test]
    async fn test_file_read_not_found() {
        let dir = setup_sandbox();
        let tool = FileReadTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({ "path": "nonexistent.txt" }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn test_file_read_too_large() {
        let dir = setup_sandbox();
        let mut config = test_config(dir.path());
        config.max_read_bytes = 5; // tiny limit
        let tool = FileReadTool::new(config);

        let result = tool
            .execute(serde_json::json!({ "path": "hello.txt" }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("exceeds"));
    }

    // -----------------------------------------------------------------------
    // FileWriteTool
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_file_write_new_file() {
        let dir = setup_sandbox();
        let tool = FileWriteTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "path": "new_file.txt",
                "content": "brand new content"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Successfully wrote"));

        let written = std::fs::read_to_string(dir.path().join("new_file.txt")).unwrap();
        assert_eq!(written, "brand new content");
    }

    #[tokio::test]
    async fn test_file_write_creates_parent_dirs() {
        let dir = setup_sandbox();
        let tool = FileWriteTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "path": "deep/nested/dir/file.txt",
                "content": "deep content"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        let written = std::fs::read_to_string(dir.path().join("deep/nested/dir/file.txt")).unwrap();
        assert_eq!(written, "deep content");
    }

    #[tokio::test]
    async fn test_file_write_overwrite_existing() {
        let dir = setup_sandbox();
        let tool = FileWriteTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "path": "hello.txt",
                "content": "overwritten!"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        let written = std::fs::read_to_string(dir.path().join("hello.txt")).unwrap();
        assert_eq!(written, "overwritten!");
    }

    // -----------------------------------------------------------------------
    // FileSearchTool
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_file_search_finds_matches() {
        let dir = setup_sandbox();
        let tool = FileSearchTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "pattern": "Hello",
                "path": "."
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("hello.txt"));
        assert!(result.content.contains("Hello, world!"));
    }

    #[tokio::test]
    async fn test_file_search_no_matches() {
        let dir = setup_sandbox();
        let tool = FileSearchTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "pattern": "ZZZZNOTFOUND",
                "path": "."
            }))
            .await
            .unwrap();

        assert!(!result.is_error); // no matches is not an error
        assert!(result.content.contains("No matches"));
    }

    #[tokio::test]
    async fn test_file_search_with_file_pattern() {
        let dir = setup_sandbox();
        let tool = FileSearchTool::new(test_config(dir.path()));

        // Search only .rs files — should find "main" in code.rs but not in .txt files
        let result = tool
            .execute(serde_json::json!({
                "pattern": "fn",
                "path": ".",
                "file_pattern": "*.rs"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("code.rs"));
    }

    #[tokio::test]
    async fn test_file_search_invalid_regex() {
        let dir = setup_sandbox();
        let tool = FileSearchTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "pattern": "[invalid regex",
                "path": "."
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("Invalid regex"));
    }

    #[tokio::test]
    async fn test_file_search_truncation() {
        let dir = setup_sandbox();
        let mut config = test_config(dir.path());
        config.max_search_results = 2; // tiny limit
        let tool = FileSearchTool::new(config);

        // Pattern that matches many lines
        let result = tool
            .execute(serde_json::json!({
                "pattern": ".",
                "path": "."
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("truncated"));
    }

    // -----------------------------------------------------------------------
    // simple_glob_match
    // -----------------------------------------------------------------------

    #[test]
    fn test_glob_wildcard_suffix() {
        assert!(simple_glob_match("*.rs", "main.rs"));
        assert!(simple_glob_match("*.rs", "lib.rs"));
        assert!(!simple_glob_match("*.rs", "main.txt"));
    }

    #[test]
    fn test_glob_wildcard_prefix() {
        assert!(simple_glob_match("Cargo.*", "Cargo.toml"));
        assert!(simple_glob_match("Cargo.*", "Cargo.lock"));
        assert!(!simple_glob_match("Cargo.*", "README.md"));
    }

    #[test]
    fn test_glob_star_matches_all() {
        assert!(simple_glob_match("*", "anything.txt"));
    }

    #[test]
    fn test_glob_exact_match() {
        assert!(simple_glob_match("Makefile", "Makefile"));
        assert!(!simple_glob_match("Makefile", "makefile2"));
    }

    // -----------------------------------------------------------------------
    // is_excluded_search_dir
    // -----------------------------------------------------------------------

    /// Verify that file_search skips `target/` and doesn't return results from it.
    #[tokio::test]
    async fn test_file_search_skips_target_dir() {
        let dir = TempDir::new().unwrap();

        // A file that should be found.
        std::fs::write(dir.path().join("src.rs"), "pub fn hello() {}").unwrap();

        // A file inside `target/` that should NOT be found.
        let target_dir = dir.path().join("target").join("debug");
        std::fs::create_dir_all(&target_dir).unwrap();
        std::fs::write(target_dir.join("artifact.rs"), "pub fn hello() {}").unwrap();

        let tool = FileSearchTool::new(test_config(dir.path()));
        let result = tool
            .execute(serde_json::json!({ "pattern": "hello", "path": "." }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(
            result.content.contains("src.rs"),
            "Expected src.rs in results"
        );
        assert!(
            !result.content.contains("artifact.rs"),
            "target/ contents should be excluded from search results"
        );
    }

    /// Verify that file_search skips hidden directories like `.git/`.
    #[tokio::test]
    async fn test_file_search_skips_hidden_dirs() {
        let dir = TempDir::new().unwrap();

        std::fs::write(dir.path().join("readme.txt"), "findme").unwrap();

        let git_dir = dir.path().join(".git");
        std::fs::create_dir_all(&git_dir).unwrap();
        std::fs::write(git_dir.join("config"), "findme").unwrap();

        let tool = FileSearchTool::new(test_config(dir.path()));
        let result = tool
            .execute(serde_json::json!({ "pattern": "findme", "path": "." }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("readme.txt"));
        assert!(
            !result.content.contains(".git"),
            ".git/ contents should be excluded from search results"
        );
    }
}
