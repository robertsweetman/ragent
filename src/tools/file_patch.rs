//! Surgical find-and-replace editing of existing files via `FilePatchTool`.
//!
//! # Phase 2 (PLAN.md)
//! This tool complements `FileWriteTool`. Where `FileWriteTool` is ideal for
//! creating new files or completely replacing a file's contents, `FilePatchTool`
//! is safer and more precise for editing **existing** files: the LLM provides
//! only the exact text to change, leaving the rest of the file untouched.
//!
//! # Design: Why "exact match once"?
//!
//! Requiring `old_text` to appear **exactly once** is a deliberate safety
//! constraint. If the LLM provides ambiguous context that matches multiple
//! locations, blindly patching the wrong site could silently corrupt a file.
//! By returning an error when the count is 0 or ≥ 2, we force the LLM to
//! provide more specific context, leading to correct and auditable edits.
//!
//! # Security
//!
//! Like all file tools, `FilePatchTool` validates paths against the sandbox
//! via the shared [`crate::tools::file_ops::validate_path`] helper. The file
//! must already exist before patching — this tool does not create new files.

use std::future::Future;
use std::pin::Pin;

use anyhow::Result;
use serde_json::Value;
use tracing::{debug, info, instrument};

use crate::tools::file_ops::{FileToolsConfig, validate_path};
use crate::tools::traits::{Tool, ToolResult};

// ===========================================================================
// FilePatchTool
// ===========================================================================

/// Surgically edits an existing file by replacing one exact occurrence of
/// `old_text` with `new_text`.
///
/// The LLM must provide text that matches the file precisely — including all
/// whitespace, indentation, and newlines. The tool deliberately errors when
/// `old_text` appears zero or more-than-once to prevent ambiguous edits.
///
/// # Example (LLM tool call arguments)
///
/// ```json
/// {
///   "path": "src/main.rs",
///   "old_text": "println!(\"hello\");",
///   "new_text": "println!(\"hello, world!\");"
/// }
/// ```
pub struct FilePatchTool {
    config: FileToolsConfig,
}

impl FilePatchTool {
    /// Create a new `FilePatchTool` with the given shared configuration.
    pub fn new(config: FileToolsConfig) -> Self {
        Self { config }
    }
}

impl Tool for FilePatchTool {
    fn name(&self) -> &str {
        "file_patch"
    }

    fn description(&self) -> &str {
        "Surgically edit an existing file by replacing one exact occurrence of `old_text` with \
         `new_text`. The match must be exact — including whitespace and newlines. Use this instead \
         of `file_write` when you want to change a small portion of a file without rewriting the \
         whole thing. The tool will error if `old_text` is not found, or if it appears more than \
         once (in which case you must provide more surrounding context to make it unique)."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative to the working directory \
                                    or absolute). The file must already exist."
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find in the file. Must appear exactly once. \
                                    Include enough surrounding context to make it unique if the \
                                    snippet could otherwise match multiple locations."
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text. May be empty to delete `old_text`."
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    /// Execute the patch.
    ///
    /// # Steps
    /// 1. Extract and validate all three parameters.
    /// 2. Resolve and sandbox-check the path; verify the file exists.
    /// 3. Read the file contents.
    /// 4. Count occurrences of `old_text` — must be exactly one.
    /// 5. Replace the single occurrence and write the result back.
    #[allow(clippy::manual_async_fn)]
    #[instrument(skip(self, args), fields(tool = "file_patch"))]
    fn execute(
        &self,
        args: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
        Box::pin(async move {
            // ------------------------------------------------------------------
            // 1. Extract parameters
            // ------------------------------------------------------------------
            let raw_path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) if !p.is_empty() => p,
                _ => {
                    return Ok(ToolResult::error(
                        "Missing or empty parameter 'path'. Provide the path to the file to edit.",
                    ));
                }
            };

            let old_text = match args.get("old_text").and_then(|v| v.as_str()) {
                Some(t) if !t.is_empty() => t,
                _ => {
                    return Ok(ToolResult::error(
                        "Missing or empty parameter 'old_text'. Provide the exact text to replace.",
                    ));
                }
            };

            // `new_text` may legitimately be an empty string (deletion), but the
            // key must be present. We distinguish "key absent" from "empty string"
            // so the LLM gets a precise error when it forgets the parameter.
            let new_text = match args.get("new_text").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => {
                    return Ok(ToolResult::error(
                        "Missing parameter 'new_text'. Provide the replacement text \
                         (use an empty string \"\" to delete 'old_text').",
                    ));
                }
            };

            debug!(path = %raw_path, old_len = old_text.len(), new_len = new_text.len(), "Patching file");

            // ------------------------------------------------------------------
            // 2. Validate path and confirm the file exists
            // ------------------------------------------------------------------
            let resolved = match validate_path(raw_path, &self.config.sandbox_path) {
                Ok(p) => p,
                Err(tool_err) => return Ok(tool_err),
            };

            // `validate_path` succeeds even for non-existent paths (the write tool
            // relies on this). For patching we require the file to exist first.
            if !resolved.exists() {
                return Ok(ToolResult::error(format!(
                    "File not found: '{}'. Use 'file_write' to create new files.",
                    raw_path
                )));
            }

            // ------------------------------------------------------------------
            // 3. Read the file
            // ------------------------------------------------------------------
            let content = match std::fs::read_to_string(&resolved) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(ToolResult::error(format!(
                        "Failed to read '{}': {}",
                        raw_path, e
                    )));
                }
            };

            // ------------------------------------------------------------------
            // 4. Count occurrences — must be exactly 1
            // ------------------------------------------------------------------
            let count = count_occurrences(&content, old_text);

            if count == 0 {
                return Ok(ToolResult::error(
                    "Text not found in file. Ensure old_text matches exactly including \
                     whitespace and newlines. Use 'file_read' to inspect the current content.",
                ));
            }

            if count >= 2 {
                return Ok(ToolResult::error(format!(
                    "Text appears {count} times in the file. Provide more surrounding context \
                     to make old_text unique.",
                )));
            }

            // ------------------------------------------------------------------
            // 5. Patch and write
            // ------------------------------------------------------------------
            // `replacen` with limit=1 is safe here because we know count==1.
            // We capture the lengths before the replacement consumes the strings.
            let old_text_len = old_text.len();
            let new_text_len = new_text.len();

            let patched = content.replacen(old_text, new_text, 1);

            if let Err(e) = std::fs::write(&resolved, &patched) {
                return Ok(ToolResult::error(format!(
                    "Failed to write patched content to '{}': {}",
                    raw_path, e
                )));
            }

            info!(
                path = %raw_path,
                old_chars = old_text_len,
                new_chars = new_text_len,
                "File patched successfully"
            );

            Ok(ToolResult::success(format!(
                "Successfully patched '{}': replaced {old_text_len} chars with {new_text_len} chars.",
                raw_path
            )))
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Count non-overlapping occurrences of `needle` in `haystack`.
///
/// We use this instead of `str::matches().count()` to be explicit about
/// the semantics: standard `str::matches` is already non-overlapping for
/// string patterns, but naming the helper makes the intent clear at the
/// call site and keeps `execute` readable.
fn count_occurrences(haystack: &str, needle: &str) -> usize {
    // `str::matches` returns an iterator of non-overlapping matches, left to
    // right. Counting it is O(n) in the file length — exactly what we want.
    haystack.matches(needle).count()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Create a temporary sandbox and return the dir handle (keeps it alive).
    fn setup_sandbox() -> TempDir {
        TempDir::new().unwrap()
    }

    fn test_config(sandbox: &Path) -> FileToolsConfig {
        FileToolsConfig {
            max_read_bytes: 1_048_576,
            max_search_results: 50,
            sandbox_path: sandbox.to_path_buf(),
        }
    }

    // Write a file inside the sandbox and return its relative name.
    fn write_file(sandbox: &Path, name: &str, content: &str) -> String {
        std::fs::write(sandbox.join(name), content).unwrap();
        name.to_string()
    }

    // Read a file inside the sandbox.
    fn read_file(sandbox: &Path, name: &str) -> String {
        std::fs::read_to_string(sandbox.join(name)).unwrap()
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    #[test]
    fn test_tool_metadata() {
        let dir = setup_sandbox();
        let tool = FilePatchTool::new(test_config(dir.path()));

        assert_eq!(tool.name(), "file_patch");

        let desc = tool.description();
        assert!(!desc.is_empty(), "description should not be empty");
        assert!(
            desc.contains("old_text"),
            "description should mention old_text"
        );

        let schema = tool.parameters_schema();
        let props = &schema["properties"];
        assert!(props["path"].is_object(), "schema should have 'path'");
        assert!(
            props["old_text"].is_object(),
            "schema should have 'old_text'"
        );
        assert!(
            props["new_text"].is_object(),
            "schema should have 'new_text'"
        );

        let required = schema["required"].as_array().unwrap();
        let required_names: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(required_names.contains(&"path"));
        assert!(required_names.contains(&"old_text"));
        assert!(required_names.contains(&"new_text"));
    }

    // -----------------------------------------------------------------------
    // Happy path
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_patch_success() {
        let dir = setup_sandbox();
        write_file(dir.path(), "greet.txt", "Hello, world!\nGoodbye, world!\n");
        let tool = FilePatchTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "path": "greet.txt",
                "old_text": "Hello, world!",
                "new_text": "Hello, Rust!"
            }))
            .await
            .unwrap();

        assert!(
            !result.is_error,
            "expected success, got: {}",
            result.content
        );
        assert!(
            result.content.contains("patched"),
            "message should mention 'patched'"
        );

        let on_disk = read_file(dir.path(), "greet.txt");
        assert_eq!(on_disk, "Hello, Rust!\nGoodbye, world!\n");
    }

    // Replacement with an empty string effectively deletes the matched text.
    #[tokio::test]
    async fn test_patch_delete_text() {
        let dir = setup_sandbox();
        write_file(
            dir.path(),
            "data.txt",
            "keep this\ndelete me\nkeep this too\n",
        );
        let tool = FilePatchTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "path": "data.txt",
                "old_text": "delete me\n",
                "new_text": ""
            }))
            .await
            .unwrap();

        assert!(!result.is_error, "{}", result.content);
        assert_eq!(
            read_file(dir.path(), "data.txt"),
            "keep this\nkeep this too\n"
        );
    }

    // -----------------------------------------------------------------------
    // Error cases
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_patch_not_found() {
        let dir = setup_sandbox();
        write_file(dir.path(), "sample.txt", "actual content here\n");
        let tool = FilePatchTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "path": "sample.txt",
                "old_text": "this text does not exist",
                "new_text": "replacement"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(
            result.content.contains("not found"),
            "error should say 'not found': {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_patch_multiple_occurrences() {
        let dir = setup_sandbox();
        write_file(dir.path(), "repeat.txt", "duplicate line\nduplicate line\n");
        let tool = FilePatchTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "path": "repeat.txt",
                "old_text": "duplicate line",
                "new_text": "unique line"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(
            result.content.contains("2 times") || result.content.contains("times"),
            "error should mention count: {}",
            result.content
        );
        // File must not have been modified.
        assert_eq!(
            read_file(dir.path(), "repeat.txt"),
            "duplicate line\nduplicate line\n"
        );
    }

    #[tokio::test]
    async fn test_patch_missing_params() {
        let dir = setup_sandbox();
        let tool = FilePatchTool::new(test_config(dir.path()));

        // Missing 'path'
        let r = tool
            .execute(serde_json::json!({ "old_text": "x", "new_text": "y" }))
            .await
            .unwrap();
        assert!(r.is_error);
        assert!(r.content.to_lowercase().contains("path"));

        // Missing 'old_text'
        let r = tool
            .execute(serde_json::json!({ "path": "f.txt", "new_text": "y" }))
            .await
            .unwrap();
        assert!(r.is_error);
        assert!(r.content.to_lowercase().contains("old_text"));

        // Missing 'new_text'
        let r = tool
            .execute(serde_json::json!({ "path": "f.txt", "old_text": "x" }))
            .await
            .unwrap();
        assert!(r.is_error);
        assert!(r.content.to_lowercase().contains("new_text"));
    }

    #[tokio::test]
    async fn test_patch_nonexistent_file() {
        let dir = setup_sandbox();
        // Do NOT create the file — the tool must detect its absence.
        let tool = FilePatchTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "path": "ghost.txt",
                "old_text": "anything",
                "new_text": "replacement"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        // The explicit existence check in execute() should fire.
        assert!(
            result.content.contains("not found") || result.content.contains("File not found"),
            "expected 'not found' in error, got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn test_patch_empty_old_text() {
        let dir = setup_sandbox();
        write_file(dir.path(), "file.txt", "some content\n");
        let tool = FilePatchTool::new(test_config(dir.path()));

        let result = tool
            .execute(serde_json::json!({
                "path": "file.txt",
                "old_text": "",
                "new_text": "replacement"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(
            result.content.to_lowercase().contains("old_text"),
            "error should mention old_text: {}",
            result.content
        );
    }
}
