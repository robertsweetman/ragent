//! ProjectContext — workspace introspection for richer system prompts.
//!
//! Reads the workspace directory tree and key project files, formats them as
//! Markdown, and returns a string that can be appended to a skill's system
//! prompt via [`Skill::build_system_prompt`][crate::agent::skill::Skill::build_system_prompt].
//!
//! # Why this exists
//!
//! Without project context, the LLM starts every session "blind" — it must
//! call `file_search` repeatedly just to understand the layout.  Injecting a
//! concise directory tree + key files (README, Cargo.toml, src/main.rs …)
//! at the very start of the context window gives it an immediate mental model
//! of the workspace, typically saving 2–4 tool round-trips per session.
//!
//! # Usage
//! ```ignore
//! let ctx    = ProjectContext::load(workspace_root, 4, 8192)?;
//! let prompt = skill.build_system_prompt(Some(&ctx.format_for_prompt()));
//! ```
//!
//! # Phase reference
//! Introduced in Phase 3 of PLAN.md: "Skill & ProjectContext wiring".

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};
use walkdir::WalkDir;

// ---------------------------------------------------------------------------
// ProjectType
// ---------------------------------------------------------------------------

/// The kind of project detected in the workspace root.
///
/// Detection is intentionally simple (presence of key files) — good enough
/// for selecting which additional files to load without any heavyweight
/// package-manager integration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectType {
    /// A Rust project (`Cargo.toml` present).
    Rust {
        /// `true` when `Cargo.toml` contains a `[workspace]` table.
        ///
        /// Workspace roots typically don't have `src/` themselves — the crates
        /// live in subdirectories — so knowing this avoids attempting to load
        /// non-existent `src/main.rs`.
        is_workspace: bool,
    },
    /// A Node.js / JavaScript / TypeScript project (`package.json` present).
    Node,
    /// A Python project (`pyproject.toml` or `setup.py` present).
    Python,
    /// Could not determine the project type from the files present.
    Unknown,
}

impl ProjectType {
    /// Returns a short, human-readable label for use in the prompt header.
    pub fn display_name(&self) -> &str {
        match self {
            ProjectType::Rust { is_workspace: true } => "Rust (workspace)",
            ProjectType::Rust {
                is_workspace: false,
            } => "Rust",
            ProjectType::Node => "Node.js",
            ProjectType::Python => "Python",
            ProjectType::Unknown => "Unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// ProjectContext
// ---------------------------------------------------------------------------

/// A snapshot of the workspace structure, ready to be embedded in a system prompt.
///
/// Created by [`ProjectContext::load`].  All the expensive I/O (directory
/// walking, file reading) happens once at startup; [`format_for_prompt`] is
/// then a cheap string-formatting operation.
///
/// [`format_for_prompt`]: ProjectContext::format_for_prompt
#[derive(Debug, Clone)]
pub struct ProjectContext {
    /// Canonicalised absolute path to the workspace root.
    pub root: PathBuf,

    /// Detected project type.
    pub project_type: ProjectType,

    /// ASCII-art directory tree (depth-limited, skips hidden dirs and build artefacts).
    pub file_tree: String,

    /// Key project files as `(display_path, content)` pairs, in load order.
    ///
    /// The display path is relative to `root` (e.g. `"src/main.rs"`).
    /// Content may be truncated — see [`ProjectContext::load`] for details.
    pub key_files: Vec<(String, String)>,
}

impl ProjectContext {
    /// Load a [`ProjectContext`] from the given workspace root.
    ///
    /// # Arguments
    /// * `root` — Path to the workspace root directory.
    /// * `max_tree_depth` — How many directory levels deep to walk.
    ///   4 is a good default; 2 is appropriate for very large monorepos.
    /// * `max_key_file_bytes` — Maximum bytes to include *per* key file.
    ///   8192 (8 KiB) is a reasonable default that keeps context affordable.
    ///
    /// # Errors
    /// Returns an error if `root` cannot be canonicalised (does not exist or
    /// the process lacks read permission).
    pub fn load(root: &Path, max_tree_depth: usize, max_key_file_bytes: usize) -> Result<Self> {
        // `canonicalize` resolves the path to an absolute form and verifies
        // the directory exists.  We store the canonical form so that the
        // prompt always shows a full, unambiguous path.
        let root = root
            .canonicalize()
            .with_context(|| format!("Cannot canonicalise workspace root: {}", root.display()))?;

        info!(
            root = %root.display(),
            max_tree_depth,
            max_key_file_bytes,
            "Loading project context"
        );

        let project_type = detect_project_type(&root);
        debug!(project_type = ?project_type, "Detected project type");

        let file_tree = build_file_tree(&root, max_tree_depth);
        let key_files = load_key_files(&root, &project_type, max_key_file_bytes);

        info!(
            key_files = key_files.len(),
            project_type = project_type.display_name(),
            "Project context loaded"
        );

        Ok(Self {
            root,
            project_type,
            file_tree,
            key_files,
        })
    }

    /// Format the context as a Markdown string for appending to a system prompt.
    ///
    /// The returned string has the following top-level sections:
    ///
    /// - **`## Project Context`** — root path and detected project type.
    /// - **`### File Tree`** — indented ASCII directory tree in a fenced code block.
    /// - **`### Key Files`** — each loaded file in its own language-tagged fenced
    ///   code block (e.g. ` ```rust `, ` ```toml `).
    ///
    /// This string is designed to be passed to
    /// [`Skill::build_system_prompt`][crate::agent::skill::Skill::build_system_prompt]
    /// as the `project_context` argument.
    pub fn format_for_prompt(&self) -> String {
        let mut out = String::with_capacity(4096);

        out.push_str("## Project Context\n\n");
        out.push_str(&format!("**Root**: {}  \n", self.root.display()));
        out.push_str(&format!("**Type**: {}\n", self.project_type.display_name()));

        // File tree — plain code block (no language tag needed).
        out.push_str("\n### File Tree\n```\n");
        out.push_str(&self.file_tree);
        out.push_str("\n```\n");

        if !self.key_files.is_empty() {
            out.push_str("\n### Key Files\n");
            for (path, content) in &self.key_files {
                let lang = lang_tag_for_path(path);
                // Format: bold heading, then fenced code block.
                out.push_str(&format!(
                    "\n**{path}**\n```{lang}\n{content}\n```\n",
                    path = path,
                    lang = lang,
                    content = content,
                ));
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Project type detection
// ---------------------------------------------------------------------------

/// Inspect the files present in `root` to infer the project type.
///
/// Priority: Rust > Node > Python > Unknown.
/// First match wins — if someone has both `Cargo.toml` and `package.json`
/// (uncommon but possible), we call it a Rust project.
fn detect_project_type(root: &Path) -> ProjectType {
    let cargo_toml = root.join("Cargo.toml");
    if cargo_toml.exists() {
        // Read the file to check for a `[workspace]` table.
        // `map(|s| s.contains(...))` keeps the logic on one line and
        // gracefully falls back to `false` on read errors.
        let is_workspace = std::fs::read_to_string(&cargo_toml)
            .map(|s| s.contains("[workspace]"))
            .unwrap_or(false);
        return ProjectType::Rust { is_workspace };
    }

    if root.join("package.json").exists() {
        return ProjectType::Node;
    }

    // Python projects may use either the modern `pyproject.toml` or the
    // legacy `setup.py` — check both.
    if root.join("pyproject.toml").exists() || root.join("setup.py").exists() {
        return ProjectType::Python;
    }

    ProjectType::Unknown
}

// ---------------------------------------------------------------------------
// Directory tree builder
// ---------------------------------------------------------------------------

/// Directory names to skip entirely during tree traversal.
///
/// These are build artefact directories and VCS metadata that add noise
/// without helping the LLM understand the project structure.
///
/// `walkdir`'s `filter_entry` prunes the *entire subtree* when we return
/// `false` for a directory — so skipping "target" also skips all its
/// (potentially huge) contents.
const SKIP_DIRS: &[&str] = &["target", "node_modules", "__pycache__", ".git"];

/// Maximum number of entries to display in a single directory.
///
/// Directories with more entries get a `... (N more)` summary line.  This
/// prevents enormous `src/` trees from flooding the context window.
const MAX_ENTRIES_PER_DIR: usize = 20;

/// Build an ASCII directory tree from `root` up to `max_depth` levels deep.
///
/// Uses [`walkdir`] for safe, iterative traversal (no manual recursion stack),
/// then groups entries by parent directory to compute the `├──` / `└──`
/// connectors correctly.
///
/// # Why two passes?
/// `walkdir` visits entries in DFS pre-order.  To know whether an entry is the
/// *last* sibling in its directory (and thus gets `└──`), we need to see all
/// siblings first.  A two-pass approach — collect → group → format — is the
/// cleanest way to achieve this without fighting the iterator API.
fn build_file_tree(root: &Path, max_depth: usize) -> String {
    // Pass 1: walk the directory, collecting (name, is_dir) per parent path.
    //
    // `BTreeMap` keeps parent paths in sorted order, which means when we
    // later call `format_subtree` we process directories in a consistent,
    // reproducible sequence.
    let mut dir_children: BTreeMap<PathBuf, Vec<(String, bool)>> = BTreeMap::new();

    let walker = WalkDir::new(root)
        .max_depth(max_depth)
        // `filter_entry` prunes whole subtrees for filtered-out directories,
        // which is far more efficient than filtering individual entries.
        .into_iter()
        .filter_entry(|e| {
            // Always descend into the root itself (depth 0).
            if e.depth() == 0 {
                return true;
            }
            let name = e.file_name().to_string_lossy();
            if e.file_type().is_dir() {
                // Skip hidden directories (names starting with '.') and known
                // build/VCS directories.
                if name.starts_with('.') {
                    return false;
                }
                if SKIP_DIRS.iter().any(|&skip| skip == name.as_ref()) {
                    return false;
                }
            }
            true
        });

    for result in walker {
        let entry = match result {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, "Error reading directory entry during tree walk");
                continue;
            }
        };

        // Skip the root itself — we only list its *children*.
        if entry.depth() == 0 {
            continue;
        }

        let parent = match entry.path().parent() {
            Some(p) => p.to_path_buf(),
            None => continue, // shouldn't happen given depth > 0
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_dir = entry.file_type().is_dir();

        dir_children.entry(parent).or_default().push((name, is_dir));
    }

    // Sort within each directory: directories first, then files, both
    // alphabetically.  This matches the output of the classic `tree` command.
    for entries in dir_children.values_mut() {
        entries.sort_by(|a, b| {
            // `b.1.cmp(&a.1)` puts `true` (is_dir) before `false` (is_file).
            b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        });
    }

    // Pass 2: format the collected entries as an indented tree.
    let mut lines: Vec<String> = Vec::new();
    format_subtree(root, &dir_children, "", &mut lines);
    lines.join("\n")
}

/// Recursively format one directory's children into `lines`.
///
/// `prefix` is the left-margin string accumulated from parent calls
/// (e.g. `"│   │   "`).  Each call appends the `├──` / `└──` connector
/// for each child and, for directory children, recurses.
fn format_subtree(
    dir: &Path,
    dir_children: &BTreeMap<PathBuf, Vec<(String, bool)>>,
    prefix: &str,
    lines: &mut Vec<String>,
) {
    let entries = match dir_children.get(dir) {
        Some(e) => e,
        None => return, // empty directory or leaf node — nothing to print
    };

    let total = entries.len();
    let shown = total.min(MAX_ENTRIES_PER_DIR);
    // `has_overflow` is true when there are entries we won't display.
    // In that case, all `shown` entries keep "├──" (more lines follow),
    // and we append one final "└── ... (N more)" summary.
    let has_overflow = total > MAX_ENTRIES_PER_DIR;

    for (i, (name, is_dir)) in entries.iter().take(shown).enumerate() {
        // An entry is "last" (gets └──) only when no more lines follow it:
        // no more siblings AND no overflow summary line.
        let is_last = i == shown - 1 && !has_overflow;
        let connector = if is_last { "└── " } else { "├── " };
        // The child prefix carries the visual "column" for nested entries.
        // "    " (four spaces) under └── so the column appears empty;
        // "│   " under ├── to continue the vertical line.
        let child_prefix = if is_last { "    " } else { "│   " };

        // Directories get a trailing "/" to visually distinguish them.
        let display_name = if *is_dir {
            format!("{}/", name)
        } else {
            name.clone()
        };

        lines.push(format!("{}{}{}", prefix, connector, display_name));

        if *is_dir {
            let child_dir = dir.join(name);
            format_subtree(
                &child_dir,
                dir_children,
                &format!("{}{}", prefix, child_prefix),
                lines,
            );
        }
    }

    if has_overflow {
        let extra = total - shown;
        lines.push(format!("{}└── ... ({} more)", prefix, extra));
    }
}

// ---------------------------------------------------------------------------
// Key file loader
// ---------------------------------------------------------------------------

/// Load important project files from disk and return them as
/// `(display_path, content)` pairs.
///
/// Always tries: `README.md`, `README.rst`, `.ragent.toml`, `PLAN.md`.
/// Then loads project-type-specific files (e.g. `Cargo.toml` for Rust).
/// Files that don't exist are silently skipped.
///
/// `max_bytes` controls the per-file truncation limit.  `Cargo.lock` is
/// special-cased to a hard maximum of 2 000 bytes because it's often
/// enormous and its lower sections are rarely useful to the LLM.
fn load_key_files(
    root: &Path,
    project_type: &ProjectType,
    max_bytes: usize,
) -> Vec<(String, String)> {
    // Ordered list of relative paths to try.
    // `&str` slices are cheap — we only allocate Strings for files that exist.
    let mut candidates: Vec<&str> = vec!["README.md", "README.rst", ".ragent.toml", "PLAN.md"];

    match project_type {
        ProjectType::Rust { .. } => {
            // `Cargo.lock` is included but capped separately (see below).
            candidates.extend_from_slice(&[
                "Cargo.toml",
                "Cargo.lock",
                "src/main.rs",
                "src/lib.rs",
            ]);
        }
        ProjectType::Node => {
            candidates.extend_from_slice(&["package.json", "tsconfig.json"]);
        }
        ProjectType::Python => {
            candidates.extend_from_slice(&["pyproject.toml", "setup.py", "requirements.txt"]);
        }
        ProjectType::Unknown => {}
    }

    let mut key_files: Vec<(String, String)> = Vec::new();

    for relative_path in candidates {
        let abs = root.join(relative_path);
        if !abs.exists() {
            continue;
        }

        // Cargo.lock is special — it can be tens of thousands of lines long.
        // We only need the preamble (the `[package]` / workspace metadata)
        // so we cap it tightly, independently of the caller's `max_bytes`.
        let limit = if relative_path == "Cargo.lock" {
            2000.min(max_bytes)
        } else {
            max_bytes
        };

        match read_truncated(&abs, limit) {
            Ok(content) => {
                debug!(path = %relative_path, bytes = content.len(), "Loaded key file");
                key_files.push((relative_path.to_string(), content));
            }
            Err(e) => {
                warn!(path = %relative_path, error = %e, "Failed to read key file — skipping");
            }
        }
    }

    key_files
}

/// Read at most `max_bytes` bytes from a file, appending a truncation notice
/// if the file was larger.
///
/// Uses `String::from_utf8_lossy` so that binary or invalid-UTF-8 files don't
/// cause a hard error — they're shown with replacement characters instead.
fn read_truncated(path: &Path, max_bytes: usize) -> Result<String> {
    let raw =
        std::fs::read(path).with_context(|| format!("Reading key file: {}", path.display()))?;

    if raw.len() <= max_bytes {
        // Common case: file fits within the limit.
        Ok(String::from_utf8_lossy(&raw).into_owned())
    } else {
        // Truncate at a byte boundary.  `from_utf8_lossy` handles the case
        // where the cut falls in the middle of a multi-byte character by
        // replacing the malformed sequence with U+FFFD.
        let truncated = String::from_utf8_lossy(&raw[..max_bytes]).into_owned();
        Ok(format!(
            "{}\n[... truncated at {} bytes]",
            truncated, max_bytes
        ))
    }
}

// ---------------------------------------------------------------------------
// Language tag helper
// ---------------------------------------------------------------------------

/// Infer the Markdown fenced-code-block language tag from a file path.
///
/// Returns a `'static str` (a string literal) so the caller doesn't need to
/// allocate.  Returns `""` (empty, no tag) for unrecognised extensions.
fn lang_tag_for_path(path: &str) -> &'static str {
    // Extract the extension using `Path::extension` so we handle dotfiles
    // and paths with multiple dots correctly.
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "rs" => "rust",
        "toml" => "toml",
        "json" => "json",
        "md" => "md",
        "py" => "python",
        "ts" => "typescript",
        "js" => "javascript",
        "rst" => "rst",
        "txt" => "text",
        _ => "",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // Helper: create a fresh temp directory for each test.
    fn make_temp() -> TempDir {
        tempfile::tempdir().expect("failed to create temp dir")
    }

    // ---------------------------------------------------------------------------
    // ProjectType detection
    // ---------------------------------------------------------------------------

    #[test]
    fn test_detect_rust_project() {
        let dir = make_temp();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"foo\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();

        let pt = detect_project_type(dir.path());
        assert!(
            matches!(
                pt,
                ProjectType::Rust {
                    is_workspace: false
                }
            ),
            "expected Rust (non-workspace), got {:?}",
            pt
        );
    }

    #[test]
    fn test_detect_rust_workspace() {
        let dir = make_temp();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"crate_a\", \"crate_b\"]\n",
        )
        .unwrap();

        let pt = detect_project_type(dir.path());
        assert!(
            matches!(pt, ProjectType::Rust { is_workspace: true }),
            "expected Rust (workspace), got {:?}",
            pt
        );
    }

    #[test]
    fn test_detect_node_project() {
        let dir = make_temp();
        fs::write(dir.path().join("package.json"), "{}").unwrap();

        let pt = detect_project_type(dir.path());
        assert_eq!(pt, ProjectType::Node);
    }

    #[test]
    fn test_detect_unknown_project() {
        // Empty directory → Unknown.
        let dir = make_temp();
        let pt = detect_project_type(dir.path());
        assert_eq!(pt, ProjectType::Unknown);
    }

    // ---------------------------------------------------------------------------
    // File tree
    // ---------------------------------------------------------------------------

    #[test]
    fn test_file_tree_basic() {
        let dir = make_temp();
        fs::write(dir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();
        fs::write(dir.path().join("subdir").join("lib.rs"), "").unwrap();

        let tree = build_file_tree(dir.path(), 4);

        assert!(
            tree.contains("main.rs"),
            "tree should contain main.rs — got:\n{}",
            tree
        );
        assert!(
            tree.contains("subdir"),
            "tree should contain subdir — got:\n{}",
            tree
        );
        assert!(
            tree.contains("lib.rs"),
            "tree should contain nested lib.rs — got:\n{}",
            tree
        );
    }

    #[test]
    fn test_file_tree_skips_hidden() {
        let dir = make_temp();
        fs::create_dir(dir.path().join(".hidden")).unwrap();
        fs::write(dir.path().join(".hidden").join("secret.txt"), "shh").unwrap();
        fs::write(dir.path().join("visible.txt"), "hello").unwrap();

        let tree = build_file_tree(dir.path(), 4);

        assert!(
            !tree.contains(".hidden"),
            "hidden directory should not appear in tree — got:\n{}",
            tree
        );
        assert!(
            tree.contains("visible.txt"),
            "visible file should appear in tree — got:\n{}",
            tree
        );
        // secret.txt is inside the hidden dir, so it must also be absent.
        assert!(
            !tree.contains("secret.txt"),
            "contents of hidden dir should not appear — got:\n{}",
            tree
        );
    }

    #[test]
    fn test_file_tree_skips_target() {
        let dir = make_temp();
        fs::create_dir(dir.path().join("target")).unwrap();
        fs::write(dir.path().join("target").join("build_output"), "...").unwrap();
        fs::write(dir.path().join("src.rs"), "").unwrap();

        let tree = build_file_tree(dir.path(), 4);

        assert!(
            !tree.contains("target"),
            "target/ should not appear in tree — got:\n{}",
            tree
        );
        assert!(
            tree.contains("src.rs"),
            "src.rs should appear in tree — got:\n{}",
            tree
        );
    }

    #[test]
    fn test_file_tree_dirs_before_files() {
        // Directories should be listed before files at the same level.
        let dir = make_temp();
        fs::write(dir.path().join("zzz_file.txt"), "").unwrap();
        fs::create_dir(dir.path().join("aaa_dir")).unwrap();

        let tree = build_file_tree(dir.path(), 4);

        let dir_pos = tree.find("aaa_dir").unwrap();
        let file_pos = tree.find("zzz_file.txt").unwrap();
        assert!(
            dir_pos < file_pos,
            "directory (aaa_dir) should appear before file (zzz_file.txt) — got:\n{}",
            tree
        );
    }

    #[test]
    fn test_file_tree_overflow_summary() {
        // Create more than MAX_ENTRIES_PER_DIR files in one directory.
        let dir = make_temp();
        for i in 0..=MAX_ENTRIES_PER_DIR {
            fs::write(dir.path().join(format!("file_{:02}.txt", i)), "").unwrap();
        }

        let tree = build_file_tree(dir.path(), 4);

        assert!(
            tree.contains("more)"),
            "tree should contain an overflow summary — got:\n{}",
            tree
        );
    }

    // ---------------------------------------------------------------------------
    // Key files
    // ---------------------------------------------------------------------------

    #[test]
    fn test_key_files_loaded() {
        let dir = make_temp();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"test\"\n",
        )
        .unwrap();
        fs::write(dir.path().join("README.md"), "# Test Project\n").unwrap();

        let files = load_key_files(
            dir.path(),
            &ProjectType::Rust {
                is_workspace: false,
            },
            8192,
        );
        let paths: Vec<&str> = files.iter().map(|(p, _)| p.as_str()).collect();

        assert!(
            paths.contains(&"Cargo.toml"),
            "Cargo.toml should be in key_files — got: {:?}",
            paths
        );
        assert!(
            paths.contains(&"README.md"),
            "README.md should be in key_files — got: {:?}",
            paths
        );
    }

    #[test]
    fn test_key_file_truncation() {
        let dir = make_temp();
        // Create a file clearly larger than our limit.
        let big_content = "a".repeat(500);
        fs::write(dir.path().join("README.md"), &big_content).unwrap();

        let files = load_key_files(dir.path(), &ProjectType::Unknown, 100);
        let readme = files
            .iter()
            .find(|(p, _)| p == "README.md")
            .expect("README.md should be in key_files");

        assert!(
            readme.1.contains("[... truncated at 100 bytes]"),
            "truncation notice missing — content:\n{}",
            readme.1
        );
        // The content before the newline should be exactly 100 'a' characters.
        let first_line = readme.1.lines().next().unwrap();
        assert_eq!(
            first_line.len(),
            100,
            "first line should be exactly 100 bytes of content"
        );
    }

    #[test]
    fn test_cargo_lock_capped_at_2000_bytes() {
        let dir = make_temp();
        // Simulate a large Cargo.lock (common in real projects).
        let lock_content = "x".repeat(10_000);
        fs::write(dir.path().join("Cargo.lock"), &lock_content).unwrap();
        // Set max_bytes high so only the Cargo.lock special-case cap applies.
        let files = load_key_files(
            dir.path(),
            &ProjectType::Rust {
                is_workspace: false,
            },
            8192,
        );
        let lock = files
            .iter()
            .find(|(p, _)| p == "Cargo.lock")
            .expect("Cargo.lock should be present");

        assert!(
            lock.1.contains("[... truncated at 2000 bytes]"),
            "Cargo.lock should be capped at 2000 bytes — content starts with: {}",
            &lock.1[..50.min(lock.1.len())]
        );
    }

    // ---------------------------------------------------------------------------
    // format_for_prompt
    // ---------------------------------------------------------------------------

    #[test]
    fn test_format_for_prompt_contains_sections() {
        let dir = make_temp();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"test\"\n",
        )
        .unwrap();

        let ctx = ProjectContext::load(dir.path(), 4, 8192).unwrap();
        let output = ctx.format_for_prompt();

        assert!(
            output.contains("Project Context"),
            "output should contain 'Project Context' header — got:\n{}",
            output
        );
        assert!(
            output.contains("File Tree"),
            "output should contain 'File Tree' section — got:\n{}",
            output
        );
        assert!(
            output.contains("Key Files"),
            "output should contain 'Key Files' section — got:\n{}",
            output
        );
        assert!(
            output.contains("Rust"),
            "output should mention the project type — got:\n{}",
            output
        );
    }

    #[test]
    fn test_format_for_prompt_lang_tags() {
        // Verify that language tags are applied to recognised extensions.
        assert_eq!(lang_tag_for_path("src/main.rs"), "rust");
        assert_eq!(lang_tag_for_path("Cargo.toml"), "toml");
        assert_eq!(lang_tag_for_path("package.json"), "json");
        assert_eq!(lang_tag_for_path("README.md"), "md");
        assert_eq!(lang_tag_for_path("script.py"), "python");
        assert_eq!(lang_tag_for_path("app.ts"), "typescript");
        assert_eq!(lang_tag_for_path("index.js"), "javascript");
        assert_eq!(lang_tag_for_path("binary"), "");
    }

    #[test]
    fn test_project_type_display_name() {
        assert_eq!(
            ProjectType::Rust { is_workspace: true }.display_name(),
            "Rust (workspace)"
        );
        assert_eq!(
            ProjectType::Rust {
                is_workspace: false
            }
            .display_name(),
            "Rust"
        );
        assert_eq!(ProjectType::Node.display_name(), "Node.js");
        assert_eq!(ProjectType::Python.display_name(), "Python");
        assert_eq!(ProjectType::Unknown.display_name(), "Unknown");
    }
}
