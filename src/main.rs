//! ragent CLI — interactive chat with a local LLM via Ollama.
//!
//! Phase 3 entry point: adds the Skill system (code-writer / chat), project
//! context auto-loading (directory tree + key files injected into the prompt),
//! the `file_patch` tool for surgical edits, and CARGO_TARGET_DIR isolation
//! to prevent Windows MAX_PATH issues in Cargo workspaces.
//!
//! It also handles:
//! - Checking if Ollama is installed and running
//! - Auto-starting Ollama if it's not running
//! - Verifying the requested model is available
//! - Detecting models that don't support tool calling
//! - Isolating the agent's file/shell operations to a configurable workspace
//!
//! # Phase 3 (PLAN.md)
//! Goal: "Agent writes, tests, and iterates on code effectively."

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use tracing::{error, info, warn};

use ragent::agent::agent_loop::AgentLoop;
use ragent::agent::{ProjectContext, Skill};
use ragent::config::AppConfig;
use ragent::llm::ollama::{OllamaBackend, OllamaError};
use ragent::tools::ToolRegistry;
use ragent::tools::file_ops::{FileReadTool, FileSearchTool, FileToolsConfig, FileWriteTool};
use ragent::tools::file_patch::FilePatchTool;
use ragent::tools::shell::ShellExecTool;

/// Models known NOT to support Ollama's tool/function calling format.
/// Used for an early warning at startup before the user wastes time chatting.
const MODELS_WITHOUT_TOOL_SUPPORT: &[&str] = &[
    "deepseek-r1",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "phi",
    "phi3",
    "tinyllama",
];

/// ragent — a local-only agentic AI framework.
///
/// Chat with a local LLM through Ollama with full request/response logging.
/// The agent can use tools (file read/write/patch/search, shell execution) to
/// accomplish multi-step tasks autonomously.
///
/// # Workspace isolation
///
/// By default ragent operates in the current directory. This means the agent
/// can read and write files here — including ragent's own source code if you
/// run it from the project root. Use `--workspace` to point the agent at a
/// separate directory for its work.
#[derive(Parser, Debug)]
#[command(name = "ragent", version, about)]
struct Cli {
    /// Model name to use (overrides config/default.toml).
    #[arg(short, long)]
    model: Option<String>,

    /// Ollama server URL (overrides config/default.toml).
    #[arg(long, value_name = "URL")]
    ollama_url: Option<String>,

    /// Log level: error, warn, info, debug, trace.
    /// Use 'debug' to see full HTTP request/response payloads and tool calls.
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Skill to use. Built-in options: "code" (code-writer, default) or "chat" (general assistant).
    ///
    /// The "code" skill uses a detailed system prompt that guides the agent
    /// through a read → patch → test → verify workflow for software development.
    /// The "chat" skill uses a simpler prompt suitable for general conversation.
    #[arg(long, default_value = "code")]
    skill: Option<String>,

    /// Workspace directory for the agent's file and shell operations.
    ///
    /// All file reads/writes and shell commands will be sandboxed to this
    /// directory. Defaults to the current directory if not specified.
    ///
    /// IMPORTANT: If you run ragent from its own project root, set this to a
    /// different directory (e.g. --workspace ~/projects/myapp) so the agent
    /// cannot accidentally overwrite ragent's own source files.
    ///
    /// Example:
    ///   cargo run -- --workspace C:\Users\you\sandbox
    ///   cargo run -- --workspace /tmp/agent-workspace
    #[arg(long, short = 'w', value_name = "DIR")]
    workspace: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // --- Set up structured logging ---
    setup_logging(&cli.log_level)?;

    // --- Load config ---
    let mut config = AppConfig::load()
        .context("Failed to load configuration")?
        .with_overrides(cli.model, cli.ollama_url, cli.skill);

    // --- Apply workspace override ---
    //
    // The workspace flag sets *both* the shell working directory and the file
    // sandbox. This ensures the agent operates entirely within the specified
    // directory and cannot touch files outside it.
    let workspace = if let Some(ref ws) = cli.workspace {
        // Resolve to absolute path so it's unambiguous in logs.
        let abs = ws
            .canonicalize()
            .with_context(|| format!("Workspace directory does not exist: {}", ws.display()))?;
        info!(workspace = %abs.display(), "Using workspace directory");
        abs
    } else {
        // Default to current directory — warn the user if it looks like
        // they're running from ragent's own project root.
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        if cwd.join("src").join("main.rs").exists() && cwd.join("Cargo.toml").exists() {
            eprintln!(
                "\n{}",
                "Warning: No --workspace specified and the current directory looks like a Rust project.".yellow().bold()
            );
            eprintln!(
                "{}",
                "The agent may read or overwrite source files here, including ragent's own code."
                    .yellow()
            );
            eprintln!(
                "{}",
                "To isolate the agent, run with: ragent --workspace /path/to/your/workspace\n"
                    .cyan()
            );
        }
        cwd
    };

    // Push workspace into config so build_tool_registry picks it up.
    config.tools.file.sandbox_path = Some(workspace.to_string_lossy().into_owned());
    config.tools.shell.working_directory = Some(workspace.to_string_lossy().into_owned());

    // --- Auto-set CARGO_TARGET_DIR to avoid Windows MAX_PATH issues ---
    //
    // On Windows, Cargo's default `target/` lives inside the workspace, quickly
    // producing paths >260 chars that can't be deleted from cmd.exe/PowerShell.
    // We redirect it to a short path in %TEMP% shared across all sessions.
    // This speeds up incremental builds too (fingerprints survive workspace cleans).
    if config.tools.shell.cargo_target_dir.is_none() {
        let cargo_target = std::env::temp_dir().join("ragent-target");
        info!(
            cargo_target_dir = %cargo_target.display(),
            "Auto-setting CARGO_TARGET_DIR to avoid Windows MAX_PATH issues"
        );
        config.tools.shell.cargo_target_dir = Some(cargo_target.to_string_lossy().into_owned());
    }

    info!(
        model = %config.ollama.model,
        url = %config.ollama.url,
        workspace = %workspace.display(),
        "Starting ragent"
    );

    // --- Ensure Ollama is running ---
    let backend = OllamaBackend::new(config.ollama.clone());

    if let Err(e) = ensure_ollama_running(&backend).await {
        eprintln!("\n{}", format!("Error: {e}").red().bold());
        eprintln!(
            "\n{}",
            "ragent requires Ollama to be installed and running.".yellow()
        );
        eprintln!("  1. Install Ollama from: {}", "https://ollama.com".cyan());
        eprintln!(
            "  2. Pull a model:        {}",
            format!("ollama pull {}", config.ollama.model).cyan()
        );
        eprintln!("  3. Start Ollama:        {}", "ollama serve".cyan());
        std::process::exit(1);
    }

    // --- Check model availability ---
    if let Err(e) = backend.check_model().await {
        match e {
            OllamaError::ModelNotFound { ref model } => {
                eprintln!("\n{}", format!("Error: {e}").red().bold());
                let models = backend.list_models().await.unwrap_or_default();
                if !models.is_empty() {
                    eprintln!("\n{}", "Available models:".yellow());
                    for m in &models {
                        eprintln!("  - {}", m.cyan());
                    }
                }
                eprintln!(
                    "\n  Pull it with: {}",
                    format!("ollama pull {model}").cyan()
                );
                std::process::exit(1);
            }
            _ => {
                eprintln!("\n{}", format!("Error: {e}").red().bold());
                std::process::exit(1);
            }
        }
    }

    // --- Warn if model likely doesn't support tools ---
    warn_if_tools_unsupported(&config.ollama.model, &backend).await;

    // --- Build the tool registry ---
    let registry = build_tool_registry(&config);

    // --- Resolve skill and build system prompt ---
    //
    // A Skill is a named bundle of (system prompt + tool allowlist). The CLI
    // `--skill` flag selects which one to use; the config's `[skill]` section
    // provides the default. If an unknown name is given, we fall back to "chat"
    // with a warning so the user isn't silently broken.
    let skill = Skill::builtin(&config.skill.name).unwrap_or_else(|| {
        warn!(
            skill = %config.skill.name,
            "Unknown skill name, falling back to 'chat'"
        );
        Skill::chat()
    });
    info!(skill = %skill.name, "Skill selected");

    // --- Load project context (if enabled) ---
    //
    // The ProjectContext loader reads the workspace's directory tree and key
    // files (Cargo.toml, README, etc.) and formats them as Markdown. This
    // Markdown is appended to the skill's system prompt so the LLM starts
    // every session with a mental model of the project structure.
    let project_context_str = if config.project.auto_detect {
        match ProjectContext::load(
            &workspace,
            config.project.max_tree_depth,
            config.project.max_key_file_bytes,
        ) {
            Ok(ctx) => {
                info!(
                    project_type = %ctx.project_type.display_name(),
                    key_files = ctx.key_files.len(),
                    "Project context loaded"
                );
                Some(ctx.format_for_prompt())
            }
            Err(e) => {
                warn!(error = %e, "Failed to load project context — continuing without it");
                None
            }
        }
    } else {
        None
    };

    // Build the final system prompt: skill base + optional project context.
    let system_prompt = skill.build_system_prompt(project_context_str.as_deref());

    // --- Build the agent loop ---
    let agent = AgentLoop::new(
        backend,
        registry,
        &system_prompt,
        config.agent.max_iterations,
    );

    // --- Start REPL ---
    println!(
        "{}",
        format!(
            "ragent v{} -- chatting with {} via Ollama  [skill: {}]",
            env!("CARGO_PKG_VERSION"),
            config.ollama.model,
            skill.name
        )
        .green()
        .bold()
    );
    println!("{}", format!("Workspace: {}", workspace.display()).dimmed());
    println!(
        "{}",
        "Tools: file_read, file_write, file_patch, file_search, shell_exec".dimmed()
    );
    println!(
        "{}",
        "Type your message and press Enter. Commands: /clear, /quit, /help".dimmed()
    );
    println!("{}", "-".repeat(60).dimmed());

    run_repl(agent, &system_prompt, &config).await
}

/// Build the tool registry, wiring config into each tool.
fn build_tool_registry(config: &AppConfig) -> ToolRegistry {
    let sandbox_path = config
        .tools
        .file
        .sandbox_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let file_config = FileToolsConfig {
        max_read_bytes: config.tools.file.max_read_bytes,
        max_search_results: config.tools.file.max_search_results,
        sandbox_path,
    };

    let shell_config = ragent::tools::shell::ShellConfig {
        timeout_secs: config.tools.shell.timeout_secs,
        max_output_bytes: config.tools.shell.max_output_bytes,
        working_directory: config.tools.shell.working_directory.clone(),
        cargo_target_dir: config.tools.shell.cargo_target_dir.clone(),
    };

    let mut registry = ToolRegistry::new();
    registry.register(FileReadTool::new(file_config.clone()));
    registry.register(FileWriteTool::new(file_config.clone()));
    registry.register(FilePatchTool::new(file_config.clone()));
    registry.register(FileSearchTool::new(file_config));
    registry.register(ShellExecTool::new(shell_config));

    info!(tools = ?registry.names(), "Tool registry built");

    registry
}

/// Set up tracing/logging with the specified level.
fn setup_logging(level: &str) -> Result<()> {
    use tracing_subscriber::{EnvFilter, fmt};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("ragent={level},reqwest={level},{level}")));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .compact()
        .init();

    Ok(())
}

/// Try to connect to Ollama, auto-starting it if needed.
async fn ensure_ollama_running(backend: &OllamaBackend) -> Result<(), OllamaError> {
    if backend.check_connection().await.is_ok() {
        return Ok(());
    }

    info!("Ollama is not running, attempting to start it...");
    println!(
        "{}",
        "Ollama is not running. Attempting to start it...".yellow()
    );

    let ollama_exists = std::process::Command::new("ollama")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok();

    if !ollama_exists {
        return Err(OllamaError::NotInstalled);
    }

    let spawn_result = std::process::Command::new("ollama")
        .arg("serve")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();

    match spawn_result {
        Ok(child) => {
            info!(pid = child.id(), "Started Ollama serve process");
        }
        Err(e) => {
            warn!(error = %e, "Failed to start ollama serve (it might already be starting)");
        }
    }

    let max_retries = 15;
    let retry_delay = std::time::Duration::from_secs(1);

    for attempt in 1..=max_retries {
        tokio::time::sleep(retry_delay).await;

        if backend.check_connection().await.is_ok() {
            println!("{}", "Ollama is now running!".green());
            return Ok(());
        }

        if attempt % 3 == 0 {
            println!(
                "{}",
                format!("  Still waiting for Ollama... (attempt {attempt}/{max_retries})").dimmed()
            );
        }
    }

    Err(OllamaError::ConnectionFailed {
        url: "configured URL".to_string(),
    })
}

/// Check at startup whether the chosen model is known not to support tools.
async fn warn_if_tools_unsupported(model: &str, backend: &OllamaBackend) {
    let family = model.split(':').next().unwrap_or(model);

    if MODELS_WITHOUT_TOOL_SUPPORT.iter().any(|m| *m == family) {
        eprintln!(
            "\n{}",
            format!("Warning: '{}' is not known to support tool calling.", model)
                .yellow()
                .bold()
        );
        eprintln!(
            "{}",
            "The agent loop requires tool support. You may see errors.".yellow()
        );
        eprintln!(
            "Consider switching: {}",
            "ragent --model qwen2.5-coder:7b".cyan()
        );

        if let Ok(models) = backend.list_models().await {
            let tool_capable: Vec<_> = models
                .iter()
                .filter(|m| {
                    !MODELS_WITHOUT_TOOL_SUPPORT
                        .iter()
                        .any(|bad| m.split(':').next().unwrap_or("") == *bad)
                })
                .collect();
            if !tool_capable.is_empty() {
                eprintln!("\n{}", "Models you have that may support tools:".yellow());
                for m in &tool_capable {
                    eprintln!("  - {}", m.cyan());
                }
            }
        }
        eprintln!();
    }
}

/// Run the interactive REPL loop using the agent loop.
async fn run_repl(
    mut agent: AgentLoop<OllamaBackend>,
    system_prompt: &str,
    config: &AppConfig,
) -> Result<()> {
    let stdin = io::stdin();
    let mut reader = stdin.lock().lines();

    loop {
        print!("\n{} ", "you>".blue().bold());
        io::stdout().flush()?;

        let line = match reader.next() {
            Some(Ok(line)) => line,
            Some(Err(e)) => {
                error!(error = %e, "Failed to read input");
                break;
            }
            None => {
                println!("\n{}", "Goodbye!".green());
                break;
            }
        };

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/quit" | "/exit" => {
                println!("{}", "Goodbye!".green());
                break;
            }
            "/clear" => {
                agent.clear_conversation(system_prompt);
                println!("{}", "Conversation cleared.".yellow());
                continue;
            }
            "/help" => {
                println!("{}", "Commands:".yellow().bold());
                println!("  {}  -- Clear conversation history", "/clear".cyan());
                println!("  {}   -- Exit ragent", "/quit".cyan());
                println!("  {}   -- Exit ragent", "/exit".cyan());
                println!("  {}   -- Show this help", "/help".cyan());
                println!();
                println!("{}", "Available tools:".yellow().bold());
                println!("  {}  -- Read a file's contents", "file_read".cyan());
                println!(
                    "  {} -- Write content to a file (whole-file)",
                    "file_write".cyan()
                );
                println!(
                    "  {} -- Surgical find/replace edit in a file",
                    "file_patch".cyan()
                );
                println!(
                    "  {} -- Search for patterns in files (grep)",
                    "file_search".cyan()
                );
                println!("  {}  -- Execute a shell command", "shell_exec".cyan());
                println!(
                    "\n{}",
                    "The agent uses tools automatically when needed. Just describe what you want!"
                        .dimmed()
                );
                continue;
            }
            _ if input.starts_with('/') => {
                println!(
                    "{}",
                    format!("Unknown command: {input}. Type /help for available commands.")
                        .yellow()
                );
                continue;
            }
            _ => {}
        }

        print!("{} ", "assistant>".magenta().bold());
        io::stdout().flush()?;

        match agent.process_message(input).await {
            Ok(response) => {
                println!("{}", response);
            }
            Err(e) => {
                // Check if this is the "model doesn't support tools" error.
                let err_msg = format!("{e:#}");
                if err_msg.contains("does not support tool") {
                    eprintln!("\n{}", format!("Error: {e}").red().bold());
                    eprintln!(
                        "\n{}",
                        "This model does not support tool/function calling.".yellow()
                    );
                    eprintln!("Try one of these models with tool support:");
                    eprintln!("  {}", "ragent --model qwen2.5-coder:7b".cyan());
                    eprintln!("  {}", "ragent --model qwen2.5-coder:32b".cyan());
                    eprintln!("  {}", "ragent --model mistral:7b".cyan());

                    if let Ok(models) =
                        ragent::llm::ollama::OllamaBackend::new(config.ollama.clone())
                            .list_models()
                            .await
                    {
                        let tool_capable: Vec<_> = models
                            .iter()
                            .filter(|m| {
                                !MODELS_WITHOUT_TOOL_SUPPORT
                                    .iter()
                                    .any(|bad| m.split(':').next().unwrap_or("") == *bad)
                            })
                            .collect();
                        if !tool_capable.is_empty() {
                            eprintln!(
                                "\n{}",
                                "Models you already have that may support tools:".yellow()
                            );
                            for m in &tool_capable {
                                eprintln!("  - {}", m.cyan());
                            }
                        }
                    }
                    std::process::exit(1);
                }

                eprintln!("{}", format!("Error: {e}").red());
                warn!(error = %e, "Agent loop failed");
            }
        }
    }

    Ok(())
}
