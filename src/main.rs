//! ragent CLI — interactive chat with a local LLM via Ollama.
//!
//! This is the Phase 1 entry point: a REPL that sends messages to Ollama
//! and prints responses with coloured output. It also handles:
//! - Checking if Ollama is installed and running
//! - Auto-starting Ollama if it's not running
//! - Verifying the requested model is available
//!
//! # Phase 1 (PLAN.md)
//! Goal: "Chat with a local LLM via CLI, see all traffic."

use anyhow::{Context, Result};
use clap::Parser;
use colored::Colorize;
use ragent::agent::message::Conversation;
use ragent::config::AppConfig;
use ragent::llm::LlmBackend;
use ragent::llm::ollama::{OllamaBackend, OllamaError};
use std::io::{self, BufRead, Write};
use tracing::{error, info, warn};

/// ragent — a local-only agentic AI framework.
///
/// Chat with a local LLM through Ollama with full request/response logging.
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
    /// Use 'debug' to see full HTTP request/response payloads.
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // --- Set up structured logging ---
    setup_logging(&cli.log_level)?;

    // --- Load config ---
    let config = AppConfig::load()
        .context("Failed to load configuration")?
        .with_overrides(cli.model, cli.ollama_url);

    info!(
        model = %config.ollama.model,
        url = %config.ollama.url,
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

    // --- Start REPL ---
    println!(
        "\n{}",
        format!(
            "ragent v{} -- chatting with {} via Ollama",
            env!("CARGO_PKG_VERSION"),
            config.ollama.model
        )
        .green()
        .bold()
    );
    println!(
        "{}",
        "Type your message and press Enter. Commands: /clear, /quit, /help".dimmed()
    );
    println!("{}", "-".repeat(60).dimmed());

    run_repl(&backend).await
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
///
/// Strategy:
/// 1. Try connecting to the configured URL
/// 2. If that fails, check if `ollama` is in PATH
/// 3. If found, spawn `ollama serve` as a background process
/// 4. Poll for readiness with retries
async fn ensure_ollama_running(backend: &OllamaBackend) -> Result<(), OllamaError> {
    // First attempt: maybe Ollama is already running
    if backend.check_connection().await.is_ok() {
        return Ok(());
    }

    info!("Ollama is not running, attempting to start it...");
    println!(
        "{}",
        "Ollama is not running. Attempting to start it...".yellow()
    );

    // Check if 'ollama' is available in PATH
    let ollama_exists = std::process::Command::new("ollama")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok();

    if !ollama_exists {
        return Err(OllamaError::NotInstalled);
    }

    // Spawn `ollama serve` as a background process
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

    // Poll for readiness
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

/// Run the interactive REPL loop.
async fn run_repl(backend: &OllamaBackend) -> Result<()> {
    let mut conversation = Conversation::with_system_prompt(
        "You are a helpful AI assistant running locally via ragent. \
         Be concise and helpful.",
    );

    let stdin = io::stdin();
    let mut reader = stdin.lock().lines();

    loop {
        // Print prompt
        print!("\n{} ", "you>".blue().bold());
        io::stdout().flush()?;

        // Read input (synchronous — async stdin has edge cases on Windows)
        let line = match reader.next() {
            Some(Ok(line)) => line,
            Some(Err(e)) => {
                error!(error = %e, "Failed to read input");
                break;
            }
            None => {
                // EOF (Ctrl+Z on Windows, Ctrl+D on Unix)
                println!("\n{}", "Goodbye!".green());
                break;
            }
        };

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        match input {
            "/quit" | "/exit" => {
                println!("{}", "Goodbye!".green());
                break;
            }
            "/clear" => {
                conversation.clear();
                println!("{}", "Conversation cleared.".yellow());
                continue;
            }
            "/help" => {
                println!("{}", "Commands:".yellow().bold());
                println!("  {}  -- Clear conversation history", "/clear".cyan());
                println!("  {}   -- Exit ragent", "/quit".cyan());
                println!("  {}   -- Exit ragent", "/exit".cyan());
                println!("  {}   -- Show this help", "/help".cyan());
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

        // Add user message to conversation
        conversation.add_user_message(input);

        // Send to LLM
        print!("{} ", "assistant>".magenta().bold());
        io::stdout().flush()?;

        match backend.chat(conversation.messages(), &[]).await {
            Ok(response) => {
                println!("{}", response.content);
                conversation.push(response);
            }
            Err(e) => {
                eprintln!("{}", format!("Error communicating with Ollama: {e}").red());
                warn!(error = %e, "LLM request failed");
                // Don't add the failed exchange to history
            }
        }
    }

    Ok(())
}
