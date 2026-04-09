//! System prompt construction — modeled after Claude Code's prompt architecture.
//!
//! Static sections are stored as markdown files alongside this module and
//! embedded at compile time via `include_str!`. Dynamic sections (env, tools)
//! are assembled at runtime.

pub struct PromptOptions<'a> {
    pub tool_names: &'a [&'a str],
    pub cwd: &'a str,
    /// Enable acolyte mode: stricter verification, comment philosophy,
    /// faithful reporting, richer communication style.
    pub acolyte_mode: bool,
}

// Static sections — embedded at compile time
const INTRO: &str = include_str!("intro.md");
const SYSTEM: &str = include_str!("system.md");
const TASKS: &str = include_str!("tasks.md");
const TASKS_ACOLYTE: &str = include_str!("tasks_acolyte.md");
const ACTIONS: &str = include_str!("actions.md");
const STYLE: &str = include_str!("style.md");
const OUTPUT_EFFICIENCY: &str = include_str!("output_efficiency.md");
const OUTPUT_ACOLYTE: &str = include_str!("output_acolyte.md");

/// Build the complete system prompt.
pub fn build_system_prompt(opts: &PromptOptions) -> String {
    let mut sections: Vec<&str> = vec![INTRO, SYSTEM];

    // Tasks section — base + optional acolyte additions
    if opts.acolyte_mode {
        let combined = format!("{}\n{}", TASKS, TASKS_ACOLYTE);
        let tools = using_tools_section(opts.tool_names);

        let style = if opts.acolyte_mode {
            // Acolyte: no "short and concise" bullet, use richer output section
            format!("{}\n\n{}", STYLE, OUTPUT_ACOLYTE)
        } else {
            format!(
                "{}\n - Your responses should be short and concise.\n\n{}",
                STYLE, OUTPUT_EFFICIENCY
            )
        };

        let env = env_section(opts.cwd);

        return [
            INTRO,
            SYSTEM,
            &combined,
            ACTIONS,
            &tools,
            &style,
            &env,
        ]
        .join("\n\n");
    }

    sections.push(TASKS);
    sections.push(ACTIONS);

    let tools = using_tools_section(opts.tool_names);
    let style = format!(
        "{}\n - Your responses should be short and concise.",
        STYLE
    );
    let env = env_section(opts.cwd);

    [
        INTRO,
        SYSTEM,
        TASKS,
        ACTIONS,
        &tools,
        &style,
        OUTPUT_EFFICIENCY,
        &env,
    ]
    .join("\n\n")
}

// ============================================================================
// Dynamic sections — assembled at runtime
// ============================================================================

fn using_tools_section(tool_names: &[&str]) -> String {
    let has_bash = tool_names.contains(&"bash");
    let has_read = tool_names.contains(&"read");
    let has_edit = tool_names.contains(&"edit");
    let has_write = tool_names.contains(&"write");
    let has_glob = tool_names.contains(&"glob");
    let has_grep = tool_names.contains(&"grep");

    let mut s = String::from("# Using your tools\n");

    if has_bash {
        s.push_str(
            " - Do NOT use the Bash tool to run commands when a relevant dedicated tool is provided. Using dedicated tools allows the user to better understand and review your work. This is CRITICAL to assisting the user:\n",
        );
        if has_read {
            s.push_str("   - To read files use Read instead of cat, head, tail, or sed\n");
        }
        if has_edit {
            s.push_str("   - To edit files use Edit instead of sed or awk\n");
        }
        if has_write {
            s.push_str(
                "   - To create files use Write instead of cat with heredoc or echo redirection\n",
            );
        }
        if has_glob {
            s.push_str("   - To search for files use Glob instead of find or ls\n");
        }
        if has_grep {
            s.push_str(
                "   - To search the content of files, use Grep instead of grep or rg\n",
            );
        }
        s.push_str("   - Reserve using the Bash exclusively for system commands and terminal operations that require shell execution.\n");
    }

    s.push_str(" - You can call multiple tools in a single response. If you intend to call multiple tools and there are no dependencies between them, make all independent tool calls in parallel.");

    s
}

fn env_section(cwd: &str) -> String {
    let platform = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let shell = std::env::var("SHELL").unwrap_or_default();

    let mut s = format!(
        "# Environment\n - Working directory: {}\n - Platform: {} ({})",
        cwd, platform, arch,
    );

    if !shell.is_empty() {
        s.push_str(&format!("\n - Shell: {}", shell));
    }

    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
    {
        if output.status.success() {
            let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
            s.push_str(&format!("\n - Git branch: {}", branch));
        }
    }

    s
}
