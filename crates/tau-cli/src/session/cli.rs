//! `tau sessions ls` rendering.

use super::store::SessionManager;

pub(crate) fn list_sessions_cli() -> anyhow::Result<()> {
    match SessionManager::list_sessions() {
        Ok(sessions) => {
            if sessions.is_empty() {
                println!("No saved sessions found.");
                println!(
                    "Sessions are stored in: {}",
                    SessionManager::sessions_dir().display()
                );
            } else {
                println!("Saved sessions:\n");
                println!(
                    "{:<10} {:<17} {:>5}  {:<44} Working Dir",
                    "ID", "Created", "Msgs", "First prompt"
                );
                println!("{}", "-".repeat(110));
                for s in sessions {
                    // Short id — `resume` accepts any unique prefix.
                    let short_id: String = s.id.chars().take(8).collect();
                    let preview = crate::utils::truncate_chars(&s.preview, 42);
                    println!(
                        "{:<10} {:<17} {:>5}  {:<44} {}",
                        short_id,
                        s.created_at_display(),
                        s.message_count,
                        preview,
                        s.working_dir
                    );
                }
                println!("\nResume with: tau sessions resume <id-prefix>");
            }
        }
        Err(e) => {
            eprintln!("Error listing sessions: {}", e);
        }
    }
    Ok(())
}
