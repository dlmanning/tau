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
                println!("{:<38} {:<20} {:<8} Working Dir", "ID", "Created", "Msgs");
                println!("{}", "-".repeat(90));
                for s in sessions {
                    println!(
                        "{:<38} {:<20} {:<8} {}",
                        s.id,
                        s.created_at_display(),
                        s.message_count,
                        s.working_dir
                    );
                }
                println!("\nResume with: tau sessions resume <session-id>");
            }
        }
        Err(e) => {
            eprintln!("Error listing sessions: {}", e);
        }
    }
    Ok(())
}
