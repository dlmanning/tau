//! Git worktree management for isolated subagent execution.

use std::path::PathBuf;

pub(crate) struct WorktreeInfo {
    pub path: PathBuf,
    pub branch: String,
    pub head_commit: String,
}

pub(crate) async fn create_worktree(agent_id: &str) -> Result<WorktreeInfo, String> {
    let git_root_output = tokio::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .await
        .map_err(|e| format!("git rev-parse failed: {}", e))?;

    if !git_root_output.status.success() {
        return Err("Not in a git repository".into());
    }

    let git_root = PathBuf::from(String::from_utf8_lossy(&git_root_output.stdout).trim());
    let branch = format!("worktree-agent-{}", agent_id);
    let path = git_root.join(format!(".tau-worktrees/agent-{}", agent_id));

    tokio::fs::create_dir_all(path.parent().expect("joined path has parent"))
        .await
        .map_err(|e| format!("Failed to create worktree directory: {}", e))?;

    let output = tokio::process::Command::new("git")
        .args([
            "worktree",
            "add",
            &path.display().to_string(),
            "-b",
            &branch,
        ])
        .output()
        .await
        .map_err(|e| format!("git worktree add failed: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let head_output = tokio::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .await
        .map_err(|e| format!("git rev-parse HEAD failed: {}", e))?;

    let head_commit = String::from_utf8_lossy(&head_output.stdout)
        .trim()
        .to_string();

    Ok(WorktreeInfo {
        path,
        branch,
        head_commit,
    })
}

pub(crate) async fn cleanup_worktree(info: &WorktreeInfo) -> Result<bool, String> {
    let path_str = info.path.display().to_string();

    let diff = tokio::process::Command::new("git")
        .args(["-C", &path_str, "diff", "--quiet", &info.head_commit])
        .status()
        .await
        .map_err(|e| format!("git diff failed: {}", e))?;

    let untracked = tokio::process::Command::new("git")
        .args([
            "-C",
            &path_str,
            "ls-files",
            "--others",
            "--exclude-standard",
        ])
        .output()
        .await
        .map_err(|e| format!("git ls-files failed: {}", e))?;

    let has_untracked = !String::from_utf8_lossy(&untracked.stdout)
        .trim()
        .is_empty();

    if diff.success() && !has_untracked {
        let _ = tokio::process::Command::new("git")
            .args(["worktree", "remove", &path_str])
            .output()
            .await;
        let _ = tokio::process::Command::new("git")
            .args(["branch", "-D", &info.branch])
            .output()
            .await;
        Ok(true) // removed
    } else {
        Ok(false) // kept — has changes
    }
}
