// Desk shell: opens the window and runs the Python engine (app/engine/server.py) beside it.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

struct Engine(Mutex<Option<Child>>);

/// Repo root: $DESK_REPO if set, else three levels up from app/ui/src-tauri.
fn repo_root() -> PathBuf {
    if let Ok(p) = std::env::var("DESK_REPO") {
        return PathBuf::from(p);
    }
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    here.canonicalize().unwrap_or(here)
}

/// $DESK_PYTHON (app/run.sh points it at the env it keeps outside iCloud), else the repo's
/// virtualenv so the engine sees the same packages as the CLI tools, else python3.
fn python(root: &PathBuf) -> PathBuf {
    if let Ok(p) = std::env::var("DESK_PYTHON") {
        let p = PathBuf::from(p);
        if p.exists() {
            return p;
        }
    }
    // the env app/run.sh builds outside the repo; a Finder-launched Desk (built).app has no DESK_PYTHON
    if let Ok(home) = std::env::var("HOME") {
        let local = PathBuf::from(home).join(".local/share/desk/venv/bin/python");
        if local.exists() {
            return local;
        }
    }
    let venv = root.join(".venv/bin/python");
    if venv.exists() { venv } else { PathBuf::from("python3") }
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let root = repo_root();
            // Start from a fresh engine every time the window starts (including when `tauri dev`
            // relaunches it after a change here), so new engine routes are always live: stop any
            // engine left from an earlier window, then give the port a moment to free up.
            let stopped = Command::new("pkill")
                .args(["-f", "uvicorn app.engine.server:app"])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if stopped {
                std::thread::sleep(std::time::Duration::from_millis(600));
            }
            let child = Command::new(python(&root))
                .args(["-m", "uvicorn", "app.engine.server:app", "--host", "127.0.0.1", "--port", "8765", "--log-level", "warning"])
                .current_dir(&root)
                .spawn()
                .map_err(|e| eprintln!("Desk: could not start the engine: {e}"))
                .ok();
            app.manage(Engine(Mutex::new(child)));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Desk")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(state) = app.try_state::<Engine>() {
                    if let Some(mut c) = state.0.lock().unwrap().take() {
                        let _ = c.kill();
                    }
                }
            }
        });
}
