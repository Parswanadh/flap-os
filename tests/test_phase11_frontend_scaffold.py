from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_views_exist() -> None:
    required = [
        "frontend/src/lib/ChatView.svelte",
        "frontend/src/lib/TerminalView.svelte",
        "frontend/src/lib/MemoryView.svelte",
        "frontend/src/lib/AgentView.svelte",
        "frontend/src/lib/SettingsView.svelte",
        "frontend/src/routes/+page.svelte",
        "frontend/src/routes/+layout.svelte",
        "frontend/package.json",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert not missing, f"Missing frontend files: {missing}"


def test_tauri_config_exists() -> None:
    assert (ROOT / "frontend/src-tauri/Cargo.toml").exists()
    assert (ROOT / "frontend/src-tauri/tauri.conf.json").exists()
    assert (ROOT / "frontend/src-tauri/src/main.rs").exists()
