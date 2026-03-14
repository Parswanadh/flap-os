from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deployment_scripts_exist_and_have_core_commands() -> None:
    setup_script = (ROOT / "scripts/setup_ubuntu.sh").read_text(encoding="utf-8")
    tailscale_script = (ROOT / "scripts/tailscale_setup.sh").read_text(encoding="utf-8")

    assert "docker compose up -d" in setup_script
    assert "systemctl enable flap-compose.service" in setup_script
    assert "systemctl enable screenpipe.service" in setup_script
    assert "tailscale up" in tailscale_script


def test_systemd_units_exist() -> None:
    assert (ROOT / "scripts/systemd/flap-compose.service").exists()
    assert (ROOT / "scripts/systemd/screenpipe.service").exists()


def test_readme_deployment_section_present() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Ubuntu Server Deployment" in readme
    assert "Local Development" in readme
