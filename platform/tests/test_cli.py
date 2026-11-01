import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from activation_monitor.cli import main, parser

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_confirms_project_isolation(capsys):
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert Path(report["environment"]) == ROOT / ".venv"
    assert Path(report["base_python"]).is_relative_to(ROOT / ".python")
    assert Path(report["torch_path"]).is_relative_to(ROOT / ".venv")
    assert Path(report["huggingface_cache"]).is_relative_to(ROOT / ".cache")
    assert not any("mech_interp/.venv" in path for path in sys.path)


def test_launcher_ignores_foreign_environment(tmp_path):
    poison = tmp_path / "activation_monitor"
    poison.mkdir()
    (poison / "__init__.py").write_text("raise RuntimeError('foreign package imported')")
    env = dict(
        os.environ,
        VIRTUAL_ENV=str(tmp_path),
        PYTHONPATH=str(tmp_path),
        UV_PROJECT_ENVIRONMENT=str(tmp_path),
    )
    result = subprocess.run(
        [str(ROOT / "dev"), "doctor"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)["environment"] == str(ROOT / ".venv")


def test_setup_refuses_symlink_to_another_environment(tmp_path):
    launcher = tmp_path / "dev"
    shutil.copy2(ROOT / "dev", launcher)
    (tmp_path / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    result = subprocess.run([str(launcher), "setup"], text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "Refusing symlinked project directory" in result.stderr


@pytest.mark.parametrize(
    "option,value",
    [
        ("--max-new-tokens", "0"),
        ("--max-input-tokens", "-1"),
        ("--warmup-tokens", "-1"),
    ],
)
def test_invalid_limits_are_rejected(option, value):
    with pytest.raises(SystemExit):
        parser().parse_args(["generate", "--prompt", "hello", option, value])


def test_empty_prompt_does_not_load_model(capsys):
    assert main(["generate", "--prompt", " "]) == 1
    assert "Prompt must not be empty" in capsys.readouterr().err
