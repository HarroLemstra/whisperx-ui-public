from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from core.config import AppConfig


@dataclass
class PreflightReport:
    ok: bool
    checks: List[Dict[str, str]]


def _check_command(command: list[str], timeout_seconds: int = 15) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return False, f"{command[0]} not found in PATH"
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return False, str(exc)
    if completed.returncode != 0:
        error_text = (completed.stderr or completed.stdout).strip()
        return False, error_text or f"{command[0]} returned {completed.returncode}"
    return True, executable


def run_preflight(config: AppConfig, hf_token: Optional[str], diarize: bool = True) -> PreflightReport:
    checks: List[Dict[str, str]] = []

    checks.append(
        {
            "name": "python",
            "ok": "true",
            "detail": f"{sys.executable} ({sys.version.split()[0]})",
        }
    )

    ffmpeg_ok, ffmpeg_detail = _check_command(["ffmpeg", "-version"])
    checks.append(
        {
            "name": "ffmpeg",
            "ok": str(ffmpeg_ok).lower(),
            "detail": ffmpeg_detail,
        }
    )

    whisperx_ok, whisperx_detail = _check_command(["whisperx", "--help"])
    checks.append(
        {
            "name": "whisperx",
            "ok": str(whisperx_ok).lower(),
            "detail": whisperx_detail,
        }
    )

    config.ensure_directories()
    checks.append(
        {
            "name": "directories",
            "ok": "true",
            "detail": f"{config.output_root} | {config.logs_dir} | {config.temp_dir}",
        }
    )

    if diarize:
        token_ok = bool(hf_token and hf_token.strip())
        checks.append(
            {
                "name": "hf_token",
                "ok": str(token_ok).lower(),
                "detail": "set" if token_ok else "Missing HF_TOKEN / HF_token",
            }
        )

    overall_ok = all(check["ok"] == "true" for check in checks)
    return PreflightReport(ok=overall_ok, checks=checks)
