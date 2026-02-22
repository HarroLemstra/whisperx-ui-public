from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download

from core.command_resolver import resolve_whisperx_command
from core.config import AppConfig


@dataclass
class PreflightReport:
    ok: bool
    checks: List[Dict[str, str]]


def _summarize_hf_error(raw_detail: str) -> str:
    text = (raw_detail or "").strip()
    lower = text.lower()
    if "public gated repositories" in lower:
        return (
            "HF token mist toestemming voor public gated repos. "
            "Zet in Hugging Face token settings: 'Read access to contents of all public gated repos', "
            "of maak een classic READ token."
        )
    if "403" in lower or "forbidden" in lower:
        return (
            "HF toegang geweigerd (403) voor diarization model. "
            "Controleer token-rechten en dat je terms hebt geaccepteerd voor "
            "pyannote/speaker-diarization-3.1 en pyannote/segmentation-3.0."
        )
    if "401" in lower or "unauthorized" in lower:
        return "HF token ongeldig of verlopen voor diarization model."
    if "localentrynotfounderror" in lower:
        return (
            "Diarization model niet in cache en downloaden faalde. "
            "Controleer internetverbinding en HF toegang."
        )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "Unable to access diarization model on Hugging Face."
    return " | ".join(lines[:2])


def _check_command(command: list[str], timeout_seconds: int = 15) -> tuple[bool, str]:
    executable_path = Path(command[0])
    if executable_path.is_file():
        executable = str(executable_path)
    else:
        executable = shutil.which(command[0])
    if executable is None:
        return False, f"{command[0]} not found"
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

    whisperx_command, whisperx_display = resolve_whisperx_command()
    whisperx_ok, whisperx_detail = _check_command([*whisperx_command, "--help"])
    if whisperx_ok:
        whisperx_detail = whisperx_display
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
        if token_ok:
            try:
                hf_hub_download(
                    repo_id=config.diarize_model_default,
                    filename="config.yaml",
                    token=hf_token,
                    cache_dir=str(config.hf_hub_cache_dir),
                )
            except Exception as exc:  # pragma: no cover - defensive
                combined = str(exc)
                cause = getattr(exc, "__cause__", None)
                if cause:
                    combined = f"{combined}\n{cause}"
                checks.append(
                    {
                        "name": "hf_diarize_model",
                        "ok": "false",
                        "detail": _summarize_hf_error(combined),
                    }
                )
            else:
                checks.append(
                    {
                        "name": "hf_diarize_model",
                        "ok": "true",
                        "detail": config.diarize_model_default,
                    }
                )

    overall_ok = all(check["ok"] == "true" for check in checks)
    return PreflightReport(ok=overall_ok, checks=checks)
