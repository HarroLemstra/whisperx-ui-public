from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def _default_threads() -> int:
    logical = os.cpu_count() or 8
    return max(4, logical - 2)


def parse_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'") and len(value) >= 2:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_environment(base_dir: Path) -> Dict[str, str]:
    merged: Dict[str, str] = dict(os.environ)
    env_path = base_dir / ".env"
    merged.update(parse_env_file(env_path))
    return merged


def resolve_hf_token(ui_token: Optional[str], env_map: Dict[str, str]) -> Optional[str]:
    if ui_token and ui_token.strip():
        return ui_token.strip()
    if "HF_TOKEN" in env_map and env_map["HF_TOKEN"].strip():
        return env_map["HF_TOKEN"].strip()
    for key, value in env_map.items():
        if key.upper() == "HF_TOKEN" and str(value).strip():
            return str(value).strip()
    return None


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "audio"


def build_output_folder_name(source_path: Path) -> str:
    stat = source_path.stat()
    created = datetime.fromtimestamp(stat.st_ctime)
    timestamp = created.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}__{sanitize_name(source_path.stem)}"


@dataclass
class AppConfig:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    model_name: str = "large-v3"
    language: str = "nl"
    device: str = "cpu"
    compute_type: str = "float32"
    vad_method: str = "pyannote"
    segment_resolution: str = "sentence"
    diarize_model_default: str = "pyannote/speaker-diarization-3.1"
    align_fallback_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-dutch"
    default_min_speakers: int = 2
    default_max_speakers: int = 4
    default_chunk_size: int = 15
    default_threads: int = field(default_factory=_default_threads)
    watch_interval_seconds: int = 30
    gradio_host: str = "127.0.0.1"
    gradio_port: int = 7860

    logs_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    state_file: Path = field(init=False)
    output_root: Path = field(init=False)
    temp_dir: Path = field(init=False)
    app_log_path: Path = field(init=False)
    hf_home_dir: Path = field(init=False)
    hf_hub_cache_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"
        self.state_file = self.data_dir / "queue_state.json"
        self.output_root = self.base_dir / "out"
        self.temp_dir = self.base_dir / "tmp"
        self.app_log_path = self.logs_dir / "app.log"
        self.hf_home_dir = self.data_dir / "hf_home"
        self.hf_hub_cache_dir = self.hf_home_dir / "hub"

    def ensure_directories(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.hf_home_dir.mkdir(parents=True, exist_ok=True)
        self.hf_hub_cache_dir.mkdir(parents=True, exist_ok=True)
