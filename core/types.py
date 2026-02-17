from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class JobSpec:
    job_id: str
    source_path: str
    source_ctime: float
    source_mtime_ns: int
    source_size: int
    fingerprint: str
    enqueued_at: str
    min_speakers: int
    max_speakers: int
    output_root: str
    threads: int
    chunk_size: int
    diarize_model: str
    language: str = "nl"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_path": self.source_path,
            "source_ctime": self.source_ctime,
            "source_mtime_ns": self.source_mtime_ns,
            "source_size": self.source_size,
            "fingerprint": self.fingerprint,
            "enqueued_at": self.enqueued_at,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "output_root": self.output_root,
            "threads": self.threads,
            "chunk_size": self.chunk_size,
            "diarize_model": self.diarize_model,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobSpec":
        return cls(
            job_id=str(data["job_id"]),
            source_path=str(data["source_path"]),
            source_ctime=float(data["source_ctime"]),
            source_mtime_ns=int(data["source_mtime_ns"]),
            source_size=int(data["source_size"]),
            fingerprint=str(data["fingerprint"]),
            enqueued_at=str(data["enqueued_at"]),
            min_speakers=int(data["min_speakers"]),
            max_speakers=int(data["max_speakers"]),
            output_root=str(data["output_root"]),
            threads=int(data["threads"]),
            chunk_size=int(data["chunk_size"]),
            diarize_model=str(data.get("diarize_model", "pyannote/speaker-diarization-3.1")),
            language=str(data.get("language", "nl")),
        )


@dataclass
class JobRecord:
    job_id: str
    source_path: str
    status: str
    attempts: int
    started_at: str
    finished_at: str
    output_dir: Optional[str] = None
    error_message: Optional[str] = None
    artifacts: Optional[Dict[str, str]] = None
    meta_path: Optional[str] = None
    fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_path": self.source_path,
            "status": self.status,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_dir": self.output_dir,
            "error_message": self.error_message,
            "artifacts": self.artifacts or {},
            "meta_path": self.meta_path,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRecord":
        return cls(
            job_id=str(data["job_id"]),
            source_path=str(data["source_path"]),
            status=str(data["status"]),
            attempts=int(data.get("attempts", 1)),
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            output_dir=data.get("output_dir"),
            error_message=data.get("error_message"),
            artifacts=data.get("artifacts") or {},
            meta_path=data.get("meta_path"),
            fingerprint=data.get("fingerprint"),
        )
