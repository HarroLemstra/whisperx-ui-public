from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.config import AppConfig, build_output_folder_name, load_environment, resolve_hf_token
from core.preflight import PreflightReport, run_preflight
from core.runner import WhisperXRunner
from core.types import JobRecord, JobSpec, utc_now_iso

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
}


class QueueManager:
    def __init__(self, config: AppConfig, runner: WhisperXRunner, logger) -> None:
        self.config = config
        self.runner = runner
        self.logger = logger

        self._lock = threading.RLock()
        self._wake_event = threading.Event()

        self.pending: List[JobSpec] = []
        self.running: Optional[JobSpec] = None
        self.running_attempt: int = 0
        self.running_started_at: Optional[str] = None
        self.done: List[JobRecord] = []
        self.failed: List[JobRecord] = []
        self.stop_after_current: bool = False
        self.watch_folder: Optional[str] = None
        self._session_token: Optional[str] = None
        self.runtime_profile: Dict[str, object] = {
            "min_speakers": self.config.default_min_speakers,
            "max_speakers": self.config.default_max_speakers,
            "output_root": str(self.config.output_root),
            "threads": self.config.default_threads,
            "chunk_size": self.config.default_chunk_size,
            "diarize_model": self.config.diarize_model_default,
            "language": self.config.language,
        }

        self._worker_thread: Optional[threading.Thread] = None
        self._watch_thread = threading.Thread(target=self._watch_loop, name="watch-folder", daemon=True)

        self._load_state()
        self._watch_thread.start()

    def enqueue_file(
        self,
        source_path: str,
        min_speakers: int,
        max_speakers: int,
        output_root: str,
        threads: int,
        chunk_size: int,
        diarize_model: str,
        language: str,
    ) -> Tuple[bool, str]:
        path = Path(source_path).expanduser()
        if not path.exists() or not path.is_file():
            return False, f"File not found: {path}"
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return False, f"Unsupported extension: {path.suffix}"

        stat = path.stat()
        fingerprint = self._fingerprint(path, stat.st_size, stat.st_mtime_ns)
        with self._lock:
            self.runtime_profile.update(
                {
                    "min_speakers": int(min_speakers),
                    "max_speakers": int(max_speakers),
                    "output_root": str(Path(output_root).expanduser()),
                    "threads": int(threads),
                    "chunk_size": int(chunk_size),
                    "diarize_model": diarize_model,
                    "language": language,
                }
            )
            if self._fingerprint_exists(fingerprint):
                return False, "File already in queue/history (duplicate skipped)."

            job = JobSpec(
                job_id=uuid.uuid4().hex[:10],
                source_path=str(path),
                source_ctime=stat.st_ctime,
                source_mtime_ns=stat.st_mtime_ns,
                source_size=stat.st_size,
                fingerprint=fingerprint,
                enqueued_at=utc_now_iso(),
                min_speakers=int(min_speakers),
                max_speakers=int(max_speakers),
                output_root=str(Path(output_root).expanduser()),
                threads=int(threads),
                chunk_size=int(chunk_size),
                diarize_model=diarize_model,
                language=language,
            )
            self.pending.append(job)
            self._save_state_locked()

        self._wake_event.set()
        self.logger.info("Queued job %s for %s", job.job_id, job.source_path)
        return True, f"Queued: {path.name} ({job.job_id})"

    def enqueue_from_watch_folder(
        self,
        min_speakers: int,
        max_speakers: int,
        output_root: str,
        threads: int,
        chunk_size: int,
        diarize_model: str,
        language: str,
    ) -> int:
        with self._lock:
            self.runtime_profile.update(
                {
                    "min_speakers": int(min_speakers),
                    "max_speakers": int(max_speakers),
                    "output_root": str(Path(output_root).expanduser()),
                    "threads": int(threads),
                    "chunk_size": int(chunk_size),
                    "diarize_model": diarize_model,
                    "language": language,
                }
            )

        with self._lock:
            folder = self.watch_folder
        if not folder:
            return 0

        watch_path = Path(folder).expanduser()
        if not watch_path.exists() or not watch_path.is_dir():
            return 0

        added = 0
        for candidate in sorted(watch_path.iterdir()):
            if not candidate.is_file() or candidate.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            ok, _ = self.enqueue_file(
                source_path=str(candidate),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                output_root=output_root,
                threads=threads,
                chunk_size=chunk_size,
                diarize_model=diarize_model,
                language=language,
            )
            if ok:
                added += 1
        return added

    def set_watch_folder(self, folder: str) -> Tuple[bool, str]:
        cleaned = folder.strip()
        if not cleaned:
            with self._lock:
                self.watch_folder = None
                self._save_state_locked()
            return True, "Watch folder cleared."

        path = Path(cleaned).expanduser()
        if not path.exists() or not path.is_dir():
            return False, f"Watch folder not found: {path}"

        with self._lock:
            self.watch_folder = str(path)
            self._save_state_locked()
        return True, f"Watch folder set: {path}"

    def clear_pending(self) -> int:
        with self._lock:
            count = len(self.pending)
            self.pending.clear()
            self._save_state_locked()
        if count:
            self.logger.info("Cleared %s pending jobs.", count)
        return count

    def request_stop_after_current(self) -> str:
        with self._lock:
            self.stop_after_current = True
            self._save_state_locked()
        self.logger.info("Stop-after-current requested.")
        return "Stop requested: current file will finish, then queue pauses."

    def start_processing(self, ui_token: Optional[str]) -> Tuple[bool, str, PreflightReport]:
        env_map = load_environment(self.config.base_dir)
        token = resolve_hf_token(ui_token, env_map)
        report = run_preflight(self.config, token, diarize=True)
        if not report.ok:
            return False, "Preflight failed. Fix checks before starting queue.", report

        with self._lock:
            self._session_token = ui_token.strip() if ui_token and ui_token.strip() else None
            self.stop_after_current = False
            self._ensure_worker_locked()
            self._save_state_locked()

        self._wake_event.set()
        return True, "Queue started.", report

    def get_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "pending": [job.to_dict() for job in self.pending],
                "running": self.running.to_dict() if self.running else None,
                "running_attempt": self.running_attempt,
                "running_started_at": self.running_started_at,
                "done": [record.to_dict() for record in self.done],
                "failed": [record.to_dict() for record in self.failed],
                "stop_after_current": self.stop_after_current,
                "watch_folder": self.watch_folder,
                "runtime_profile": dict(self.runtime_profile),
                "worker_alive": bool(self._worker_thread and self._worker_thread.is_alive()),
            }

    def _ensure_worker_locked(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(target=self._worker_loop, name="queue-worker", daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        self.logger.info("Queue worker started.")
        while True:
            self._wake_event.wait(timeout=1.0)
            self._wake_event.clear()

            while True:
                with self._lock:
                    if self.stop_after_current:
                        break
                    if not self.pending:
                        break
                    job = self.pending.pop(0)
                    self.running = job
                    self.running_attempt = 0
                    self.running_started_at = utc_now_iso()
                    self._save_state_locked()

                self._process_job(job)

    def _process_job(self, job: JobSpec) -> None:
        output_dir = self._allocate_output_dir(Path(job.output_root), Path(job.source_path), job.job_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        job_log_path = output_dir / "job.log"
        started_at = utc_now_iso()
        status = "failed"
        attempts = 0
        artifacts: Dict[str, str] = {}
        error_message: Optional[str] = None

        token = self._resolve_runtime_token()
        if not token:
            error_message = "Missing HF token (HF_TOKEN/HF_token or UI session token)."
            self.logger.error("Job %s failed before start: %s", job.job_id, error_message)
        else:
            for attempt in (1, 2):
                attempts = attempt
                with self._lock:
                    self.running_attempt = attempt
                    self._save_state_locked()
                try:
                    artifacts = self.runner.execute(
                        job=job,
                        hf_token=token,
                        output_dir=output_dir,
                        job_log_path=job_log_path,
                        attempt=attempt,
                    )
                    status = "done"
                    error_message = None
                    break
                except Exception as exc:  # pragma: no cover - subprocess path
                    error_message = str(exc)
                    self.logger.exception("Job %s attempt %s failed.", job.job_id, attempt)
                    if attempt == 1:
                        self._append_job_log(job_log_path, "Retrying once after failure...")

        finished_at = utc_now_iso()
        record = JobRecord(
            job_id=job.job_id,
            source_path=job.source_path,
            status=status,
            attempts=attempts if attempts else 1,
            started_at=started_at,
            finished_at=finished_at,
            output_dir=str(output_dir),
            error_message=error_message,
            artifacts=artifacts,
            fingerprint=job.fingerprint,
        )
        meta_path = output_dir / "meta.json"
        meta_payload = self._build_meta_payload(job, record, output_dir)
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        record.meta_path = str(meta_path)

        with self._lock:
            self.running = None
            self.running_attempt = 0
            self.running_started_at = None
            if status == "done":
                self.done.insert(0, record)
            else:
                self.failed.insert(0, record)
            self._save_state_locked()

        if status == "done":
            self.logger.info("Job %s completed in %s attempt(s).", job.job_id, record.attempts)
        else:
            self.logger.error("Job %s failed after %s attempt(s): %s", job.job_id, record.attempts, error_message)

    def _watch_loop(self) -> None:
        while True:
            time.sleep(self.config.watch_interval_seconds)
            with self._lock:
                folder = self.watch_folder
                runtime = dict(self.runtime_profile)
            if not folder:
                continue
            try:
                added = self.enqueue_from_watch_folder(
                    min_speakers=int(runtime["min_speakers"]),
                    max_speakers=int(runtime["max_speakers"]),
                    output_root=str(runtime["output_root"]),
                    threads=int(runtime["threads"]),
                    chunk_size=int(runtime["chunk_size"]),
                    diarize_model=str(runtime["diarize_model"]),
                    language=str(runtime["language"]),
                )
                if added:
                    self.logger.info("Watch folder enqueued %s new file(s).", added)
                    self._wake_event.set()
            except Exception:
                self.logger.exception("Watch folder scan failed.")

    def _fingerprint(self, path: Path, size: int, mtime_ns: int) -> str:
        return f"{path.resolve()}::{size}::{mtime_ns}"

    def _fingerprint_exists(self, fingerprint: str) -> bool:
        path_part = fingerprint.split("::", 1)[0]
        if any(job.fingerprint == fingerprint for job in self.pending):
            return True
        if self.running and self.running.fingerprint == fingerprint:
            return True
        if any(
            (record.fingerprint == fingerprint)
            or (record.fingerprint is None and str(Path(record.source_path)) == path_part)
            for record in self.done
        ):
            return True
        if any(
            (record.fingerprint == fingerprint)
            or (record.fingerprint is None and str(Path(record.source_path)) == path_part)
            for record in self.failed
        ):
            return True
        return False

    def _resolve_runtime_token(self) -> Optional[str]:
        env_map = load_environment(self.config.base_dir)
        return resolve_hf_token(self._session_token, env_map)

    def _allocate_output_dir(self, output_root: Path, source_path: Path, job_id: str) -> Path:
        output_root.mkdir(parents=True, exist_ok=True)
        base_name = build_output_folder_name(source_path)
        candidate = output_root / base_name
        if not candidate.exists():
            return candidate
        return output_root / f"{base_name}__{job_id}"

    def _build_meta_payload(self, job: JobSpec, record: JobRecord, output_dir: Path) -> Dict[str, object]:
        return {
            "job": job.to_dict(),
            "result": record.to_dict(),
            "runtime": {
                "model": self.config.model_name,
                "device": self.config.device,
                "compute_type": self.config.compute_type,
                "vad_method": self.config.vad_method,
                "segment_resolution": self.config.segment_resolution,
                "align_fallback_model": self.config.align_fallback_model,
                "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            },
            "output_dir": str(output_dir),
        }

    def _append_job_log(self, job_log_path: Path, text: str) -> None:
        with job_log_path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")

    def _load_state(self) -> None:
        self.config.ensure_directories()
        if not self.config.state_file.exists():
            return
        try:
            payload = json.loads(self.config.state_file.read_text(encoding="utf-8"))
        except Exception:
            self.logger.exception("Failed to read queue state; starting with empty state.")
            return

        self.pending = [JobSpec.from_dict(item) for item in payload.get("pending", [])]
        running_data = payload.get("running")
        self.running = JobSpec.from_dict(running_data) if running_data else None
        self.running_attempt = int(payload.get("running_attempt", 0))
        self.running_started_at = payload.get("running_started_at")
        self.done = [JobRecord.from_dict(item) for item in payload.get("done", [])]
        self.failed = [JobRecord.from_dict(item) for item in payload.get("failed", [])]
        self.stop_after_current = bool(payload.get("stop_after_current", False))
        self.watch_folder = payload.get("watch_folder")
        self.runtime_profile.update(payload.get("runtime_profile") or {})

        if self.running:
            interrupted = JobRecord(
                job_id=self.running.job_id,
                source_path=self.running.source_path,
                status="failed",
                attempts=max(1, self.running_attempt),
                started_at=self.running_started_at or utc_now_iso(),
                finished_at=utc_now_iso(),
                output_dir=None,
                error_message="Application restarted while job was running.",
            )
            self.failed.insert(0, interrupted)
            self.running = None
            self.running_attempt = 0
            self.running_started_at = None
            self._save_state_locked()

    def _save_state_locked(self) -> None:
        payload = {
            "pending": [job.to_dict() for job in self.pending],
            "running": self.running.to_dict() if self.running else None,
            "running_attempt": self.running_attempt,
            "running_started_at": self.running_started_at,
            "done": [record.to_dict() for record in self.done],
            "failed": [record.to_dict() for record in self.failed],
            "stop_after_current": self.stop_after_current,
            "watch_folder": self.watch_folder,
            "runtime_profile": dict(self.runtime_profile),
            "updated_at": utc_now_iso(),
        }
        temp_path = self.config.state_file.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self.config.state_file)
