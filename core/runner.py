from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from core.command_resolver import resolve_whisperx_command
from core.config import AppConfig
from core.types import JobSpec


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()


class WhisperXRunner:
    def __init__(self, config: AppConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.whisperx_command, self.whisperx_display = resolve_whisperx_command()

    def execute(
        self,
        job: JobSpec,
        hf_token: str,
        output_dir: Path,
        job_log_path: Path,
        attempt: int,
    ) -> Dict[str, str]:
        source_path = Path(job.source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if not hf_token.strip():
            raise RuntimeError("HF token is empty; diarization requires a token.")

        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = self.config.temp_dir / f"{job.job_id}_attempt{attempt}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        normalized_wav = temp_dir / "input_16k_mono.wav"

        try:
            self._append_job_log(job_log_path, f"Attempt {attempt} started at {self._now_iso()}")
            self._convert_to_wav(source_path, normalized_wav, job_log_path)
            self._run_whisperx_with_align_fallback(
                wav_path=normalized_wav,
                output_dir=output_dir,
                job=job,
                hf_token=hf_token,
                job_log_path=job_log_path,
            )
            artifacts = self._postprocess_outputs(output_dir)
            self._append_job_log(job_log_path, f"Attempt {attempt} finished successfully.")
            return artifacts
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _convert_to_wav(self, source_path: Path, target_wav: Path, job_log_path: Path) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(target_wav),
        ]
        result = self._run_command(command, job_log_path)
        if result.returncode != 0:
            raise RuntimeError(self._format_command_error("ffmpeg conversion failed", result))

    def _run_whisperx_with_align_fallback(
        self,
        wav_path: Path,
        output_dir: Path,
        job: JobSpec,
        hf_token: str,
        job_log_path: Path,
    ) -> None:
        base_cmd = self._build_whisperx_command(
            wav_path=wav_path,
            output_dir=output_dir,
            job=job,
            hf_token=hf_token,
            align_model=None,
        )
        first_result = self._run_command(base_cmd, job_log_path, sensitive_values=[hf_token])
        if first_result.returncode == 0:
            return

        if self._looks_like_alignment_failure(first_result.combined):
            self._append_job_log(
                job_log_path,
                "Detected possible alignment issue, retrying with explicit NL align model.",
            )
            fallback_cmd = self._build_whisperx_command(
                wav_path=wav_path,
                output_dir=output_dir,
                job=job,
                hf_token=hf_token,
                align_model=self.config.align_fallback_model,
            )
            second_result = self._run_command(fallback_cmd, job_log_path, sensitive_values=[hf_token])
            if second_result.returncode == 0:
                return
            raise RuntimeError(self._format_command_error("whisperx failed after align fallback", second_result))

        raise RuntimeError(self._format_command_error("whisperx failed", first_result))

    def _build_whisperx_command(
        self,
        wav_path: Path,
        output_dir: Path,
        job: JobSpec,
        hf_token: str,
        align_model: Optional[str],
    ) -> List[str]:
        cmd = [
            *self.whisperx_command,
            str(wav_path),
            "--model",
            self.config.model_name,
            "--language",
            job.language,
            "--device",
            self.config.device,
            "--compute_type",
            self.config.compute_type,
            "--vad_method",
            self.config.vad_method,
            "--chunk_size",
            str(job.chunk_size),
            "--diarize",
            "--diarize_model",
            job.diarize_model,
            "--hf_token",
            hf_token,
            "--min_speakers",
            str(job.min_speakers),
            "--max_speakers",
            str(job.max_speakers),
            "--output_dir",
            str(output_dir),
            "--output_format",
            "all",
            "--segment_resolution",
            self.config.segment_resolution,
            "--threads",
            str(job.threads),
        ]
        if align_model:
            cmd.extend(["--align_model", align_model])
        return cmd

    def _run_command(
        self,
        command: List[str],
        job_log_path: Path,
        sensitive_values: Optional[Iterable[str]] = None,
    ) -> CommandResult:
        masked = self._mask_command(command, sensitive_values or [])
        self.logger.info("Running: %s", masked)
        self._append_job_log(job_log_path, f"$ {masked}")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.stdout:
            self._append_job_log(job_log_path, completed.stdout.rstrip())
        if completed.stderr:
            self._append_job_log(job_log_path, completed.stderr.rstrip())
        self._append_job_log(job_log_path, f"Return code: {completed.returncode}")
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    def _postprocess_outputs(self, output_dir: Path) -> Dict[str, str]:
        json_path = self._latest_file(output_dir, ".json")
        if json_path is None:
            raise RuntimeError("No JSON output generated by whisperx.")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        segments = payload.get("segments") or []

        transcript_json = output_dir / "transcript.json"
        if json_path.resolve() != transcript_json.resolve():
            shutil.copyfile(json_path, transcript_json)
        else:
            transcript_json = json_path

        transcript_txt = output_dir / "transcript.txt"
        transcript_srt = output_dir / "transcript.srt"

        if segments:
            txt_text = self._render_txt(segments)
            srt_text = self._render_srt(segments)
            transcript_txt.write_text(txt_text, encoding="utf-8")
            transcript_srt.write_text(srt_text, encoding="utf-8")
        else:
            fallback_txt = self._latest_file(output_dir, ".txt")
            fallback_srt = self._latest_file(output_dir, ".srt")
            if fallback_txt and fallback_txt.resolve() != transcript_txt.resolve():
                shutil.copyfile(fallback_txt, transcript_txt)
            else:
                transcript_txt.write_text(str(payload.get("text", "")).strip(), encoding="utf-8")
            if fallback_srt and fallback_srt.resolve() != transcript_srt.resolve():
                shutil.copyfile(fallback_srt, transcript_srt)
            else:
                transcript_srt.write_text("", encoding="utf-8")

        return {
            "srt": str(transcript_srt),
            "txt": str(transcript_txt),
            "json": str(transcript_json),
        }

    def _render_txt(self, segments: List[dict]) -> str:
        lines: List[str] = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            speaker = str(segment.get("speaker", "")).strip()
            if speaker:
                lines.append(f"[{speaker}] {text}")
            else:
                lines.append(text)
        return "\n".join(lines).strip() + "\n"

    def _render_srt(self, segments: List[dict]) -> str:
        lines: List[str] = []
        index = 1
        for segment in segments:
            if "start" not in segment or "end" not in segment:
                continue
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            speaker = str(segment.get("speaker", "")).strip()
            if speaker:
                text = f"[{speaker}] {text}"
            start = self._srt_timestamp(float(segment["start"]))
            end = self._srt_timestamp(float(segment["end"]))
            lines.append(str(index))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
            index += 1
        return "\n".join(lines).strip() + "\n"

    def _latest_file(self, directory: Path, suffix: str) -> Optional[Path]:
        candidates = [path for path in directory.glob(f"*{suffix}") if path.is_file()]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.stat().st_mtime)

    def _looks_like_alignment_failure(self, combined_output: str) -> bool:
        text = combined_output.lower()
        signals = [
            "align",
            "alignment",
            "wav2vec",
            "ctc",
            "phoneme",
        ]
        return any(token in text for token in signals)

    def _format_command_error(self, prefix: str, result: CommandResult) -> str:
        lines = (result.combined or "").splitlines()
        tail = "\n".join(lines[-20:]).strip()
        if not tail:
            tail = f"return code {result.returncode}"
        return f"{prefix}: {tail}"

    def _mask_command(self, command: List[str], sensitive_values: Iterable[str]) -> str:
        masked = list(command)
        for value in sensitive_values:
            if not value:
                continue
            masked = ["***" if token == value else token for token in masked]
        return subprocess.list2cmdline(masked)

    def _append_job_log(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _srt_timestamp(self, seconds: float) -> str:
        total_ms = max(0, int(round(seconds * 1000)))
        hours, rem = divmod(total_ms, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        sec, ms = divmod(rem, 1000)
        return f"{hours:02d}:{minutes:02d}:{sec:02d},{ms:03d}"
