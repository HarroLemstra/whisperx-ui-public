from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, List, Optional, Tuple

import gradio as gr

from core.config import AppConfig
from core.logging_setup import setup_logging, tail_text_file
from core.queue_manager import QueueManager
from core.runner import WhisperXRunner


config = AppConfig()
config.ensure_directories()
logger = setup_logging(config.logs_dir)
runner = WhisperXRunner(config=config, logger=logger)
queue_manager = QueueManager(config=config, runner=runner, logger=logger)


def _coerce_source_path(uploaded: Any, manual_path: str, explorer_value: Any) -> Optional[str]:
    if manual_path and manual_path.strip():
        return manual_path.strip()

    if isinstance(explorer_value, str) and explorer_value.strip():
        return explorer_value.strip()
    if isinstance(explorer_value, list) and explorer_value:
        first = explorer_value[0]
        if isinstance(first, str) and first.strip():
            return first.strip()

    if isinstance(uploaded, str) and uploaded.strip():
        return uploaded.strip()
    name = getattr(uploaded, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _render_table(snapshot: dict) -> List[List[str]]:
    rows: List[List[str]] = []
    for job in snapshot.get("pending", []):
        rows.append(
            [
                "pending",
                str(job.get("job_id", "")),
                str(job.get("source_path", "")),
                "",
                "",
                "",
            ]
        )

    running = snapshot.get("running")
    if running:
        rows.append(
            [
                "running",
                str(running.get("job_id", "")),
                str(running.get("source_path", "")),
                str(snapshot.get("running_attempt", "")),
                "",
                "",
            ]
        )

    for record in snapshot.get("done", [])[:80]:
        rows.append(
            [
                "done",
                str(record.get("job_id", "")),
                str(record.get("source_path", "")),
                str(record.get("attempts", "")),
                str(record.get("output_dir", "")),
                "",
            ]
        )

    for record in snapshot.get("failed", [])[:80]:
        rows.append(
            [
                "failed",
                str(record.get("job_id", "")),
                str(record.get("source_path", "")),
                str(record.get("attempts", "")),
                str(record.get("output_dir", "")),
                str(record.get("error_message", "")),
            ]
        )

    return rows


def _render_summary(snapshot: dict) -> str:
    pending_count = len(snapshot.get("pending", []))
    done_count = len(snapshot.get("done", []))
    failed_count = len(snapshot.get("failed", []))
    running = snapshot.get("running")
    running_text = str(running.get("source_path")) if running else "-"
    watch_folder = snapshot.get("watch_folder") or "-"
    stop_flag = bool(snapshot.get("stop_after_current"))
    worker_alive = bool(snapshot.get("worker_alive"))

    return (
        f"**Queue summary**\n\n"
        f"- Pending: `{pending_count}`\n"
        f"- Running: `{running_text}`\n"
        f"- Done: `{done_count}`\n"
        f"- Failed: `{failed_count}`\n"
        f"- Stop-after-current: `{stop_flag}`\n"
        f"- Worker active: `{worker_alive}`\n"
        f"- Watch folder: `{watch_folder}`"
    )


def _render_dashboard() -> Tuple[List[List[str]], str, str]:
    snapshot = queue_manager.get_snapshot()
    table = _render_table(snapshot)
    summary = _render_summary(snapshot)
    logs = tail_text_file(config.app_log_path, max_lines=180)
    return table, summary, logs


def _status_and_dashboard(message: str) -> Tuple[str, List[List[str]], str, str]:
    table, summary, logs = _render_dashboard()
    return message, table, summary, logs


def _format_preflight(checks: list[dict[str, str]]) -> str:
    lines = ["Preflight checks:"]
    for check in checks:
        ok = check.get("ok") == "true"
        icon = "OK" if ok else "FAIL"
        lines.append(f"- {icon} {check.get('name')}: {check.get('detail')}")
    return "\n".join(lines)


def add_file_callback(
    uploaded: Any,
    manual_path: str,
    explorer_value: Any,
    min_speakers: float,
    max_speakers: float,
    output_root: str,
    threads: float,
    chunk_size: float,
    diarize_model: str,
    language: str,
):
    source_path = _coerce_source_path(uploaded, manual_path, explorer_value)
    if not source_path:
        return _status_and_dashboard("No source file selected.")

    min_spk = max(1, int(min_speakers))
    max_spk = max(min_spk, int(max_speakers))
    thread_count = max(1, int(threads))
    chunk = max(5, int(chunk_size))
    output = output_root.strip() or str(config.output_root)
    diarize_model = diarize_model.strip() or config.diarize_model_default
    language = language.strip() or config.language

    ok, message = queue_manager.enqueue_file(
        source_path=source_path,
        min_speakers=min_spk,
        max_speakers=max_spk,
        output_root=output,
        threads=thread_count,
        chunk_size=chunk,
        diarize_model=diarize_model,
        language=language,
    )
    status = message if ok else f"Queue add failed: {message}"
    return _status_and_dashboard(status)


def set_watch_folder_callback(folder: str):
    ok, message = queue_manager.set_watch_folder(folder)
    status = message if ok else f"Watch folder update failed: {message}"
    return _status_and_dashboard(status)


def rescan_watch_callback(
    min_speakers: float,
    max_speakers: float,
    output_root: str,
    threads: float,
    chunk_size: float,
    diarize_model: str,
    language: str,
):
    added = queue_manager.enqueue_from_watch_folder(
        min_speakers=max(1, int(min_speakers)),
        max_speakers=max(max(1, int(min_speakers)), int(max_speakers)),
        output_root=output_root.strip() or str(config.output_root),
        threads=max(1, int(threads)),
        chunk_size=max(5, int(chunk_size)),
        diarize_model=diarize_model.strip() or config.diarize_model_default,
        language=language.strip() or config.language,
    )
    message = f"Rescan complete. Added {added} file(s) from watch folder."
    return _status_and_dashboard(message)


def start_queue_callback(ui_token: str):
    ok, message, report = queue_manager.start_processing(ui_token)
    details = _format_preflight(report.checks)
    if ok:
        combined = f"{message}\n\n{details}"
    else:
        combined = f"Queue NOT started.\n{message}\n\n{details}"
    return _status_and_dashboard(combined)


def stop_after_current_callback():
    message = queue_manager.request_stop_after_current()
    return _status_and_dashboard(message)


def clear_pending_callback():
    count = queue_manager.clear_pending()
    return _status_and_dashboard(f"Cleared {count} pending queue item(s).")


def kill_all_callback():
    message = queue_manager.kill_all()
    return _status_and_dashboard(message)


def refresh_callback():
    return _render_dashboard()


def build_ui() -> gr.Blocks:
    file_explorer_available = hasattr(gr, "FileExplorer")
    initial_snapshot = queue_manager.get_snapshot()
    initial_watch = initial_snapshot.get("watch_folder") or ""

    with gr.Blocks(title="WhisperX Night Queue") as demo:
        gr.Markdown("# WhisperX Night Queue (CPU + Diarization)")
        status_box = gr.Markdown("Ready.")

        with gr.Row():
            manual_path = gr.Textbox(label="Input file path", placeholder=r"C:\path\to\audio.wav")
            uploaded_file = gr.File(label="Optional upload", type="filepath")

        explorer = None
        if file_explorer_available:
            root_dir = str(Path(config.base_dir).anchor or config.base_dir)
            explorer = gr.FileExplorer(
                label="Browse local disk (server filesystem)",
                root_dir=root_dir,
                file_count="single",
            )
        else:
            gr.Markdown("`FileExplorer` is not available in this Gradio version.")

        with gr.Row():
            min_speakers = gr.Number(label="Min speakers", value=config.default_min_speakers, precision=0)
            max_speakers = gr.Number(label="Max speakers", value=config.default_max_speakers, precision=0)
            threads = gr.Number(label="Threads", value=config.default_threads, precision=0)
            chunk_size = gr.Number(label="Chunk size", value=config.default_chunk_size, precision=0)

        with gr.Row():
            output_root = gr.Textbox(label="Output root folder", value=str(config.output_root))
            diarize_model = gr.Textbox(
                label="Diarization model",
                value=config.diarize_model_default,
            )
            language = gr.Textbox(label="Language", value=config.language)

        hf_token = gr.Textbox(
            label="HF token override (optional session token)",
            type="password",
            placeholder="hf_...",
        )

        with gr.Row():
            add_button = gr.Button("Add file to queue", variant="primary")
            start_button = gr.Button("Start queue")
            stop_button = gr.Button("Stop after current")
            clear_button = gr.Button("Clear rest")
            kill_button = gr.Button("Kill all (force)", variant="stop")

        with gr.Row():
            watch_folder = gr.Textbox(label="Watch folder (optional)", value=initial_watch)
            set_watch_button = gr.Button("Set watch folder")
            rescan_button = gr.Button("Rescan watch folder")

        queue_table = gr.Dataframe(
            headers=["Status", "Job ID", "Source", "Attempts", "Output", "Error"],
            datatype=["str", "str", "str", "str", "str", "str"],
            row_count=10,
            column_count=(6, "fixed"),
            interactive=False,
            wrap=True,
            label="Queue / History",
        )
        summary_md = gr.Markdown()
        logs_box = gr.Textbox(label="Application log tail", lines=18, interactive=False)

        add_inputs = [
            uploaded_file,
            manual_path,
            explorer if explorer is not None else manual_path,
            min_speakers,
            max_speakers,
            output_root,
            threads,
            chunk_size,
            diarize_model,
            language,
        ]

        add_button.click(
            fn=add_file_callback,
            inputs=add_inputs,
            outputs=[status_box, queue_table, summary_md, logs_box],
        )

        set_watch_button.click(
            fn=set_watch_folder_callback,
            inputs=[watch_folder],
            outputs=[status_box, queue_table, summary_md, logs_box],
        )

        rescan_button.click(
            fn=rescan_watch_callback,
            inputs=[min_speakers, max_speakers, output_root, threads, chunk_size, diarize_model, language],
            outputs=[status_box, queue_table, summary_md, logs_box],
        )

        start_button.click(
            fn=start_queue_callback,
            inputs=[hf_token],
            outputs=[status_box, queue_table, summary_md, logs_box],
        )

        stop_button.click(
            fn=stop_after_current_callback,
            inputs=[],
            outputs=[status_box, queue_table, summary_md, logs_box],
        )

        clear_button.click(
            fn=clear_pending_callback,
            inputs=[],
            outputs=[status_box, queue_table, summary_md, logs_box],
        )

        kill_button.click(
            fn=kill_all_callback,
            inputs=[],
            outputs=[status_box, queue_table, summary_md, logs_box],
        )

        demo.load(fn=refresh_callback, inputs=[], outputs=[queue_table, summary_md, logs_box])
        if hasattr(gr, "Timer"):
            timer = gr.Timer(3.0)
            timer.tick(fn=refresh_callback, inputs=[], outputs=[queue_table, summary_md, logs_box])

    return demo


app = build_ui()


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_launch_port(host: str, start_port: int, span: int = 20) -> int:
    for candidate in range(start_port, start_port + span + 1):
        if _is_port_free(host, candidate):
            return candidate
    raise OSError(f"No free port found in range {start_port}-{start_port + span}.")


def launch_ui(
    preferred_port: Optional[int] = None,
    inbrowser: bool = False,
    prevent_thread_lock: bool = False,
) -> int:
    if preferred_port is None:
        preferred_port = int(os.getenv("GRADIO_SERVER_PORT", str(config.gradio_port)))

    launch_port = _pick_launch_port(config.gradio_host, int(preferred_port), span=20)
    if launch_port != preferred_port:
        logger.warning(
            "Preferred port %s is busy, using fallback port %s.",
            preferred_port,
            launch_port,
        )
    logger.info("Starting WhisperX Night Queue UI on http://%s:%s", config.gradio_host, launch_port)
    app.queue()
    app.launch(
        server_name=config.gradio_host,
        server_port=launch_port,
        inbrowser=inbrowser,
        show_error=True,
        prevent_thread_lock=prevent_thread_lock,
    )
    return launch_port


if __name__ == "__main__":
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", str(config.gradio_port)))
    launch_ui(preferred_port=preferred_port, inbrowser=False, prevent_thread_lock=False)
