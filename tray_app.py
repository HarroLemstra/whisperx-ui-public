from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
import logging

from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem


MUTEX_NAME = r"Global\WhisperXNightQueueTray"
ERROR_ALREADY_EXISTS = 183

BASE_DIR = Path(__file__).resolve().parent
APP_HOME = Path(os.getenv("LOCALAPPDATA", str(BASE_DIR))) / "WhisperXNightQueue"
LOGS_DIR = APP_HOME / "logs"
DATA_DIR = APP_HOME / "data"
TRAY_LOG_PATH = LOGS_DIR / "tray.log"
TRAY_STATUS_PATH = DATA_DIR / "tray_status.json"

_logger = logging.getLogger("whisperx_tray")
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_std_streams() -> None:
    # pythonw can run without stdout/stderr, but uvicorn logging expects
    # stream objects with `isatty()`.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if _logger.handlers:
        return
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = RotatingFileHandler(TRAY_LOG_PATH, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)


def _write_status(
    *,
    state: str,
    pid: int,
    host: str,
    port: int,
    url: str,
    started_at: str | None,
    last_error: str | None,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "pid": int(pid),
        "url": url,
        "host": host,
        "port": int(port),
        "started_at": started_at,
        "updated_at": _now_iso(),
        "last_error": last_error,
    }
    temp_path = TRAY_STATUS_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(TRAY_STATUS_PATH)


def _mark_failed(last_error: str) -> None:
    try:
        _write_status(
            state="failed",
            pid=os.getpid(),
            host="",
            port=0,
            url="",
            started_at=None,
            last_error=last_error,
        )
    except Exception:
        _logger.exception("Failed to write failed tray status.")


def _install_exception_hooks() -> None:
    def _sys_hook(exc_type, exc, tb) -> None:
        _logger.exception("Unhandled exception in tray process.", exc_info=(exc_type, exc, tb))
        _mark_failed(str(exc))

    def _thread_hook(args) -> None:
        _logger.exception(
            "Unhandled exception in tray thread.",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        _mark_failed(str(args.exc_value))

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook


def _acquire_mutex() -> int | None:
    handle = _kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    last_error = ctypes.get_last_error()
    if last_error == ERROR_ALREADY_EXISTS:
        _kernel32.CloseHandle(handle)
        return None
    return int(handle)


def _release_mutex(handle: int | None) -> None:
    if handle:
        _kernel32.CloseHandle(handle)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            # Prefer existing process env vars; `.env` only fills gaps.
            if key not in os.environ:
                os.environ[key] = value

    if not os.environ.get("HF_TOKEN") and os.environ.get("HF_token"):
        os.environ["HF_TOKEN"] = os.environ["HF_token"]


def _create_tray_image(size: int = 64) -> Image.Image:
    image = Image.new("RGBA", (size, size), (20, 28, 38, 255))
    draw = ImageDraw.Draw(image)

    pad = 8
    draw.rounded_rectangle((pad, pad, size - pad, size - pad), radius=12, fill=(31, 111, 235, 255))
    draw.rounded_rectangle((pad + 8, pad + 10, size - pad - 8, size - pad - 10), radius=8, fill=(250, 250, 250, 255))
    draw.rectangle((size // 2 - 2, 18, size // 2 + 2, size - 18), fill=(31, 111, 235, 255))
    draw.rectangle((18, size // 2 - 2, size - 18, size // 2 + 2), fill=(31, 111, 235, 255))
    return image


def main() -> int:
    _ensure_std_streams()
    _setup_logging()
    _install_exception_hooks()
    _logger.info("Tray bootstrap started.")

    _load_dotenv(BASE_DIR / ".env")

    mutex_handle = None
    host = ""
    port = 0
    url = ""
    started_at: str | None = None

    try:
        mutex_handle = _acquire_mutex()
        if mutex_handle is None:
            _logger.info("Tray already running (mutex exists).")
            return 0

        _write_status(
            state="starting",
            pid=os.getpid(),
            host=host,
            port=port,
            url=url,
            started_at=started_at,
            last_error=None,
        )

        import app as web_app

        port = web_app.launch_ui(inbrowser=False, prevent_thread_lock=True)
        host = web_app.config.gradio_host
        url = f"http://{host}:{port}"
        started_at = _now_iso()

        _write_status(
            state="running",
            pid=os.getpid(),
            host=host,
            port=port,
            url=url,
            started_at=started_at,
            last_error=None,
        )
        _logger.info("Tray running at %s", url)

        def _open_dashboard(icon: Icon, item: MenuItem) -> None:
            del icon, item
            webbrowser.open(url)

        def _open_settings(icon: Icon, item: MenuItem) -> None:
            del icon, item
            webbrowser.open(f"{url}#instellingen")

        def _exit_app(icon: Icon, item: MenuItem) -> None:
            del item
            _logger.info("Tray exit requested by user.")
            try:
                _write_status(
                    state="stopping",
                    pid=os.getpid(),
                    host=host,
                    port=port,
                    url=url,
                    started_at=started_at,
                    last_error=None,
                )
                web_app.app.close()
            finally:
                _write_status(
                    state="stopped",
                    pid=os.getpid(),
                    host=host,
                    port=port,
                    url=url,
                    started_at=started_at,
                    last_error=None,
                )
                icon.stop()

        menu = Menu(
            MenuItem("Open Dashboard", _open_dashboard),
            MenuItem("Instellingen", _open_settings),
            MenuItem("Exit", _exit_app),
        )
        icon = Icon("whisperx_night_queue", _create_tray_image(), "WhisperX Night Queue", menu)
        icon.run()
        _logger.info("Tray icon loop ended.")
        return 0
    except Exception as exc:
        _logger.exception("Tray startup failed.")
        _write_status(
            state="failed",
            pid=os.getpid(),
            host=host,
            port=port,
            url=url,
            started_at=started_at,
            last_error=str(exc),
        )
        return 1
    finally:
        _release_mutex(mutex_handle)


if __name__ == "__main__":
    sys.exit(main())
