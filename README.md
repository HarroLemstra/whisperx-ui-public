# WhisperX Night Queue UI

Local Gradio UI for CPU-based WhisperX transcription + diarization with a night queue workflow.

## What it does

- Select files from disk (manual path, optional upload, optional file explorer).
- Queue multiple files and process sequentially overnight.
- Always converts input to 16kHz mono WAV before WhisperX.
- Runs WhisperX with diarization (`pyannote/speaker-diarization-3.1` by default).
- Retries failed jobs once, then logs and continues.
- Supports `Stop after current` and `Clear rest`.
- Optional watch-folder scan (manual + periodic polling).
- Writes `transcript.srt`, `transcript.txt`, `transcript.json`, `job.log`, `meta.json` per job.

## Directory layout

- `app.py`: Gradio app entrypoint.
- `core/`: queue, runner, preflight, config.
- `start_ui.ps1`: one-click startup script.
- `out/`: output folders.
- `logs/app.log`: app-level logs.
- `data/queue_state.json`: queue persistence.

## Prerequisites

- Windows PowerShell
- Python venv at `.venv`
- `ffmpeg` on `PATH`
- `whisperx` installed in `.venv`
- Hugging Face token with access to:
  - `pyannote/speaker-diarization-3.1`
  - `pyannote/segmentation-3.0`

## Token setup

Use `.env` in project root:

```env
HF_TOKEN=hf_xxx
```

`start_ui.ps1` also accepts legacy `HF_token` and maps it to `HF_TOKEN` for the session.

## Start

```powershell
cd C:\whisperx
.\.venv\Scripts\python.exe -m pip install -U gradio
.\start_ui.ps1
```

Open: `http://127.0.0.1:7860`

## Start as tray app (recommended)

```powershell
cd C:\whisperx
.\start_tray.ps1
```

- No persistent console window.
- Tray icon in system tray.
- Right click menu: `Open Dashboard`, `Instellingen`, `Exit`.
- Admin PowerShell is **not** required.
- Autostart on Windows logon is automatically registered by `start_tray.ps1`.
- If Task Scheduler is blocked, a Startup-folder fallback is created automatically.
- Tray runtime state/logs are stored in `%LOCALAPPDATA%\WhisperXNightQueue`.

## Troubleshooting (tray)

```powershell
.\start_tray.ps1
Get-Content "$env:LOCALAPPDATA\WhisperXNightQueue\logs\tray.log" -Tail 80
Get-Content "$env:LOCALAPPDATA\WhisperXNightQueue\data\tray_status.json"
```
## Note about `orjson` policy blocks

Some locked-down Windows environments block native `orjson` DLL loading. This project includes a local `orjson.py` fallback so Gradio can still run locally.

## Default runtime

- Model: `openai/whisper-large-v3-turbo`
- Device: `cpu`
- Compute type: `int8`
- Language: `nl`
- Segment resolution: `sentence`
- VAD: `silero`
- Speakers: min `2`, max `4` (UI adjustable)
- Retry policy: 1 retry per job
