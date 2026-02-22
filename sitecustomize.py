from __future__ import annotations

import os
import shutil
import threading


def _patch_huggingface_hub_for_windows() -> None:
    if os.name != "nt":
        return

    try:
        from huggingface_hub import _snapshot_download as snapshot_module
        from huggingface_hub import file_download as file_download_module
        import huggingface_hub
    except Exception:
        return

    if not getattr(file_download_module, "_whisperx_symlink_patch_applied", False):
        are_lock = threading.Lock()
        original_are_symlinks_supported = file_download_module.are_symlinks_supported
        original_create_symlink = file_download_module._create_symlink

        def locked_are_symlinks_supported(cache_dir=None):  # type: ignore[no-untyped-def]
            # Avoid a race where parallel download threads can read a temporary True.
            with are_lock:
                return original_are_symlinks_supported(cache_dir)

        def safe_create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
            try:
                original_create_symlink(src=src, dst=dst, new_blob=new_blob)
                return
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise

            abs_src = os.path.abspath(os.path.expanduser(src))
            abs_dst = os.path.abspath(os.path.expanduser(dst))
            os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
            try:
                os.remove(abs_dst)
            except OSError:
                pass
            if new_blob:
                shutil.move(abs_src, abs_dst, copy_function=file_download_module._copy_no_matter_what)
            else:
                shutil.copyfile(abs_src, abs_dst)

        file_download_module.are_symlinks_supported = locked_are_symlinks_supported
        file_download_module._create_symlink = safe_create_symlink
        file_download_module._whisperx_symlink_patch_applied = True

    if not getattr(snapshot_module.snapshot_download, "_whisperx_single_thread_patch", False):
        original_snapshot_download = snapshot_module.snapshot_download

        def patched_snapshot_download(*args, **kwargs):  # type: ignore[no-untyped-def]
            # Keep Windows downloads deterministic and avoid symlink support races.
            kwargs.setdefault("max_workers", 1)
            return original_snapshot_download(*args, **kwargs)

        patched_snapshot_download._whisperx_single_thread_patch = True  # type: ignore[attr-defined]
        snapshot_module.snapshot_download = patched_snapshot_download
        huggingface_hub.snapshot_download = patched_snapshot_download


_patch_huggingface_hub_for_windows()


def _patch_torch_serialization_defaults() -> None:
    # PyTorch 2.6 changed torch.load default to weights_only=True. pyannote checkpoints
    # still expect full pickle loading in trusted local workflows like this app.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


_patch_torch_serialization_defaults()
