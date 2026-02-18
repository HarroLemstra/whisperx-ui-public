from __future__ import annotations

import sys
from typing import List, Tuple


def resolve_whisperx_command() -> Tuple[List[str], str]:
    display = f"{sys.executable} -m whisperx"
    return [sys.executable, "-m", "whisperx"], display
