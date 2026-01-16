import asyncio
import platform
import re
from typing import Any, Dict, List, Optional

import numpy as np
import sounddevice as sd

# Optional: filter common virtual/aggregate devices (not selected by default).
_VIRTUAL_PATTERNS = [
    r"blackhole",
    r"aggregate",
    r"multi[-\s]?output",  # macOS
    r"monitor",
    r"echo[-\s]?cancel",  # Linux Pulse/PipeWire
    r"vb[-\s]?cable",
    r"voicemeeter",
    r"cable (input|output)",  # Windows
    r"loopback",
]


def _is_virtual(name: str) -> bool:
    n = name.casefold()
    return any(re.search(pat, n) for pat in _VIRTUAL_PATTERNS)


def downmix_to_mono(
    pcm: np.ndarray | bytes,
    *,
    keepdims: bool = True,
    dtype: np.dtype | str = np.int16,
    in_channels: int | None = None,
) -> np.ndarray | bytes:
    """Downmix audio of any format to mono.

    Supports two input types:
    1. np.ndarray: PCM arrays shaped (N,) or (N, C)
    2. bytes: PCM byte stream (dtype and in_channels required)

    Args:
        pcm: Input audio data (ndarray or bytes)
        keepdims: True returns (N,1), False returns (N,) (ndarray only)
        dtype: PCM data type (bytes input only)
        in_channels: Input channel count (required for bytes input)

    Returns:
        Mono audio data (same type as input)

    Examples:
        >>> # ndarray input
        >>> stereo = np.random.randint(-32768, 32767, (1000, 2), dtype=np.int16)
        >>> mono = downmix_to_mono(stereo, keepdims=False)  # shape: (1000,)

        >>> # bytes input
        >>> stereo_bytes = b'...'  # Stereo PCM data
        >>> mono_bytes = downmix_to_mono(stereo_bytes, dtype=np.int16, in_channels=2)
    """
    # Bytes input: convert -> process -> convert back to bytes.
    if isinstance(pcm, bytes):
        if in_channels is None:
            raise ValueError("bytes input requires the in_channels parameter")
        arr = np.frombuffer(pcm, dtype=dtype).reshape(-1, in_channels)
        mono_arr = downmix_to_mono(arr, keepdims=False)  # bytes output ignores keepdims
        return mono_arr.tobytes()

    # ndarray input: process directly.
    x = np.asarray(pcm)
    if x.ndim == 1:
        return x[:, None] if keepdims else x

    # Already mono.
    if x.shape[1] == 1:
        return x if keepdims else x[:, 0]

    # Downmix multichannel.
    if np.issubdtype(x.dtype, np.integer):
        # Convert to float, average, then round back to avoid overflow.
        y = np.rint(x.astype(np.float32).mean(axis=1))
        info = np.iinfo(x.dtype)
        y = np.clip(y, info.min, info.max).astype(x.dtype)
    else:
        # Float: keep the original dtype (e.g., float32) instead of float64.
        y = x.mean(axis=1, dtype=x.dtype)

    return y[:, None] if keepdims else y


def safe_queue_put(
    queue: asyncio.Queue, item: Any, replace_oldest: bool = True
) -> bool:
    """Safely put an item into a queue, optionally dropping oldest data.

    Args:
        queue: asyncio.Queue instance
        item: Item to enqueue
        replace_oldest: True=drop oldest then enqueue, False=drop new item

    Returns:
        True=queued successfully, False=queue full and not enqueued
    """
    try:
        queue.put_nowait(item)
        return True
    except asyncio.QueueFull:
        if replace_oldest:
            try:
                queue.get_nowait()  # Drop oldest.
                queue.put_nowait(item)  # Enqueue new item.
                return True
            except asyncio.QueueEmpty:
                # Should not happen, but be safe.
                queue.put_nowait(item)
                return True
        return False


def upmix_mono_to_channels(mono_data: np.ndarray, num_channels: int) -> np.ndarray:
    """Upmix mono audio to multichannel (copy to all channels).

    Args:
        mono_data: Mono audio data, shape (N,)
        num_channels: Target channel count

    Returns:
        Multichannel audio data, shape (N, num_channels)
    """
    if num_channels == 1:
        return mono_data.reshape(-1, 1)

    # Copy mono data to all channels.
    return np.tile(mono_data.reshape(-1, 1), (1, num_channels))


def _valid(devs: List[dict], idx: int, kind: str, include_virtual: bool) -> bool:
    if not isinstance(idx, int) or idx < 0 or idx >= len(devs):
        return False
    d = devs[idx]
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    if int(d.get(key, 0)) <= 0:
        return False
    if not include_virtual and _is_virtual(d.get("name", "")):
        return False
    return True


def select_audio_device(
    kind: str,
    *,
    include_virtual: bool = False,
    allow_name_hints: Optional[bool] = None,  # None=Linux only; True/False to force
) -> Optional[Dict[str, Any]]:
    """
    Select an audio device: HostAPI default → (optional name hints on Linux) →
    sounddevice system default → first available. Returns {index, name,
    sample_rate, channels} or None.
    """
    assert kind in ("input", "output")
    system = platform.system().lower()

    # HostAPI preference order.
    if system == "windows":
        host_order = ["wasapi", "wdm-ks", "directsound", "mme"]
    elif system == "darwin":
        host_order = ["core audio"]
    else:
        host_order = ["alsa", "jack", "oss"]  # Most Linux PortAudio uses ALSA.

    # Enable name hints by default only on Linux (override via arg).
    if allow_name_hints is None:
        allow_name_hints = system == "linux"

    DEVICE_NAME_HINTS = {
        "input": ["default", "sysdefault", "pulse", "pipewire"],
        "output": ["default", "sysdefault", "dmix", "pulse", "pipewire"],
    }

    # Enumerate devices.
    try:
        hostapis = list(sd.query_hostapis())
        devices = list(sd.query_devices())
    except Exception:
        hostapis, devices = [], []

    key_host_default = (
        "default_input_device" if kind == "input" else "default_output_device"
    )
    key_channels = "max_input_channels" if kind == "input" else "max_output_channels"

    def pack(idx: int, base: Optional[dict] = None) -> Optional[Dict[str, Any]]:
        if base is None:
            if not _valid(devices, idx, kind, include_virtual):
                return None
            d = devices[idx]
        else:
            d = base
            if not include_virtual and _is_virtual(d.get("name", "")):
                return None
        sr = d.get("default_samplerate", None)
        return {
            "index": int(d.get("index", idx)),
            "name": d.get("name", "Unknown"),
            "sample_rate": int(sr) if isinstance(sr, (int, float)) else None,
            "channels": int(d.get(key_channels, 0)),
        }

    # 1) Match by HostAPI name (case-insensitive) → use its default device.
    for token in host_order:
        t = token.casefold()
        for ha in hostapis:
            if t in str(ha.get("name", "")).casefold():
                idx = ha.get(key_host_default, -1)
                info = pack(idx)
                if info:
                    return info

    # 1.5) Optional name hints when allow_name_hints=True (default on Linux).
    if allow_name_hints and devices:
        hints = [h.casefold() for h in DEVICE_NAME_HINTS[kind]]
        cands: List[int] = []
        for i, d in enumerate(devices):
            if not _valid(devices, i, kind, include_virtual):
                continue
            name_low = str(d.get("name", "")).casefold()
            if any(h in name_low for h in hints):
                cands.append(i)
        if cands:
            cands.sort()  # Stable: lowest index first.
            info = pack(cands[0])
            if info:
                return info

    # 2) sounddevice system default (includes platform routing).
    try:
        info = sd.query_devices(
            kind=kind
        )  # dict with index / default_samplerate / max_*_channels
        packed = pack(int(info.get("index")), base=info)
        if packed:
            return packed
    except Exception:
        pass

    # 3) Fallback: first available (and non-virtual unless allowed).
    for i, d in enumerate(devices):
        if _valid(devices, i, kind, include_virtual):
            return pack(i)

    return None
