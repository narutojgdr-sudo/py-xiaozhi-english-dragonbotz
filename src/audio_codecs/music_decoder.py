import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from src.constants.constants import AudioConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MusicDecoder:

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._process: Optional[subprocess.Process] = None
        self._decode_task: Optional[asyncio.Task] = None
        self._stopped = False

    async def start_decode(
        self, file_path: Path, output_queue: asyncio.Queue, start_position: float = 0.0
    ) -> bool:
        if not file_path.exists():
            logger.error(f"Audio file not found: {file_path}")
            return False

        self._stopped = False

        try:
            try:
                result = await asyncio.create_subprocess_exec(
                    "ffmpeg",
                    "-version",
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                await result.wait()
            except FileNotFoundError:
                logger.error("FFmpeg is not installed or not in PATH.")
                return False

            cmd = ["ffmpeg"]

            if start_position > 0:
                cmd.extend(["-ss", str(start_position)])

            cmd.extend(
                [
                    "-i",
                    str(file_path),  # Input file
                    "-f",
                    "s16le",  # Output format: 16-bit little-endian PCM
                    "-ar",
                    str(self.sample_rate),  # Sample rate
                    "-ac",
                    str(self.channels),  # Channel count
                    "-loglevel",
                    "error",  # Errors only
                    "-",  # Output to stdout
                ]
            )

            self._process = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Start read task.
            self._decode_task = asyncio.create_task(self._read_pcm_stream(output_queue))

            position_info = f" from {start_position:.1f}s" if start_position > 0 else ""
            logger.info(
                f"Starting audio decode: {file_path.name}{position_info} "
                f"[{self.sample_rate}Hz, {self.channels}ch]"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start audio decode: {e}")
            return False

    async def _read_pcm_stream(self, output_queue: asyncio.Queue):
        """
        Read PCM stream and enqueue it with queue- and time-based throttling.
        """
        import time

        frame_duration_ms = AudioConfig.FRAME_DURATION
        frame_size_samples = int(self.sample_rate * (frame_duration_ms / 1000))
        frame_size_bytes = frame_size_samples * 2 * self.channels
        logger.info(
            "Decoder parameters: frame size=%s samples, %s bytes, %sms",
            frame_size_samples,
            frame_size_bytes,
            frame_duration_ms,
        )

        eof_reached = False
        frame_count = 0
        start_time = time.time()  # Record decode start time.

        try:
            while not self._stopped:
                # Read one frame.
                chunk = await self._process.stdout.read(frame_size_bytes)

                if not chunk:
                    # EOF - file decode complete.
                    duration_decoded = frame_count * frame_duration_ms / 1000
                    logger.info(
                        "Audio decode complete: %s frames, ~%.1f seconds",
                        frame_count,
                        duration_decoded,
                    )

                    if self._process and self._process.returncode is not None:
                        try:
                            stderr_output = await self._process.stderr.read()
                            if stderr_output:
                                logger.error(
                                    "FFmpeg error output: %s",
                                    stderr_output.decode("utf-8", errors="ignore"),
                                )
                        except Exception:
                            pass

                    eof_reached = True
                    break

                frame_count += 1

                audio_array = np.frombuffer(chunk, dtype=np.int16)

                if self.channels > 1:
                    audio_array = audio_array.reshape(-1, self.channels)

                # ========== Dual throttling strategy ==========

                # Strategy 1: queue-occupancy-based throttling.
                queue_ratio = (
                    output_queue.qsize() / output_queue.maxsize
                    if output_queue.maxsize > 0
                    else 0
                )

                if queue_ratio < 0.3:
                    # Queue below 30%, fill quickly.
                    queue_based_sleep = 0
                elif queue_ratio < 0.7:
                    # Queue 30-70%, moderate throttling.
                    queue_based_sleep = 0.03
                else:
                    # Queue 70%+, heavier throttling.
                    queue_based_sleep = 0.06

                # Strategy 2: time-based throttle to avoid running faster than playback.
                expected_elapsed = frame_count * (frame_duration_ms / 1000.0)
                actual_elapsed = time.time() - start_time

                if actual_elapsed < expected_elapsed:
                    # Decode speed faster than playback, wait.
                    time_based_sleep = expected_elapsed - actual_elapsed
                else:
                    time_based_sleep = 0

                # Use the max to avoid overfilling or outrunning playback.
                target_sleep = max(queue_based_sleep, time_based_sleep)

                if target_sleep > 0:
                    await asyncio.sleep(target_sleep)

                # Enqueue with timeout protection.
                try:
                    await asyncio.wait_for(output_queue.put(audio_array), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Audio queue write timed out; skipping frame {frame_count}"
                    )
                    continue

        except asyncio.CancelledError:
            logger.debug("Decode task canceled.")
        except Exception as e:
            logger.error(f"Failed to read PCM stream: {e}")
        finally:
            if eof_reached:
                try:
                    await output_queue.put(None)
                except Exception:
                    pass

    async def stop(self):
        if self._stopped:
            return

        self._stopped = True
        logger.debug("Stopping audio decoder.")

        if self._decode_task and not self._decode_task.done():
            self._decode_task.cancel()
            try:
                await self._decode_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Force kill.
                try:
                    self._process.kill()
                    await self._process.wait()
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Failed to terminate FFmpeg process: {e}")

    def is_running(self) -> bool:
        return (
            not self._stopped
            and self._process is not None
            and self._process.returncode is None
        )

    async def wait_completion(self):
        if self._decode_task and not self._decode_task.done():
            try:
                await self._decode_task
            except Exception:
                pass
