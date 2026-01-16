import platform
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import sounddevice as sd

from src.constants.constants import AudioConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AECProcessor:
    """
    Audio echo cancellation processor for reference (speaker) and mic input AEC.
    """

    def __init__(self):
        # Platform info.
        self._platform = platform.system().lower()
        self._is_macos = self._platform == "darwin"
        self._is_linux = self._platform == "linux"
        self._is_windows = self._platform == "windows"

        # WebRTC APM instance (macOS only).
        self.apm = None
        self.apm_config = None
        self.capture_config = None
        self.render_config = None

        # Reference signal stream (macOS only).
        self.reference_stream = None
        self.reference_device_id = None
        self.reference_sample_rate = None

        # Reference signal resampler (macOS only).
        self.reference_resampler = None
        self._resample_reference_buffer = deque()

        # Buffers.
        self._reference_buffer = deque()
        self._webrtc_frame_size = 160  # WebRTC: 16kHz, 10ms = 160 samples
        self._system_frame_size = AudioConfig.INPUT_FRAME_SIZE  # System frame size

        # Status flags.
        self._is_initialized = False
        self._is_closing = False

    async def initialize(self):
        """
        Initialize the AEC processor.
        """
        try:
            if self._is_windows or self._is_linux:
                # Windows and Linux use system-level AEC.
                logger.info(
                    f"{self._platform.capitalize()} uses system-level echo cancellation; "
                    "AEC processor enabled."
                )
                self._is_initialized = True
                return
            elif self._is_macos:
                # macOS uses WebRTC + BlackHole.
                await self._initialize_apm()
                await self._initialize_reference_capture()
            else:
                logger.warning(
                    f"Current platform {self._platform} does not support AEC."
                )
                self._is_initialized = True
                return

            self._is_initialized = True
            logger.info("AEC processor initialized.")

        except Exception as e:
            logger.error(f"AEC processor initialization failed: {e}")
            await self.close()
            raise

    async def _initialize_apm(self):
        """
        Initialize the WebRTC audio processing module (macOS only).
        """
        if not self._is_macos:
            logger.warning("Non-macOS platform called _initialize_apm; unexpected.")
            return

        try:
            # Lazy import local libs only when needed on macOS.
            from libs.webrtc_apm import WebRTCAudioProcessing, create_default_config

            self.apm = WebRTCAudioProcessing()

            # Create configuration.
            self.apm_config = create_default_config()

            # Enable echo cancellation.
            self.apm_config.echo.enabled = True
            self.apm_config.echo.mobile_mode = False
            self.apm_config.echo.enforce_high_pass_filtering = True

            # Enable noise suppression.
            self.apm_config.noise_suppress.enabled = True
            self.apm_config.noise_suppress.noise_level = 2  # HIGH

            # Enable high-pass filter.
            self.apm_config.high_pass.enabled = True
            self.apm_config.high_pass.apply_in_full_band = True

            # Apply configuration.
            result = self.apm.apply_config(self.apm_config)
            if result != 0:
                raise RuntimeError(f"WebRTC APM config failed, error code: {result}")

            # Create stream configuration.
            sample_rate = AudioConfig.INPUT_SAMPLE_RATE  # 16kHz
            channels = AudioConfig.CHANNELS  # 1

            self.capture_config = self.apm.create_stream_config(sample_rate, channels)
            self.render_config = self.apm.create_stream_config(sample_rate, channels)

            # Set stream delay.
            self.apm.set_stream_delay_ms(40)  # 50ms delay

            logger.info("WebRTC APM initialized.")

        except Exception as e:
            logger.error(f"WebRTC APM initialization failed: {e}")
            raise

    async def _initialize_reference_capture(self):
        """
        Initialize reference signal capture (macOS only).
        """
        if not self._is_macos:
            return

        try:
            # Find BlackHole 2ch device.
            reference_device = self._find_blackhole_device()
            if reference_device is None:
                logger.warning(
                    "BlackHole 2ch device not found; reference capture unavailable."
                )
                return

            self.reference_device_id = reference_device["id"]
            self.reference_sample_rate = int(reference_device["default_samplerate"])

            # Create high-quality resampler if needed.
            if self.reference_sample_rate != AudioConfig.INPUT_SAMPLE_RATE:
                import soxr

                self.reference_resampler = soxr.ResampleStream(
                    self.reference_sample_rate,
                    AudioConfig.INPUT_SAMPLE_RATE,
                    num_channels=1,
                    dtype="int16",
                    quality="QQ",  # Fast quality, same as AudioCodec
                )
                logger.info(
                    "Reference resample: %sHz → %sHz (soxr)",
                    self.reference_sample_rate,
                    AudioConfig.INPUT_SAMPLE_RATE,
                )

            # Create reference input stream (10ms frames, WebRTC standard).
            webrtc_frame_duration = 0.01  # 10ms, WebRTC standard frame length
            reference_frame_size = int(
                self.reference_sample_rate * webrtc_frame_duration
            )

            self.reference_stream = sd.InputStream(
                device=self.reference_device_id,
                samplerate=self.reference_sample_rate,
                channels=AudioConfig.CHANNELS,
                dtype=np.int16,
                blocksize=reference_frame_size,
                callback=self._reference_callback,
                finished_callback=self._reference_finished_callback,
                latency="low",
            )

            self.reference_stream.start()

            logger.info(
                "Reference capture started: [%s] %s",
                self.reference_device_id,
                reference_device["name"],
            )

        except Exception as e:
            logger.error(f"Reference capture initialization failed: {e}")
            # Do not raise; allow AEC to run without reference signal.

    def _find_blackhole_device(self) -> Optional[Dict[str, Any]]:
        """
        Find the BlackHole 2ch virtual device.
        """
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_name = device["name"].lower()
                # Look for BlackHole 2ch.
                if "blackhole" in device_name and "2ch" in device_name:
                    # Ensure it's an input device.
                    if device["max_input_channels"] >= 1:
                        device_info = dict(device)
                        device_info["id"] = i
                        logger.info(f"Found BlackHole device: [{i}] {device['name']}")
                        return device_info

            # If no BlackHole 2ch, try any BlackHole device.
            for i, device in enumerate(devices):
                device_name = device["name"].lower()
                if "blackhole" in device_name and device["max_input_channels"] >= 1:
                    device_info = dict(device)
                    device_info["id"] = i
                    logger.info(f"Found BlackHole device: [{i}] {device['name']}")
                    return device_info

            return None

        except Exception as e:
            logger.error(f"Failed to find BlackHole device: {e}")
            return None

    def _reference_callback(self, indata, frames, time_info, status):
        """
        Reference signal callback.
        """
        # frames/time_info unused but kept for sounddevice callback signature.
        _ = frames, time_info

        if status and "overflow" not in str(status).lower():
            logger.warning(f"Reference stream status: {status}")

        if self._is_closing:
            return

        try:
            audio_data = indata.copy().flatten()

            # Resample with soxr.
            if self.reference_resampler:
                # Resample to 16kHz.
                resampled_data = self.reference_resampler.resample_chunk(
                    audio_data, last=False
                )
                if len(resampled_data) > 0:
                    self._resample_reference_buffer.extend(
                        resampled_data.astype(np.int16)
                    )

                # Extract WebRTC frames from buffer.
                while len(self._resample_reference_buffer) >= self._webrtc_frame_size:
                    for _ in range(self._webrtc_frame_size):
                        self._reference_buffer.append(
                            self._resample_reference_buffer.popleft()
                        )
            else:
                # No resampling needed.
                self._reference_buffer.extend(audio_data)

            # Keep buffer size reasonable.
            max_buffer_size = self._webrtc_frame_size * 20  # ~200ms of data
            while len(self._reference_buffer) > max_buffer_size:
                self._reference_buffer.popleft()

        except Exception as e:
            logger.error(f"Reference callback error: {e}")

    def _reference_finished_callback(self):
        """
        Reference stream finished callback.
        """
        logger.info("Reference stream finished.")

    def process_audio(self, capture_audio: np.ndarray) -> np.ndarray:
        """Process audio frames with AEC support for 10/20/40/60ms frames.

        Args:
            capture_audio: Microphone audio data (16kHz, int16)

        Returns:
            Processed audio data
        """
        if not self._is_initialized:
            return capture_audio

        # Windows and Linux: return raw audio (system-level processing).
        if self._is_windows or self._is_linux:
            return capture_audio

        # macOS uses WebRTC AEC.
        if not self._is_macos or self.apm is None:
            return capture_audio

        try:
            # Ensure input frame size is a multiple of the WebRTC frame size.
            if len(capture_audio) % self._webrtc_frame_size != 0:
                logger.warning(
                    "Audio frame size is not a multiple of WebRTC frame size: "
                    f"{len(capture_audio)}, WebRTC frame: {self._webrtc_frame_size}"
                )
                return capture_audio

            # Calculate chunk count.
            num_chunks = len(capture_audio) // self._webrtc_frame_size

            if num_chunks == 1:
                # 10ms frame, process directly.
                return self._process_single_aec_frame(capture_audio)
            else:
                # 20/40/60ms frames, process in chunks.
                return self._process_chunked_aec_frames(capture_audio, num_chunks)

        except Exception as e:
            logger.error(f"AEC processing failed: {e}")
            return capture_audio

    def _process_single_aec_frame(self, capture_audio: np.ndarray) -> np.ndarray:
        """
        Process a single 10ms WebRTC frame (macOS only).
        """
        if not self._is_macos:
            return capture_audio

        try:
            # Import ctypes on macOS only.
            import ctypes

            # Get reference signal.
            reference_audio = self._get_reference_frame(self._webrtc_frame_size)

            # Create ctypes buffers.
            capture_buffer = (ctypes.c_short * self._webrtc_frame_size)(*capture_audio)
            reference_buffer = (ctypes.c_short * self._webrtc_frame_size)(
                *reference_audio
            )

            processed_capture = (ctypes.c_short * self._webrtc_frame_size)()
            processed_reference = (ctypes.c_short * self._webrtc_frame_size)()

            # Process reference signal first (render stream).
            render_result = self.apm.process_reverse_stream(
                reference_buffer,
                self.render_config,
                self.render_config,
                processed_reference,
            )

            if render_result != 0:
                logger.warning(
                    f"Reference signal processing failed, error code: {render_result}"
                )

            # Then process capture signal (capture stream).
            capture_result = self.apm.process_stream(
                capture_buffer,
                self.capture_config,
                self.capture_config,
                processed_capture,
            )

            if capture_result != 0:
                logger.warning(
                    f"Capture signal processing failed, error code: {capture_result}"
                )
                return capture_audio

            # Convert back to numpy array.
            return np.array(processed_capture, dtype=np.int16)

        except Exception as e:
            logger.error(f"AEC frame processing failed: {e}")
            return capture_audio

    def _process_chunked_aec_frames(
        self, capture_audio: np.ndarray, num_chunks: int
    ) -> np.ndarray:
        """
        Process large frames in chunks (20/40/60ms, etc.).
        """
        processed_chunks = []

        for i in range(num_chunks):
            # Extract current 10ms chunk.
            start_idx = i * self._webrtc_frame_size
            end_idx = (i + 1) * self._webrtc_frame_size
            chunk = capture_audio[start_idx:end_idx]

            # Process this 10ms chunk.
            processed_chunk = self._process_single_aec_frame(chunk)
            processed_chunks.append(processed_chunk)

        # Concatenate processed chunks.
        return np.concatenate(processed_chunks)

    def _get_reference_frame(self, frame_size: int) -> np.ndarray:
        """
        Get a reference signal frame of the specified size.
        """
        # Return silence if buffer is insufficient.
        if len(self._reference_buffer) < frame_size:
            return np.zeros(frame_size, dtype=np.int16)

        # Extract one frame from buffer.
        frame_data = []
        for _ in range(frame_size):
            frame_data.append(self._reference_buffer.popleft())

        return np.array(frame_data, dtype=np.int16)

    async def close(self):
        """
        Close the AEC processor.
        """
        if self._is_closing:
            return

        self._is_closing = True
        logger.info("Closing AEC processor...")

        try:
            # Clean WebRTC resources only on macOS.
            if self._is_macos:
                # Stop reference stream.
                if self.reference_stream:
                    try:
                        self.reference_stream.stop()
                        self.reference_stream.close()
                    except Exception as e:
                        logger.warning(f"Failed to close reference stream: {e}")
                    finally:
                        self.reference_stream = None

                # Clean resampler.
                if self.reference_resampler:
                    try:
                        # Flush resampler buffer.
                        empty_array = np.array([], dtype=np.int16)
                        self.reference_resampler.resample_chunk(empty_array, last=True)
                    except Exception as e:
                        logger.debug(
                            f"Failed to flush reference resampler buffer: {e}"
                        )
                    finally:
                        self.reference_resampler = None

                # Clean WebRTC APM.
                if self.apm:
                    try:
                        if self.capture_config:
                            self.apm.destroy_stream_config(self.capture_config)
                        if self.render_config:
                            self.apm.destroy_stream_config(self.render_config)
                    except Exception as e:
                        logger.warning(f"Failed to clean APM config: {e}")
                    finally:
                        self.capture_config = None
                        self.render_config = None
                        self.apm = None

            # Clear buffers.
            self._reference_buffer.clear()
            self._resample_reference_buffer.clear()

            self._is_initialized = False
            logger.info("AEC processor closed.")

        except Exception as e:
            logger.error(f"Error closing AEC processor: {e}")
