import asyncio
import gc
from collections import deque
from typing import Callable, List, Optional, Protocol

import numpy as np
import opuslib
import sounddevice as sd
import soxr

from src.audio_codecs.aec_processor import AECProcessor
from src.constants.constants import AudioConfig
from src.utils.audio_utils import (
    downmix_to_mono,
    safe_queue_put,
    select_audio_device,
    upmix_mono_to_channels,
)
from src.utils.config_manager import ConfigManager
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AudioListener(Protocol):
    """
    Audio listener protocol (for wake-word detection, etc.).
    """

    def on_audio_data(self, audio_data: np.ndarray) -> None:
        """
        Callback for receiving audio data.
        """
        ...


class AudioCodec:
    """
    Audio codec - format adapter and encode/decode pipeline.

    Core responsibilities:
    1. Device management: device selection, stream creation, error recovery
    2. Format conversion: resampling, channel conversion, frame regrouping
    3. Codec: PCM ↔ Opus
    4. Stream buffering: avoid overflow and stutter

    Design principles:
    - Device layer: create streams using native device capabilities
    - Conversion layer: adapt device format to server protocol
    - Decoupling: callbacks and observer pattern for external dependencies

    Conversion flow:
    - Input: device native → downmix+resample → 16kHz mono → Opus encode → callback
    - Output: Opus → decode 24kHz mono → resample+upmix → device playback
    """

    def __init__(self, audio_processor: Optional[AECProcessor] = None):
        """Initialize the audio codec.

        Args:
            audio_processor: Optional audio processor (AEC, etc.) via DI
        """
        # Get configuration manager.
        self.config = ConfigManager.get_instance()

        # Opus codecs.
        self.opus_encoder = None
        self.opus_decoder = None

        # Native device info.
        self.device_input_sample_rate = None
        self.device_output_sample_rate = None
        self.input_channels = None
        self.output_channels = None
        self.mic_device_id = None
        self.speaker_device_id = None

        # Resamplers (created on demand).
        self.input_resampler = None
        self.output_resampler = None

        # Resample buffers.
        self._resample_input_buffer = deque()
        self._resample_output_buffer = deque()

        # Conversion flags.
        self._need_input_downmix = False
        self._need_output_upmix = False

        # Device frame sizes.
        self._device_input_frame_size = None
        self._device_output_frame_size = None

        # Audio stream objects.
        self.input_stream = None
        self.output_stream = None

        # Playback queue.
        self._output_buffer = asyncio.Queue(maxsize=500)

        # Callbacks and listeners (decouple external dependencies).
        self._encoded_callback: Optional[Callable] = None
        self._audio_listeners: List[AudioListener] = []

        # Audio processor (optional injection).
        self.audio_processor = audio_processor
        self._aec_enabled = False

        # Status flags.
        self._is_closing = False

    async def initialize(self):
        """Initialize audio devices and codecs.

        Strategy:
        1. First run: auto-select best devices and save config
        2. Subsequent runs: load device info from config
        3. Create streams using native device capabilities
        4. Auto-create converters (sample rate, channels)
        """
        try:
            # Load or initialize device configuration.
            await self._load_device_config()

            # Create Opus codecs.
            await self._create_opus_codecs()

            # Create resamplers and conversion flags.
            await self._create_resamplers()

            # Create audio streams using native device format.
            await self._create_streams()

            # Initialize AEC processor (if provided).
            if self.audio_processor:
                try:
                    await self.audio_processor.initialize()
                    self._aec_enabled = self.audio_processor._is_initialized
                    logger.info(
                        "AEC processor initialized: %s",
                        "enabled" if self._aec_enabled else "disabled",
                    )
                except Exception as e:
                    logger.warning(f"AEC processor initialization failed: {e}")
                    self._aec_enabled = False

            logger.info("AudioCodec initialization complete.")

        except Exception as e:
            logger.error(f"Failed to initialize audio devices: {e}")
            await self.close()
            raise

    async def _load_device_config(self):
        """
        Load or initialize device configuration.
        """
        audio_config = self.config.get_config("AUDIO_DEVICES", {}) or {}

        input_device_id = audio_config.get("input_device_id")
        output_device_id = audio_config.get("output_device_id")

        # First run: auto-select devices.
        if input_device_id is None or output_device_id is None:
            logger.info("First run: auto-selecting audio devices...")
            await self._auto_detect_devices()
            return

        # Load from config.
        self.mic_device_id = input_device_id
        self.speaker_device_id = output_device_id
        self.device_input_sample_rate = audio_config.get(
            "input_sample_rate", AudioConfig.INPUT_SAMPLE_RATE
        )
        self.device_output_sample_rate = audio_config.get(
            "output_sample_rate", AudioConfig.OUTPUT_SAMPLE_RATE
        )
        self.input_channels = audio_config.get("input_channels", 1)
        self.output_channels = audio_config.get("output_channels", 1)

        # Compute device frame sizes.
        self._device_input_frame_size = int(
            self.device_input_sample_rate * (AudioConfig.FRAME_DURATION / 1000)
        )
        self._device_output_frame_size = int(
            self.device_output_sample_rate * (AudioConfig.FRAME_DURATION / 1000)
        )

        logger.info(
            "Loaded device config | Input: %sHz %sch | Output: %sHz %sch",
            self.device_input_sample_rate,
            self.input_channels,
            self.device_output_sample_rate,
            self.output_channels,
        )

    async def _auto_detect_devices(self):
        """
        Auto-detect and select the best devices.
        """
        # Use smart device selection.
        in_info = select_audio_device("input", include_virtual=False)
        out_info = select_audio_device("output", include_virtual=False)

        if not in_info or not out_info:
            raise RuntimeError("No available audio devices found.")

        # Limit channel counts (avoid 100+ channel devices).
        raw_input_channels = in_info["channels"]
        raw_output_channels = out_info["channels"]

        self.input_channels = min(raw_input_channels, AudioConfig.MAX_INPUT_CHANNELS)
        self.output_channels = min(raw_output_channels, AudioConfig.MAX_OUTPUT_CHANNELS)

        # Record device info.
        self.mic_device_id = in_info["index"]
        self.speaker_device_id = out_info["index"]
        self.device_input_sample_rate = in_info["sample_rate"]
        self.device_output_sample_rate = out_info["sample_rate"]

        # Compute frame sizes.
        self._device_input_frame_size = int(
            self.device_input_sample_rate * (AudioConfig.FRAME_DURATION / 1000)
        )
        self._device_output_frame_size = int(
            self.device_output_sample_rate * (AudioConfig.FRAME_DURATION / 1000)
        )

        # Logging.
        if raw_input_channels > AudioConfig.MAX_INPUT_CHANNELS:
            logger.info(
                "Input device supports %s channels; using first %s channels",
                raw_input_channels,
                self.input_channels,
            )
        if raw_output_channels > AudioConfig.MAX_OUTPUT_CHANNELS:
            logger.info(
                "Output device supports %s channels; using first %s channels",
                raw_output_channels,
                self.output_channels,
            )

        logger.info(
            "Selected input device: %s (%sHz, %sch)",
            in_info["name"],
            self.device_input_sample_rate,
            self.input_channels,
        )
        logger.info(
            "Selected output device: %s (%sHz, %sch)",
            out_info["name"],
            self.device_output_sample_rate,
            self.output_channels,
        )

        # Save config on first run.
        self.config.update_config("AUDIO_DEVICES.input_device_id", self.mic_device_id)
        self.config.update_config("AUDIO_DEVICES.input_device_name", in_info["name"])
        self.config.update_config(
            "AUDIO_DEVICES.input_sample_rate", self.device_input_sample_rate
        )
        self.config.update_config("AUDIO_DEVICES.input_channels", self.input_channels)

        self.config.update_config(
            "AUDIO_DEVICES.output_device_id", self.speaker_device_id
        )
        self.config.update_config("AUDIO_DEVICES.output_device_name", out_info["name"])
        self.config.update_config(
            "AUDIO_DEVICES.output_sample_rate", self.device_output_sample_rate
        )
        self.config.update_config("AUDIO_DEVICES.output_channels", self.output_channels)

    async def _create_opus_codecs(self):
        """
        Create Opus codecs.
        """
        try:
            # Input encoder: 16kHz mono.
            self.opus_encoder = opuslib.Encoder(
                AudioConfig.INPUT_SAMPLE_RATE,
                AudioConfig.CHANNELS,
                opuslib.APPLICATION_VOIP,
            )

            # Output decoder: 24kHz mono.
            self.opus_decoder = opuslib.Decoder(
                AudioConfig.OUTPUT_SAMPLE_RATE, AudioConfig.CHANNELS
            )

            logger.info("Opus codecs created successfully.")
        except Exception as e:
            logger.error(f"Failed to create Opus codecs: {e}")
            raise

    async def _create_resamplers(self):
        """
        Create resamplers and conversion flags based on device/server differences.
        """
        # Input conversion configuration.
        # 1. Downmix flag.
        self._need_input_downmix = self.input_channels > 1
        if self._need_input_downmix:
            logger.info(f"Input downmix: {self.input_channels}ch → 1ch")

        # 2. Sample rate resampler.
        if self.device_input_sample_rate != AudioConfig.INPUT_SAMPLE_RATE:
            self.input_resampler = soxr.ResampleStream(
                self.device_input_sample_rate,
                AudioConfig.INPUT_SAMPLE_RATE,
                num_channels=1,  # Resampler handles mono after downmix.
                dtype="float32",
                quality="QQ",  # Fast quality (low latency).
            )
            logger.info(f"Input resample: {self.device_input_sample_rate}Hz → 16kHz")

        # Output conversion configuration.
        # 1. Sample rate resampler.
        if self.device_output_sample_rate != AudioConfig.OUTPUT_SAMPLE_RATE:
            self.output_resampler = soxr.ResampleStream(
                AudioConfig.OUTPUT_SAMPLE_RATE,
                self.device_output_sample_rate,
                num_channels=1,  # Resampler handles mono before upmix.
                dtype="float32",
                quality="QQ",
            )
            logger.info(
                "Output resample: %sHz → %sHz",
                AudioConfig.OUTPUT_SAMPLE_RATE,
                self.device_output_sample_rate,
            )

        # 2. Upmix flag.
        self._need_output_upmix = self.output_channels > 1
        if self._need_output_upmix:
            logger.info(f"Output upmix: 1ch → {self.output_channels}ch")

    async def _create_streams(self):
        """
        Create audio streams using native device format.
        """
        try:
            # Input stream: native sample rate and channels.
            self.input_stream = sd.InputStream(
                device=self.mic_device_id,
                samplerate=self.device_input_sample_rate,  # Native sample rate.
                channels=self.input_channels,  # Native channel count.
                dtype=np.float32,
                blocksize=self._device_input_frame_size,  # Native frame size.
                callback=self._input_callback,
                finished_callback=self._input_finished_callback,
                latency="low",
            )

            # Output stream: native sample rate and channels.
            self.output_stream = sd.OutputStream(
                device=self.speaker_device_id,
                samplerate=self.device_output_sample_rate,  # Native sample rate.
                channels=self.output_channels,  # Native channel count.
                dtype=np.float32,
                blocksize=self._device_output_frame_size,  # Native frame size.
                callback=self._output_callback,
                finished_callback=self._output_finished_callback,
                latency="low",
            )

            self.input_stream.start()
            self.output_stream.start()

            logger.info(
                "Audio streams started | Input: %sHz %sch | Output: %sHz %sch",
                self.device_input_sample_rate,
                self.input_channels,
                self.device_output_sample_rate,
                self.output_channels,
            )

        except Exception as e:
            logger.error(f"Failed to create audio streams: {e}")
            raise

    def _input_callback(self, indata, frames, time_info, status):
        """
        Input callback: native device → server protocol.
        Flow: multichannel/high-rate → downmix+resample → 16kHz mono → Opus encode.
        """
        if status and "overflow" not in str(status).lower():
            logger.warning(f"Input stream status: {status}")

        if self._is_closing:
            return

        try:
            # Step 1: downmix channels (stereo/multichannel → mono).
            if self._need_input_downmix:
                # indata shape: (frames, channels)
                audio_data = downmix_to_mono(indata, keepdims=False)
            else:
                audio_data = indata.flatten()  # Already mono.

            # Step 2: resample (device rate → 16kHz).
            if self.input_resampler is not None:
                audio_data = self._process_input_resampling(audio_data)
                if audio_data is None:  # Not enough data, wait for next frame.
                    return

            # Step 3: validate frame size.
            if len(audio_data) != AudioConfig.INPUT_FRAME_SIZE:
                return

            # Step 4: convert to int16 for Opus encoding and AEC.
            audio_data_int16 = (audio_data * 32768.0).astype(np.int16)

            # Step 5: AEC processing (if enabled).
            if self._aec_enabled and self.audio_processor._is_macos:
                try:
                    audio_data_int16 = self.audio_processor.process_audio(audio_data_int16)
                except Exception as e:
                    logger.warning(f"AEC processing failed; using raw audio: {e}")

            # Step 6: Opus encode and send in real-time.
            if self._encoded_callback:
                try:
                    pcm_data = audio_data_int16.tobytes()
                    encoded_data = self.opus_encoder.encode(
                        pcm_data, AudioConfig.INPUT_FRAME_SIZE
                    )
                    if encoded_data:
                        self._encoded_callback(encoded_data)
                except Exception as e:
                    logger.warning(f"Realtime recording encode failed: {e}")

            # Step 7: notify audio listeners (decouple wake-word detection).
            for listener in self._audio_listeners:
                try:
                    listener.on_audio_data(audio_data_int16.copy())
                except Exception as e:
                    logger.warning(f"Audio listener processing failed: {e}")

        except Exception as e:
            logger.error(f"Input callback error: {e}")

    def _process_input_resampling(self, audio_data):
        """
        Input resampling: device rate → 16kHz. Buffer until a full frame.
        """
        try:
            resampled_data = self.input_resampler.resample_chunk(audio_data, last=False)
            if len(resampled_data) > 0:
                self._resample_input_buffer.extend(resampled_data)

            # Accumulate to target frame size.
            expected_frame_size = AudioConfig.INPUT_FRAME_SIZE
            if len(self._resample_input_buffer) < expected_frame_size:
                return None

            # Pop one frame.
            frame_data = []
            for _ in range(expected_frame_size):
                frame_data.append(self._resample_input_buffer.popleft())

            return np.array(frame_data, dtype=np.float32)

        except Exception as e:
            logger.error(f"Input resampling failed: {e}")
            return None

    def _output_callback(self, outdata, frames, time_info, status):
        """
        Output callback: server protocol → device native format.
        Flow: 24kHz mono → resample+upmix → multichannel/high-rate.
        """
        if status:
            if "underflow" not in str(status).lower():
                logger.warning(f"Output stream status: {status}")

        try:
            # Get decoded 24kHz mono data.
            if self.output_resampler is not None:
                # Resample: 24kHz → device sample rate.
                self._output_callback_with_resample(outdata, frames)
            else:
                # Direct playback at 24kHz.
                self._output_callback_direct(outdata, frames)

        except Exception as e:
            logger.error(f"Output callback error: {e}")
            outdata.fill(0)

    def _output_callback_direct(self, outdata, frames):
        """Direct playback (when device supports 24kHz).

        Flow:
        1. Pop mono data from queue (OUTPUT_FRAME_SIZE samples)
        2. Trim or pad to required frames
        3. Convert int16 → float32
        4. Upmix if needed; otherwise output mono
        """
        try:
            # Get audio data from queue (mono int16).
            audio_data = self._output_buffer.get_nowait()

            # audio_data is mono, usually OUTPUT_FRAME_SIZE length.
            # Trim or pad to required frames.
            if len(audio_data) >= frames:
                mono_samples = audio_data[:frames]
            else:
                # Pad with silence.
                mono_samples = np.zeros(frames, dtype=np.int16)
                mono_samples[: len(audio_data)] = audio_data

            # Convert to float32 for playback.
            mono_samples_float = mono_samples.astype(np.float32) / 32768.0

            # Channel handling.
            if self._need_output_upmix:
                # Mono → multichannel (copy to all channels).
                multi_channel = upmix_mono_to_channels(
                    mono_samples_float, self.output_channels
                )
                outdata[:] = multi_channel
            else:
                # Mono output.
                outdata[:, 0] = mono_samples_float

        except asyncio.QueueEmpty:
            # Output silence when no data.
            outdata.fill(0)

    def _output_callback_with_resample(self, outdata, frames):
        """Resampled playback (24kHz → device sample rate).

        Flow:
        1. Pop 24kHz mono int16 data from queue
        2. Convert to float32 and resample to device rate (mono)
        3. Accumulate to buffer until enough frames
        4. Upmix if needed; otherwise output mono
        """
        try:
            # Continuously resample 24kHz mono data.
            # Buffer holds mono data, so compare frames not frames*channels.
            while len(self._resample_output_buffer) < frames:
                try:
                    audio_data = self._output_buffer.get_nowait()
                    # Convert int16 → float32.
                    audio_data_float = audio_data.astype(np.float32) / 32768.0
                    # Resample 24kHz mono → device rate mono.
                    resampled_data = self.output_resampler.resample_chunk(
                        audio_data_float, last=False
                    )
                    if len(resampled_data) > 0:
                        self._resample_output_buffer.extend(resampled_data)
                except asyncio.QueueEmpty:
                    break

            # Pop required mono frames.
            if len(self._resample_output_buffer) >= frames:
                frame_data = []
                for _ in range(frames):
                    frame_data.append(self._resample_output_buffer.popleft())
                mono_data = np.array(frame_data, dtype=np.float32)

                # Channel handling.
                if self._need_output_upmix:
                    # Mono → multichannel (copy to all channels).
                    multi_channel = upmix_mono_to_channels(
                        mono_data, self.output_channels
                    )
                    outdata[:] = multi_channel
                else:
                    # Mono output.
                    outdata[:, 0] = mono_data
            else:
                # Output silence when insufficient data.
                outdata.fill(0)

        except Exception as e:
            logger.warning(f"Resampled output failed: {e}")
            outdata.fill(0)

    def _input_finished_callback(self):
        """
        Input stream finished callback.
        """
        logger.info("Input stream finished.")

    def _output_finished_callback(self):
        """
        Output stream finished callback.
        """
        logger.info("Output stream finished.")

    # ============= Public interface methods =============

    def set_encoded_callback(self, callback: Callable[[bytes], None]):
        """Set encoded audio callback (decouple network layer).

        Args:
            callback: Callback for encoded Opus bytes

        Example:
            def network_send(opus_data: bytes):
                await websocket.send(opus_data)

            audio_codec.set_encoded_callback(network_send)
        """
        self._encoded_callback = callback

        if callback:
            logger.info("Encoded audio callback set.")
        else:
            logger.info("Encoded audio callback cleared.")

    def add_audio_listener(self, listener: AudioListener):
        """Add an audio listener (decouple wake-word detection, etc.).

        Args:
            listener: Listener implementing AudioListener

        Example:
            wake_word_detector = WakeWordDetector()  # Implements on_audio_data
            audio_codec.add_audio_listener(wake_word_detector)
        """
        if listener not in self._audio_listeners:
            self._audio_listeners.append(listener)
            logger.info(f"Added audio listener: {listener.__class__.__name__}")

    def remove_audio_listener(self, listener: AudioListener):
        """Remove an audio listener.

        Args:
            listener: Listener to remove
        """
        if listener in self._audio_listeners:
            self._audio_listeners.remove(listener)
            logger.info(f"Removed audio listener: {listener.__class__.__name__}")

    async def write_audio(self, opus_data: bytes):
        """Decode and play audio (server Opus data → speaker).

        Args:
            opus_data: Opus-encoded data from server

        Flow:
            Opus decode → 24kHz mono PCM → playback queue → output callback
        """
        try:
            # Decode Opus to 24kHz PCM.
            pcm_data = self.opus_decoder.decode(
                opus_data, AudioConfig.OUTPUT_FRAME_SIZE
            )

            audio_array = np.frombuffer(pcm_data, dtype=np.int16)

            expected_length = AudioConfig.OUTPUT_FRAME_SIZE * AudioConfig.CHANNELS
            if len(audio_array) != expected_length:
                logger.warning(
                    f"Unexpected decoded audio length: {len(audio_array)}, "
                    f"expected: {expected_length}"
                )
                return

            # Enqueue playback data (safe queue helper).
            if not safe_queue_put(
                self._output_buffer, audio_array, replace_oldest=True
            ):
                logger.warning("Playback queue full; dropping audio frame.")

        except opuslib.OpusError as e:
            logger.warning(f"Opus decode failed; dropping frame: {e}")
        except Exception as e:
            logger.warning(f"Audio write failed; dropping frame: {e}")

    async def write_pcm_direct(self, pcm_data: np.ndarray):
        """Write PCM data directly to the playback queue (MusicPlayer).

        Args:
            pcm_data: 24kHz mono PCM data (np.int16)

        Notes:
            This bypasses Opus decode and writes PCM directly.
            Mainly used for local music playback where FFmpeg already decoded.
        """
        try:
            # Validate data format.
            expected_length = AudioConfig.OUTPUT_FRAME_SIZE * AudioConfig.CHANNELS

            # Pad or trim when length mismatch.
            if len(pcm_data) != expected_length:
                if len(pcm_data) < expected_length:
                    # Pad with silence.
                    padded = np.zeros(expected_length, dtype=np.int16)
                    padded[: len(pcm_data)] = pcm_data
                    pcm_data = padded
                    logger.debug(
                        f"PCM data short; padded with silence: {len(pcm_data)} → "
                        f"{expected_length}"
                    )
                else:
                    # Trim excess data.
                    pcm_data = pcm_data[:expected_length]
                    logger.debug(
                        f"PCM data too long; trimmed: {len(pcm_data)} → "
                        f"{expected_length}"
                    )

            # Enqueue playback data (no replacement; wait).
            if not safe_queue_put(self._output_buffer, pcm_data, replace_oldest=False):
                # Wait when queue is full.
                await asyncio.wait_for(self._output_buffer.put(pcm_data), timeout=2.0)

        except asyncio.TimeoutError:
            logger.warning("Playback queue wait timed out; dropping PCM frame.")
        except Exception as e:
            logger.warning(f"Failed to write PCM data: {e}")

    async def reinitialize_stream(self, is_input: bool = True):
        """Rebuild audio stream (handle device errors/disconnect).

        Args:
            is_input: True=rebuild input stream, False=output stream

        Use cases:
            - Device hot-plug
            - Driver error recovery
            - System sleep/wake
        """
        if self._is_closing:
            return False if is_input else None

        try:
            if is_input:
                if self.input_stream:
                    self.input_stream.stop()
                    self.input_stream.close()

                self.input_stream = sd.InputStream(
                    device=self.mic_device_id,
                    samplerate=self.device_input_sample_rate,
                    channels=self.input_channels,
                    dtype=np.float32,
                    blocksize=self._device_input_frame_size,
                    callback=self._input_callback,
                    finished_callback=self._input_finished_callback,
                    latency="low",
                )
                self.input_stream.start()
                logger.info("Input stream reinitialized.")
                return True
            else:
                if self.output_stream:
                    self.output_stream.stop()
                    self.output_stream.close()

                self.output_stream = sd.OutputStream(
                    device=self.speaker_device_id,
                    samplerate=self.device_output_sample_rate,
                    channels=self.output_channels,
                    dtype=np.float32,
                    blocksize=self._device_output_frame_size,
                    callback=self._output_callback,
                    finished_callback=self._output_finished_callback,
                    latency="low",
                )
                self.output_stream.start()
                logger.info("Output stream reinitialized.")
                return None
        except Exception as e:
            stream_type = "input" if is_input else "output"
            logger.error(f"{stream_type} stream rebuild failed: {e}")
            if is_input:
                return False
            else:
                raise

    async def clear_audio_queue(self):
        """Clear audio queue.

        Use cases:
            - User interrupts playback
            - Wake word triggers interruption
            - Clear dirty data after errors
        """
        cleared_count = 0

        # Clear playback queue.
        while not self._output_buffer.empty():
            try:
                self._output_buffer.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break

        # Clear resample buffers.
        if self._resample_input_buffer:
            cleared_count += len(self._resample_input_buffer)
            self._resample_input_buffer.clear()

        if self._resample_output_buffer:
            cleared_count += len(self._resample_output_buffer)
            self._resample_output_buffer.clear()

        if cleared_count > 0:
            logger.info(f"Cleared audio queue, dropped {cleared_count} frames.")

        if cleared_count > 100:
            gc.collect()
            logger.debug("Ran garbage collection to free memory.")

    # ============= AEC control methods =============

    async def _cleanup_resampler(self, resampler, name: str):
        """
        Clean up resampler resources.
        """
        if not resampler:
            return

        try:
            # Flush buffer.
            if hasattr(resampler, "resample_chunk"):
                empty_array = np.array([], dtype=np.float32)
                resampler.resample_chunk(empty_array, last=True)
        except Exception as e:
            logger.debug(f"Failed to flush {name} resampler buffer: {e}")

        try:
            # Attempt explicit close.
            if hasattr(resampler, "close"):
                resampler.close()
                logger.debug(f"{name} resampler closed.")
        except Exception as e:
            logger.debug(f"Failed to close {name} resampler: {e}")

    def _stop_stream_sync(self, stream, name: str):
        """
        Stop a single audio stream synchronously.
        """
        if not stream:
            return
        try:
            if stream.active:
                stream.stop()
            stream.close()
        except Exception as e:
            logger.warning(f"Failed to close {name} stream: {e}")

    async def close(self):
        """Close the audio codec and release resources.

        Cleanup order:
        1. Set closing flag, stop audio streams
        2. Clear callbacks and listener references
        3. Clear queues and buffers
        4. Close AEC processor
        5. Clean resamplers
        6. Release codecs
        7. Run garbage collection
        """
        if self._is_closing:
            return

        self._is_closing = True
        logger.info("Closing audio codec...")

        try:
            # 1. Stop audio streams.
            self._stop_stream_sync(self.input_stream, "input")
            self._stop_stream_sync(self.output_stream, "output")
            self.input_stream = None
            self.output_stream = None

            # Wait for callbacks to stop.
            await asyncio.sleep(0.05)

            # 2. Clear callbacks and listeners.
            self._encoded_callback = None
            self._audio_listeners.clear()

            # 3. Clear queues and buffers.
            await self.clear_audio_queue()

            # 4. Close AEC processor.
            if self.audio_processor:
                try:
                    await self.audio_processor.close()
                except Exception as e:
                    logger.warning(f"Failed to close AEC processor: {e}")
                finally:
                    self.audio_processor = None

            # 5. Clean resamplers.
            await self._cleanup_resampler(self.input_resampler, "input")
            await self._cleanup_resampler(self.output_resampler, "output")
            self.input_resampler = None
            self.output_resampler = None

            # 6. Release codecs.
            self.opus_encoder = None
            self.opus_decoder = None

            # 7. Garbage collection.
            gc.collect()

            logger.info("Audio resources fully released.")
        except Exception as e:
            logger.error(f"Error while closing audio codec: {e}", exc_info=True)
        finally:
            self._is_closing = True

    def __del__(self):
        """Destructor - check resources are properly released."""
        if not self._is_closing:
            logger.warning("AudioCodec not properly closed; call close().")
