import asyncio
import sys
import threading
from pathlib import Path
from typing import Any, Awaitable, Optional
import shutil
import subprocess
import numpy as _np
try:
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    from PyQt5.QtCore import QUrl
    _QT_MEDIA_AVAILABLE = True
except Exception:
    _QT_MEDIA_AVAILABLE = False
try:
    from playsound import playsound  # type: ignore
    _PLAYSOUND_AVAILABLE = True
except Exception:
    _PLAYSOUND_AVAILABLE = False

# 允许作为脚本直接运行：把项目根目录加入 sys.path（src 的上一级）
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    pass

from src.constants.constants import DeviceState, ListeningMode
from src.plugins.calendar import CalendarPlugin
from src.plugins.iot import IoTPlugin
from src.plugins.manager import PluginManager
from src.plugins.mcp import McpPlugin
from src.plugins.shortcuts import ShortcutsPlugin
from src.plugins.ui import UIPlugin
from src.plugins.wake_word import WakeWordPlugin
from src.protocols.mqtt_protocol import MqttProtocol
from src.protocols.websocket_protocol import WebsocketProtocol
from src.utils.config_manager import ConfigManager
from src.utils.logging_config import get_logger
from src.utils.opus_loader import setup_opus

logger = get_logger(__name__)
setup_opus()


class Application:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Application()
        return cls._instance

    def __init__(self):
        if Application._instance is not None:
            logger.error("尝试创建Application的多个实例")
            raise Exception("Application是单例类，请使用get_instance()获取实例")
        Application._instance = self

        logger.debug("初始化Application实例")

        # 配置
        self.config = ConfigManager.get_instance()

        # 状态
        self.running = False
        # protocol instance (set in _set_protocol). Annotate as Any to satisfy
        # static checkers which cannot infer runtime assignment.
        self.protocol: Any = None
        # audio_codec is provided by AudioPlugin at runtime; annotate to avoid
        # Pylance missing-attribute warnings.
        self.audio_codec: Any = None

        # 设备状态（仅主程序改写，插件只读）
        self.device_state = DeviceState.IDLE
        try:
            aec_enabled_cfg = bool(self.config.get_config("AEC_OPTIONS.ENABLED", True))
        except Exception:
            aec_enabled_cfg = True
        self.aec_enabled = aec_enabled_cfg
        self.listening_mode = (
            ListeningMode.REALTIME if self.aec_enabled else ListeningMode.AUTO_STOP
        )
        self.keep_listening = False
        # task for auto-stop when in AUTO_STOP mode
        self._auto_stop_task: asyncio.Task | None = None

        # 统一任务池（替代 _main_tasks/_bg_tasks）
        self._tasks: set[asyncio.Task] = set()

        # 关停事件
        self._shutdown_event: asyncio.Event | None = None

        # 事件循环
        self._main_loop: asyncio.AbstractEventLoop | None = None

        # 并发控制
        self._state_lock: asyncio.Lock | None = None
        self._connect_lock: asyncio.Lock | None = None

        # 插件
        self.plugins = PluginManager()
        # optional SFX player reference (to avoid GC when using Qt player)
        self._sfx_player = None
        # SFX cache: will hold pre-decoded PCM for fast playback
        self._sfx_cached_int16: _np.ndarray | None = None
        self._sfx_cached_float: _np.ndarray | None = None
        self._sfx_cached_sr: int | None = None
        # path to a cached WAV file (written from cached PCM) for winsound/Qt
        self._sfx_cached_wav: Path | None = None

    # -------------------------
    # 生命周期
    # -------------------------
    async def run(self, *, protocol: str = "websocket", mode: str = "gui") -> int:
        logger.info("启动Application，protocol=%s", protocol)
        try:
            self.running = True
            self._main_loop = asyncio.get_running_loop()
            self._initialize_async_objects()
            self._set_protocol(protocol)
            self._setup_protocol_callbacks()
            # 插件：setup（延迟导入AudioPlugin，确保上面setup_opus已执行）
            from src.plugins.audio import AudioPlugin

            # 注册音频、UI、MCP、IoT、唤醒词、快捷键与日程插件（UI模式从run参数传入）
            # 插件会自动按 priority 排序：
            # AudioPlugin(10) -> McpPlugin(20) -> WakeWordPlugin(30) -> CalendarPlugin(40)
            # -> IoTPlugin(50) -> UIPlugin(60) -> ShortcutsPlugin(70)
            self.plugins.register(
                McpPlugin(),
                IoTPlugin(),
                AudioPlugin(),
                WakeWordPlugin(),
                CalendarPlugin(),
                UIPlugin(mode=mode),
                ShortcutsPlugin(),
            )
            await self.plugins.setup_all(self)
            # 启动后广播初始状态，确保 UI 就绪时能看到“待命”
            try:
                await self.plugins.notify_device_state_changed(self.device_state)
            except Exception:
                pass
            # await self.connect_protocol()
            # 插件：start
            await self.plugins.start_all()
            # 等待关停
            await self._wait_shutdown()
            return 0

        except Exception as e:
            logger.error(f"应用运行失败: {e}", exc_info=True)
            return 1
        finally:
            try:
                await self.shutdown()
            except Exception as e:
                logger.error(f"关闭应用时出错: {e}")

    async def connect_protocol(self):
        """
        确保协议通道打开并广播一次协议就绪。返回是否已打开。
        """
        # 已打开直接返回
        assert self.protocol is not None
        try:
            if self.is_audio_channel_opened():
                return True
            if not self._connect_lock:
                # 未初始化锁时，直接尝试一次
                opened = await asyncio.wait_for(
                    self.protocol.open_audio_channel(), timeout=12.0
                )
                if not opened:
                    logger.error("协议连接失败")
                    return False
                logger.info("协议连接已建立，按Ctrl+C退出")
                await self.plugins.notify_protocol_connected(self.protocol)
                return True

            async with self._connect_lock:
                if self.is_audio_channel_opened():
                    return True
                opened = await asyncio.wait_for(
                    self.protocol.open_audio_channel(), timeout=12.0
                )
                if not opened:
                    logger.error("协议连接失败")
                    return False
                logger.info("协议连接已建立，按Ctrl+C退出")
                await self.plugins.notify_protocol_connected(self.protocol)
                return True
        except asyncio.TimeoutError:
            logger.error("协议连接超时")
            return False

    def _initialize_async_objects(self) -> None:
        logger.debug("初始化异步对象")
        self._shutdown_event = asyncio.Event()
        self._state_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()

    def _set_protocol(self, protocol_type: str) -> None:
        logger.debug("设置协议类型: %s", protocol_type)
        if protocol_type == "mqtt":
            self.protocol = MqttProtocol(asyncio.get_running_loop())
        else:
            self.protocol = WebsocketProtocol()

    # -------------------------
    # 手动聆听（按住说话）
    # -------------------------
    async def start_listening_manual(self) -> None:
        try:
            assert self.protocol is not None
            ok = await self.connect_protocol()
            if not ok:
                return
            self.keep_listening = False

            # 如果说话中发送打断
            if self.device_state == DeviceState.SPEAKING:
                logger.info("说话中发送打断")
                await self.protocol.send_abort_speaking(None)
                await self.set_device_state(DeviceState.IDLE)
            await self.protocol.send_start_listening(ListeningMode.MANUAL)
            await self.set_device_state(DeviceState.LISTENING)
        except Exception:
            pass

    async def stop_listening_manual(self) -> None:
        try:
            assert self.protocol is not None
            await self.protocol.send_stop_listening()
            await self.set_device_state(DeviceState.IDLE)
        except Exception:
            pass

    # -------------------------
    # 自动/实时对话：根据 AEC 与当前配置选择模式，开启保持会话
    # -------------------------
    async def start_auto_conversation(self) -> None:
        try:
            assert self.protocol is not None
            ok = await self.connect_protocol()
            if not ok:
                return

            mode = (
                ListeningMode.REALTIME if self.aec_enabled else ListeningMode.AUTO_STOP
            )
            self.listening_mode = mode
            self.keep_listening = True
            await self.protocol.send_start_listening(mode)
            await self.set_device_state(DeviceState.LISTENING)
        except Exception:
            pass

    def _setup_protocol_callbacks(self) -> None:
        assert self.protocol is not None
        self.protocol.on_network_error(self._on_network_error)
        self.protocol.on_incoming_json(self._on_incoming_json)
        self.protocol.on_incoming_audio(self._on_incoming_audio)
        self.protocol.on_audio_channel_opened(self._on_audio_channel_opened)
        self.protocol.on_audio_channel_closed(self._on_audio_channel_closed)

    async def _wait_shutdown(self) -> None:
        # _shutdown_event is created in _initialize_async_objects during run().
        # Assert here so static type checkers know it's not None.
        assert self._shutdown_event is not None
        await self._shutdown_event.wait()

    # -------------------------
    # 统一任务管理（精简）
    # -------------------------
    def spawn(self, coro: Awaitable[Any], name: str) -> Optional[asyncio.Task]:
        """
        创建任务并登记，关停时统一取消。
        """
        if not self.running or (self._shutdown_event and self._shutdown_event.is_set()):
            logger.debug(f"跳过任务创建（应用正在关闭）: {name}")
            return None
        # asyncio.create_task expects a coroutine object; callers sometimes pass
        # other Awaitable types. Use a narrow type-ignore to satisfy static checkers.
        task = asyncio.create_task(coro, name=name)  # type: ignore[arg-type]
        self._tasks.add(task)

        def _done(t: asyncio.Task):
            self._tasks.discard(t)
            if not t.cancelled() and t.exception():
                logger.error(f"任务 {name} 异常结束: {t.exception()}", exc_info=True)

        task.add_done_callback(_done)
        return task

    def schedule_command_nowait(self, fn, *args, **kwargs) -> None:
        if not self._main_loop or self._main_loop.is_closed():
            logger.warning("主事件循环未就绪，拒绝调度")
            return

        def _runner():
            try:
                res = fn(*args, **kwargs)
                if asyncio.iscoroutine(res):
                    self.spawn(res, name=f"call:{getattr(fn, '__name__', 'anon')}")
            except Exception as e:
                logger.error(f"调度的可调用执行失败: {e}", exc_info=True)

        # 确保在事件循环线程里执行
        self._main_loop.call_soon_threadsafe(_runner)

    # -------------------------
    # 协议回调
    # -------------------------
    def _on_network_error(self, error_message=None):
        if error_message:
            logger.error(error_message)

        self.keep_listening = False
        # 出错即请求关闭
        # if self._shutdown_event and not self._shutdown_event.is_set():
        #     self._shutdown_event.set()

    def _on_incoming_audio(self, data: bytes):
        logger.debug(f"收到二进制消息，长度: {len(data)}")
        # 转发给插件
        self.spawn(self.plugins.notify_incoming_audio(data), "plugin:on_audio")

    def _on_incoming_json(self, json_data):
        try:
            msg_type = json_data.get("type") if isinstance(json_data, dict) else None
            logger.info(f"收到JSON消息: type={msg_type}")
            # 将 TTS start/stop 映射为设备状态（支持自动/实时，且不污染手动模式）
            if msg_type == "tts":
                state = json_data.get("state")
                if state == "start":
                    # Play the ready SFX right before TTS starts so the user
                    # knows the AI is about to speak.
                    try:
                        self._play_listening_sfx()
                    except Exception:
                        logger.debug("Failed to play SFX before TTS start", exc_info=True)
                    # 仅当保持会话且实时模式时，TTS开始期间保持LISTENING；否则显示SPEAKING
                    if (
                        self.keep_listening
                        and self.listening_mode == ListeningMode.REALTIME
                    ):
                        self.spawn(
                            self.set_device_state(DeviceState.LISTENING),
                            "state:tts_start_rt",
                        )
                    else:
                        self.spawn(
                            self.set_device_state(DeviceState.SPEAKING),
                            "state:tts_start_speaking",
                        )
                elif state == "stop":
                    if self.keep_listening:
                        # 继续对话：根据当前模式重启监听
                        async def _restart_listening():
                            try:
                                # 先设置状态为 LISTENING，触发音频队列清空和硬件停止等待
                                await self.set_device_state(DeviceState.LISTENING)

                                # 等待音频硬件完全停止后，再发送监听指令
                                # REALTIME 且已在 LISTENING 时无需重复发送
                                if not (
                                    self.listening_mode == ListeningMode.REALTIME
                                    and self.device_state == DeviceState.LISTENING
                                ):
                                    await self.protocol.send_start_listening(
                                        self.listening_mode
                                    )
                            except Exception:
                                pass

                        self.spawn(_restart_listening(), "state:tts_stop_restart")
                    else:
                        self.spawn(
                            self.set_device_state(DeviceState.IDLE),
                            "state:tts_stop_idle",
                        )
            # 转发给插件
            self.spawn(self.plugins.notify_incoming_json(json_data), "plugin:on_json")
        except Exception:
            logger.info("收到JSON消息")

    async def _on_audio_channel_opened(self):
        logger.info("协议通道已打开")
        # 通道打开后进入 LISTENING（：简化为直读直写）
        await self.set_device_state(DeviceState.LISTENING)

    async def _on_audio_channel_closed(self):
        logger.info("协议通道已关闭")
        # 通道关闭回到 IDLE
        await self.set_device_state(DeviceState.IDLE)

    async def set_device_state(self, state: str):
        """
        仅供主程序内部调用：设置设备状态。插件请只读获取。
        """
        # print(f"set_device_state: {state}")
        if not self._state_lock:
            self.device_state = state
            try:
                await self.plugins.notify_device_state_changed(state)
            except Exception:
                pass
            return
        async with self._state_lock:
            if self.device_state == state:
                return
            logger.info(f"设置设备状态: {state}")
            self.device_state = state
            # cancel any existing auto-stop task when state changes
            try:
                if self._auto_stop_task and not self._auto_stop_task.done():
                    self._auto_stop_task.cancel()
            except Exception:
                pass
        # 锁外广播，避免插件回调引起潜在的长耗时阻塞
        try:
            await self.plugins.notify_device_state_changed(state)
            if state == DeviceState.LISTENING:
                # Play a short 'ready' SFX to indicate listening started.
                try:
                    self._play_listening_sfx()
                except Exception:
                    logger.debug("Failed to play listening sfx", exc_info=True)
                await asyncio.sleep(0.5)
                self.aborted = False
                # If in AUTO_STOP mode, schedule an auto-stop after 60 seconds
                try:
                    if self.keep_listening and self.listening_mode == ListeningMode.AUTO_STOP:
                        async def _auto_stop_watch():
                            try:
                                # wait 60 seconds then stop listening if still in AUTO_STOP
                                await asyncio.sleep(60)
                                if self.keep_listening and self.listening_mode == ListeningMode.AUTO_STOP:
                                    await self.protocol.send_stop_listening()
                                    self.keep_listening = False
                                    await self.set_device_state(DeviceState.IDLE)
                            except asyncio.CancelledError:
                                return
                            except Exception:
                                return

                        # spawn and keep reference to allow cancellation on state changes
                        self._auto_stop_task = asyncio.create_task(_auto_stop_watch(), name="auto_stop_watch")
                except Exception:
                    pass
        except Exception:
            pass

    def _play_listening_sfx(self) -> None:
        """
        Play the ready SFX (`src/sfx/ready.mp3`). Try Qt multimedia first,
        fall back to `playsound` if available. Non-blocking where possible.
        """
        logger.debug("_play_listening_sfx invoked")
        try:
            # Ensure cached PCM is prepared (if possible). This will also
            # speed up playback and allow using the cached buffers below.
            try:
                self._ensure_sfx_cached()
            except Exception:
                logger.debug("_ensure_sfx_cached failed", exc_info=True)

            # locate a filesystem SFX file for Qt/playsound/winsound fallbacks if needed
            project_root = Path(__file__).resolve().parents[1]
            candidates = [
                project_root / "src" / "sfx" / "ready.wav",
                project_root / "src" / "sfx" / "ready.mp3",
                project_root / "src" / "sfx" / "ready.m4a",
            ]
            sfx_path = next((p for p in candidates if p.exists()), None)
            if sfx_path is None and self._sfx_cached_int16 is None and self._sfx_cached_float is None:
                logger.debug("SFX file not found and no cached audio available")
                return
            logger.debug(
                "SFX candidate file=%s, cached_int16=%s, cached_float=%s, cached_wav=%s",
                sfx_path,
                "yes" if self._sfx_cached_int16 is not None else "no",
                "yes" if self._sfx_cached_float is not None else "no",
                self._sfx_cached_wav,
            )
            # Preferred method on Linux: route SFX through the same audio output
            # used by the app's `AudioCodec` (sounddevice). Decode the MP3 to
            # raw PCM matching `AudioConfig.OUTPUT_SAMPLE_RATE` and feed it to
            # `audio_codec.write_pcm_direct()`, so the SFX uses the same output
            # device/sink as TTS playback.
            try:
                if hasattr(self, "audio_codec") and getattr(self, "audio_codec"):
                    # Ensure SFX is cached (decode once)
                    try:
                        self._ensure_sfx_cached()
                    except Exception:
                        logger.debug("Failed to ensure SFX cached", exc_info=True)

                    async def _play_cached_via_audio_codec():
                        try:
                            from src.constants.constants import AudioConfig

                            if self._sfx_cached_int16 is None:
                                logger.debug("No cached int16 SFX available")
                                return

                            pcm = self._sfx_cached_int16
                            frame_size = AudioConfig.OUTPUT_FRAME_SIZE
                            total = len(pcm)
                            idx = 0
                            while idx < total and self.running:
                                end = idx + frame_size
                                chunk = pcm[idx:end]
                                if len(chunk) < frame_size:
                                    pad = _np.zeros(frame_size - len(chunk), dtype=_np.int16)
                                    chunk = _np.concatenate((chunk, pad))
                                try:
                                    await self.audio_codec.write_pcm_direct(chunk)
                                except Exception:
                                    break
                                idx = end

                        except Exception:
                            logger.debug("SFX via audio_codec failed", exc_info=True)

                    self.spawn(_play_cached_via_audio_codec(), name="sfx:play_via_codec")
                    logger.debug("Routed SFX via AudioCodec (spawned task)")
                    return
            except Exception:
                # fall through to existing fallbacks
                logger.debug("Routing SFX via audio codec unavailable, falling back", exc_info=True)

            # If no audio_codec route, try playing the cached float via sounddevice
            # to ensure the sound goes to the same device as the app (if possible).
            if getattr(self, "_sfx_cached_float", None) is not None:
                try:
                    import sounddevice as sd

                    device = None
                    if hasattr(self, "audio_codec") and getattr(self, "audio_codec"):
                        device = getattr(self.audio_codec, "speaker_device_id", None)

                    try:
                        sd.play(self._sfx_cached_float, samplerate=self._sfx_cached_sr, device=device)
                        # do not block; let sounddevice stream in background
                        logger.debug("Played SFX via sounddevice (cached float)")
                        return
                    except Exception:
                        logger.debug("sounddevice play failed, falling back", exc_info=True)
                except Exception:
                    logger.debug("sounddevice module unavailable, skipping", exc_info=True)

            # On Windows, use winsound for WAV files as a reliable fallback
            try:
                if sys.platform.startswith("win"):
                    # prefer writing/using cached WAV if available (no external deps)
                    wav_to_play = None
                    if self._sfx_cached_wav and Path(self._sfx_cached_wav).exists():
                        wav_to_play = self._sfx_cached_wav
                    elif sfx_path is not None and sfx_path.suffix.lower() == ".wav":
                        wav_to_play = sfx_path

                    if wav_to_play is not None:
                        try:
                            import winsound

                            winsound.PlaySound(str(wav_to_play), winsound.SND_FILENAME | winsound.SND_ASYNC)
                            logger.debug("Played SFX via winsound (Windows WAV async): %s", wav_to_play)
                            return
                        except Exception:
                            logger.debug("winsound playback failed, falling back", exc_info=True)
            except Exception:
                # ignore platform detection errors
                pass

            if _QT_MEDIA_AVAILABLE:
                try:
                    # Keep a reference on the Application instance to avoid
                    # the QMediaPlayer being garbage-collected immediately.
                    player = QMediaPlayer()
                    if sfx_path is None:
                        raise FileNotFoundError("No SFX file available for Qt fallback")
                    url = QUrl.fromLocalFile(str(sfx_path))
                    media = QMediaContent(url)
                    player.setMedia(media)
                    player.setVolume(50)
                    player.play()
                    self._sfx_player = player
                    logger.debug("Played SFX via Qt multimedia")
                    return
                except Exception:
                    logger.debug("Qt multimedia play failed, falling back", exc_info=True)

            if _PLAYSOUND_AVAILABLE and sfx_path is not None:
                # playsound is blocking; run in background thread to avoid blocking loop
                threading.Thread(target=playsound, args=(str(sfx_path),), daemon=True).start()
                logger.debug("Played SFX via playsound fallback")
                return

            logger.debug("No available method to play SFX (audio_codec, Qt multimedia or playsound)")
        except Exception:
            logger.exception("Unexpected error while attempting to play SFX")

    def _ensure_sfx_cached(self) -> None:
        """Decode SFX once and cache int16 (target sample rate) and float32 arrays.

        This method is synchronous and idempotent. It prefers existing WAV files
        (faster) and falls back to ffmpeg to decode MP3/other formats.
        """
        if self._sfx_cached_int16 is not None and self._sfx_cached_float is not None:
            return

        project_root = Path(__file__).resolve().parents[1]
        # prefer wav for fast decoding
        candidates = [
            project_root / "src" / "sfx" / "ready.wav",
            project_root / "src" / "sfx" / "ready.mp3",
            project_root / "src" / "sfx" / "ready.m4a",
        ]
        sfx_path = None
        for p in candidates:
            if p.exists():
                sfx_path = p
                break
        if sfx_path is None:
            logger.debug("No SFX file found to cache")
            return

        from src.constants.constants import AudioConfig

        target_sr = AudioConfig.OUTPUT_SAMPLE_RATE

        try:
            # If WAV, read directly via wave
            if sfx_path.suffix.lower() == ".wav":
                import wave

                with wave.open(str(sfx_path), "rb") as wf:
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    fr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())

                if sampwidth == 2:
                    arr = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0
                elif sampwidth == 4:
                    arr = _np.frombuffer(raw, dtype=_np.int32).astype(_np.float32) / 2147483648.0
                elif sampwidth == 1:
                    arr = _np.frombuffer(raw, dtype=_np.uint8).astype(_np.float32)
                    arr = (arr - 128.0) / 128.0
                else:
                    arr = _np.frombuffer(raw, dtype=_np.int16).astype(_np.float32) / 32768.0

                if channels > 1:
                    arr = arr.reshape(-1, channels).mean(axis=1)

                # resample if needed
                if fr != target_sr:
                    try:
                        import soxr as _soxr
                        resampler = _soxr.ResampleStream(fr, target_sr, num_channels=1, dtype="float32", quality="Q")
                        arr = resampler.resample_chunk(arr, last=True)
                    except Exception:
                        # If soxr unavailable or fails, fallback to ffmpeg decode+resample
                        logger.debug("soxr resample failed; falling back to ffmpeg for WAV resample", exc_info=True)
                        ffmpeg = shutil.which("ffmpeg")
                        if not ffmpeg:
                            logger.debug("ffmpeg not found; cannot resample WAV")
                            # allow outer exception handling to log and return
                            raise
                        cmd = [
                            ffmpeg,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            str(sfx_path),
                            "-f",
                            "s16le",
                            "-acodec",
                            "pcm_s16le",
                            "-ac",
                            "1",
                            "-ar",
                            str(target_sr),
                            "-",
                        ]
                        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if not p.stdout:
                            logger.debug("ffmpeg returned no audio for WAV resample")
                            raise RuntimeError("ffmpeg_no_output")
                        arr = _np.frombuffer(p.stdout, dtype=_np.int16).astype(_np.float32) / 32768.0

            else:
                # use ffmpeg to decode to s16le mono at target_sr
                ffmpeg = shutil.which("ffmpeg")
                if not ffmpeg:
                    logger.debug("ffmpeg not found, cannot decode SFX")
                    return
                cmd = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(sfx_path),
                    "-f",
                    "s16le",
                    "-acodec",
                    "pcm_s16le",
                    "-ac",
                    "1",
                    "-ar",
                    str(target_sr),
                    "-",
                ]
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if not p.stdout:
                    logger.debug("ffmpeg returned no audio for SFX decoding")
                    return
                arr = _np.frombuffer(p.stdout, dtype=_np.int16).astype(_np.float32) / 32768.0

            # Convert to int16 cached and float32 cached
            int16 = (_np.clip(arr, -1.0, 1.0) * 32767.0).astype(_np.int16)
            float32 = arr.astype(_np.float32)

            self._sfx_cached_int16 = int16
            self._sfx_cached_float = float32
            self._sfx_cached_sr = target_sr
            # Also write a cached WAV file for reliable Windows playback (winsound)
            try:
                cache_dir = project_root / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                wav_path = cache_dir / "sfx_ready_cached.wav"
                import wave

                with wave.open(str(wav_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(target_sr)
                    wf.writeframes(int16.tobytes())
                self._sfx_cached_wav = wav_path
                logger.debug("Wrote cached SFX WAV: %s", wav_path)
            except Exception:
                logger.debug("Failed to write cached WAV", exc_info=True)
            logger.debug(f"SFX cached: frames={len(int16)}, sr={target_sr}")

        except Exception as e:
            logger.debug(f"Failed to cache SFX: {e}", exc_info=True)

    # -------------------------
    # 只读访问器（提供给插件使用）
    # -------------------------
    def get_device_state(self):
        return self.device_state

    def is_idle(self) -> bool:
        return self.device_state == DeviceState.IDLE

    def is_listening(self) -> bool:
        return self.device_state == DeviceState.LISTENING

    def is_speaking(self) -> bool:
        return self.device_state == DeviceState.SPEAKING

    def get_listening_mode(self):
        return self.listening_mode

    def is_keep_listening(self) -> bool:
        return bool(self.keep_listening)

    def is_audio_channel_opened(self) -> bool:
        try:
            return bool(self.protocol and self.protocol.is_audio_channel_opened())
        except Exception:
            return False

    def should_capture_audio(self) -> bool:
        try:
            if self.device_state == DeviceState.LISTENING and not self.aborted:
                return True

            return (
                self.device_state == DeviceState.SPEAKING
                and self.aec_enabled
                and self.keep_listening
                and self.listening_mode == ListeningMode.REALTIME
            )
        except Exception:
            return False

    def get_state_snapshot(self) -> dict:
        return {
            "device_state": self.device_state,
            "listening_mode": self.listening_mode,
            "keep_listening": bool(self.keep_listening),
            "audio_opened": self.is_audio_channel_opened(),
        }

    async def abort_speaking(self, reason):
        """
        中止语音输出.
        """

        assert self.protocol is not None
        if self.aborted:
            logger.debug(f"已经中止，忽略重复的中止请求: {reason}")
            return

        logger.info(f"中止语音输出，原因: {reason}")
        self.aborted = True
        await self.protocol.send_abort_speaking(reason)
        await self.set_device_state(DeviceState.IDLE)

    # -------------------------
    # UI 辅助：供插件或工具直接调用
    # -------------------------
    def set_chat_message(self, role, message: str) -> None:
        """将文本更新转发为 UI 可识别的 JSON 消息（复用 UIPlugin 的 on_incoming_json）。
        role: "assistant" | "user" 影响消息类型映射。
        """
        try:
            msg_type = "tts" if str(role).lower() == "assistant" else "stt"
        except Exception:
            msg_type = "tts"
        payload = {"type": msg_type, "text": message}
        # 通过插件事件总线异步派发
        self.spawn(self.plugins.notify_incoming_json(payload), "ui:text_update")

    def set_emotion(self, emotion: str) -> None:
        """
        设置情绪表情：通过 UIPlugin 的 on_incoming_json 路由。
        """
        payload = {"type": "llm", "emotion": emotion}
        self.spawn(self.plugins.notify_incoming_json(payload), "ui:emotion_update")

    # -------------------------
    # 关停
    # -------------------------
    async def shutdown(self):
        if not self.running:
            return
        logger.info("正在关闭Application...")
        self.running = False

        if self._shutdown_event is not None:
            self._shutdown_event.set()

        try:
            # 取消所有登记任务
            if self._tasks:
                for t in list(self._tasks):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
                self._tasks.clear()

            # 关闭协议（限时，避免阻塞退出)
            if self.protocol:
                try:
                    if self._main_loop is not None:
                        try:
                            self._main_loop.create_task(self.protocol.close_audio_channel())
                        except asyncio.TimeoutError:
                            logger.warning("关闭协议超时，跳过等待")
                except Exception as e:
                    logger.error(f"关闭协议失败: {e}")

            # 插件：stop/shutdown
            try:
                await self.plugins.stop_all()
            except Exception:
                pass
            try:
                await self.plugins.shutdown_all()
            except Exception:
                pass

            logger.info("Application 关闭完成")
        except Exception as e:
            logger.error(f"关闭应用时出错: {e}", exc_info=True)
