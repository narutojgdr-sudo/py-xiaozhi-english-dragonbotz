import json

from src.constants.constants import AbortReason, ListeningMode
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Protocol:
    def __init__(self):
        self.session_id = ""
        # Initialize callbacks to None.
        self._on_incoming_json = None
        self._on_incoming_audio = None
        self._on_audio_channel_opened = None
        self._on_audio_channel_closed = None
        self._on_network_error = None
        # Connection state change callbacks.
        self._on_connection_state_changed = None
        self._on_reconnecting = None

    def on_incoming_json(self, callback):
        """
        Set JSON message receive callback.
        """
        self._on_incoming_json = callback

    def on_incoming_audio(self, callback):
        """
        Set audio data receive callback.
        """
        self._on_incoming_audio = callback

    def on_audio_channel_opened(self, callback):
        """
        Set audio channel opened callback.
        """
        self._on_audio_channel_opened = callback

    def on_audio_channel_closed(self, callback):
        """
        Set audio channel closed callback.
        """
        self._on_audio_channel_closed = callback

    def on_network_error(self, callback):
        """
        Set network error callback.
        """
        self._on_network_error = callback

    def on_connection_state_changed(self, callback):
        """Set connection state change callback.

        Args:
            callback: Callback with (connected: bool, reason: str)
        """
        self._on_connection_state_changed = callback

    def on_reconnecting(self, callback):
        """Set reconnect attempt callback.

        Args:
            callback: Callback with (attempt: int, max_attempts: int)
        """
        self._on_reconnecting = callback

    async def send_text(self, message):
        """
        Abstract method for sending text messages.
        """
        raise NotImplementedError("send_text must be implemented by subclasses")

    async def send_audio(self, data: bytes):
        """
        Abstract method for sending audio data.
        """
        raise NotImplementedError("send_audio must be implemented by subclasses")

    def is_audio_channel_opened(self) -> bool:
        """
        Abstract method to check if audio channel is open.
        """
        raise NotImplementedError(
            "is_audio_channel_opened must be implemented by subclasses"
        )

    async def open_audio_channel(self) -> bool:
        """
        Abstract method to open the audio channel.
        """
        raise NotImplementedError(
            "open_audio_channel must be implemented by subclasses"
        )

    async def close_audio_channel(self):
        """
        Abstract method to close the audio channel.
        """
        raise NotImplementedError(
            "close_audio_channel must be implemented by subclasses"
        )

    async def send_abort_speaking(self, reason):
        """
        Send a message to abort speaking.
        """
        message = {"session_id": self.session_id, "type": "abort"}
        if reason == AbortReason.WAKE_WORD_DETECTED:
            message["reason"] = "wake_word_detected"
        await self.send_text(json.dumps(message))

    async def send_wake_word_detected(self, wake_word):
        """
        Send a wake-word detected message.
        """
        message = {
            "session_id": self.session_id,
            "type": "listen",
            "state": "detect",
            "text": wake_word,
        }
        await self.send_text(json.dumps(message))

    async def send_start_listening(self, mode):
        """
        Send a start listening message.
        """
        mode_map = {
            ListeningMode.REALTIME: "realtime",
            ListeningMode.AUTO_STOP: "auto",
            ListeningMode.MANUAL: "manual",
        }
        message = {
            "session_id": self.session_id,
            "type": "listen",
            "state": "start",
            "mode": mode_map[mode],
        }
        await self.send_text(json.dumps(message))

    async def send_stop_listening(self):
        """
        Send a stop listening message.
        """
        message = {"session_id": self.session_id, "type": "listen", "state": "stop"}
        await self.send_text(json.dumps(message))

    async def send_iot_descriptors(self, descriptors):
        """
        Send IoT device descriptor information.
        """
        try:
            # 解析描述符数据
            if isinstance(descriptors, str):
                descriptors_data = json.loads(descriptors)
            else:
                descriptors_data = descriptors

            # 检查是否为数组
            if not isinstance(descriptors_data, list):
                logger.error("IoT descriptors should be an array")
                return

            # 为每个描述符发送单独的消息
            for i, descriptor in enumerate(descriptors_data):
                if descriptor is None:
                    logger.error(f"Failed to get IoT descriptor at index {i}")
                    continue

                message = {
                    "session_id": self.session_id,
                    "type": "iot",
                    "update": True,
                    "descriptors": [descriptor],
                }

                try:
                    await self.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(
                        f"Failed to send JSON message for IoT descriptor "
                        f"at index {i}: {e}"
                    )
                    continue

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse IoT descriptors: {e}")
            return

    async def send_iot_states(self, states):
        """
        Send IoT device state information.
        """
        if isinstance(states, str):
            states_data = json.loads(states)
        else:
            states_data = states

        message = {
            "session_id": self.session_id,
            "type": "iot",
            "update": True,
            "states": states_data,
        }
        await self.send_text(json.dumps(message))

    async def send_mcp_message(self, payload):
        """
        Send an MCP message.
        """
        if isinstance(payload, str):
            payload_data = json.loads(payload)
        else:
            payload_data = payload

        message = {
            "session_id": self.session_id,
            "type": "mcp",
            "payload": payload_data,
        }

        await self.send_text(json.dumps(message))
