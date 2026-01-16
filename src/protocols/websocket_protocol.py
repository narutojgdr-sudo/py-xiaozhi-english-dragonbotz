import asyncio
import json
import ssl
import time

import websockets

from src.constants.constants import AudioConfig
from src.protocols.protocol import Protocol
from src.utils.config_manager import ConfigManager
from src.utils.logging_config import get_logger

ssl_context = ssl._create_unverified_context()

logger = get_logger(__name__)


class WebsocketProtocol(Protocol):
    def __init__(self):
        super().__init__()
        # Get configuration manager instance.
        self.config = ConfigManager.get_instance()
        self.websocket = None
        self.connected = False
        self.hello_received = None  # Initialize as None.
        # Message task reference for cancellation on close.
        self._message_task = None

        # Connection health monitoring.
        self._last_ping_time = None
        self._last_pong_time = None
        self._ping_interval = 30.0  # Heartbeat interval (seconds).
        self._ping_timeout = 10.0  # Ping timeout (seconds).
        self._heartbeat_task = None
        self._connection_monitor_task = None

        # Connection state flags.
        self._is_closing = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 0  # No reconnect by default.
        self._auto_reconnect_enabled = False  # Auto-reconnect disabled by default.

        self.WEBSOCKET_URL = self.config.get_config(
            "SYSTEM_OPTIONS.NETWORK.WEBSOCKET_URL"
        )
        access_token = self.config.get_config(
            "SYSTEM_OPTIONS.NETWORK.WEBSOCKET_ACCESS_TOKEN"
        )
        device_id = self.config.get_config("SYSTEM_OPTIONS.DEVICE_ID")
        client_id = self.config.get_config("SYSTEM_OPTIONS.CLIENT_ID")

        self.HEADERS = {
            "Authorization": f"Bearer {access_token}",
            "Protocol-Version": "1",
            "Device-Id": device_id,  # Device MAC address
            "Client-Id": client_id,
        }

    async def connect(self) -> bool:
        """
        Connect to the WebSocket server.
        """
        if self._is_closing:
            logger.warning("Connection is closing; cancel new connect attempt.")
            return False

        try:
            # Create Event on connect to ensure correct event loop.
            self.hello_received = asyncio.Event()

            # Determine whether to use SSL.
            current_ssl_context = None
            if self.WEBSOCKET_URL.startswith("wss://"):
                current_ssl_context = ssl_context

            # Create WebSocket connection (compatible with Python versions).
            try:
                # New style (Python 3.11+).
                self.websocket = await websockets.connect(
                    uri=self.WEBSOCKET_URL,
                    ssl=current_ssl_context,
                    additional_headers=self.HEADERS,
                    ping_interval=20,  # websockets heartbeat, 20s interval
                    ping_timeout=20,  # ping timeout 20s
                    close_timeout=10,  # close timeout 10s
                    max_size=10 * 1024 * 1024,  # max message 10MB
                    compression=None,  # disable compression for stability
                )
            except TypeError:
                # Legacy style (older Python).
                self.websocket = await websockets.connect(
                    self.WEBSOCKET_URL,
                    ssl=current_ssl_context,
                    extra_headers=self.HEADERS,
                    ping_interval=20,  # websockets heartbeat
                    ping_timeout=20,  # ping timeout 20s
                    close_timeout=10,  # close timeout 10s
                    max_size=10 * 1024 * 1024,  # max message 10MB
                    compression=None,  # disable compression
                )

            # Start message handler loop (keep task for cancellation).
            self._message_task = asyncio.create_task(self._message_handler())

            # Custom heartbeat disabled; use websockets heartbeat.
            # self._start_heartbeat()

            # Start connection monitor.
            self._start_connection_monitor()

            # Send client hello message.
            hello_message = {
                "type": "hello",
                "version": 1,
                "features": {
                    "mcp": True,
                },
                "transport": "websocket",
                "audio_params": {
                    "format": "opus",
                    "sample_rate": AudioConfig.INPUT_SAMPLE_RATE,
                    "channels": AudioConfig.CHANNELS,
                    "frame_duration": AudioConfig.FRAME_DURATION,
                },
            }
            await self.send_text(json.dumps(hello_message))

            # Wait for server hello response.
            try:
                await asyncio.wait_for(self.hello_received.wait(), timeout=10.0)
                self.connected = True
                self._reconnect_attempts = 0  # Reset reconnect counter.
                logger.info("Connected to WebSocket server.")

                # Notify connection state change.
                if self._on_connection_state_changed:
                    self._on_connection_state_changed(True, "Connected")

                return True
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for server hello response.")
                await self._cleanup_connection()
                if self._on_network_error:
                    self._on_network_error("Timed out waiting for response.")
                return False

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self._cleanup_connection()
            if self._on_network_error:
                self._on_network_error(f"Unable to connect to service: {str(e)}")
            return False

    def _start_heartbeat(self):
        """
        Start heartbeat task.
        """
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def _start_connection_monitor(self):
        """
        Start connection monitor task.
        """
        if (
            self._connection_monitor_task is None
            or self._connection_monitor_task.done()
        ):
            self._connection_monitor_task = asyncio.create_task(
                self._connection_monitor()
            )

    async def _heartbeat_loop(self):
        """
        Heartbeat loop.
        """
        try:
            while self.websocket and not self._is_closing:
                await asyncio.sleep(self._ping_interval)

                if self.websocket and not self._is_closing:
                    try:
                        self._last_ping_time = time.time()
                        # Send ping and wait for pong.
                        pong_waiter = await self.websocket.ping()
                        logger.debug("Sent heartbeat ping.")

                        # Wait for pong response.
                        try:
                            await asyncio.wait_for(
                                pong_waiter, timeout=self._ping_timeout
                            )
                            self._last_pong_time = time.time()
                            logger.debug("Received heartbeat pong.")
                        except asyncio.TimeoutError:
                            logger.warning("Heartbeat pong timed out.")
                            await self._handle_connection_loss(
                                "Heartbeat pong timed out."
                            )
                            break

                    except Exception as e:
                        logger.error(f"Failed to send heartbeat: {e}")
                        await self._handle_connection_loss("Heartbeat send failed.")
                        break
        except asyncio.CancelledError:
            logger.debug("Heartbeat task canceled.")
        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")

    async def _connection_monitor(self):
        """
        Connection health monitor.
        """
        try:
            while self.websocket and not self._is_closing:
                await asyncio.sleep(5)  # Check every 5 seconds.

                # Check connection status.
                if self.websocket:
                    if self.websocket.close_code is not None:
                        logger.warning("Detected WebSocket connection closed.")
                        await self._handle_connection_loss("Connection closed.")
                        break

        except asyncio.CancelledError:
            logger.debug("Connection monitor task canceled.")
        except Exception as e:
            logger.error(f"Connection monitor error: {e}")

    async def _handle_connection_loss(self, reason: str):
        """
        Handle connection loss.
        """
        logger.warning(f"Connection lost: {reason}")

        # Update connection state.
        was_connected = self.connected
        self.connected = False

        # Notify connection state change.
        if self._on_connection_state_changed and was_connected:
            try:
                self._on_connection_state_changed(False, reason)
            except Exception as e:
                logger.error(f"Failed to call connection state callback: {e}")

        # Clean up connection.
        await self._cleanup_connection()

        # Notify audio channel closed.
        if self._on_audio_channel_closed:
            try:
                await self._on_audio_channel_closed()
            except Exception as e:
                logger.error(f"Failed to call audio channel closed callback: {e}")

        # Reconnect only if auto-reconnect enabled and not manually closed.
        if (
            not self._is_closing
            and self._auto_reconnect_enabled
            and self._reconnect_attempts < self._max_reconnect_attempts
        ):
            await self._attempt_reconnect(reason)
        else:
            # Notify network error.
            if self._on_network_error:
                if (
                    self._auto_reconnect_enabled
                    and self._reconnect_attempts >= self._max_reconnect_attempts
                ):
                    self._on_network_error(
                        f"Connection lost and reconnect failed: {reason}"
                    )
                else:
                    self._on_network_error(f"Connection lost: {reason}")

    async def _attempt_reconnect(self, original_reason: str):
        """
        Attempt auto-reconnect.
        """
        self._reconnect_attempts += 1

        # Notify reconnect start.
        if self._on_reconnecting:
            try:
                self._on_reconnecting(
                    self._reconnect_attempts, self._max_reconnect_attempts
                )
            except Exception as e:
                logger.error(f"Failed to call reconnect callback: {e}")

        logger.info(
            "Attempting auto-reconnect (%s/%s)",
            self._reconnect_attempts,
            self._max_reconnect_attempts,
        )

        # Wait before reconnecting (exponential backoff, max 30s).
        await asyncio.sleep(min(self._reconnect_attempts * 2, 30))

        try:
            success = await self.connect()
            if success:
                logger.info("Auto-reconnect succeeded.")
                # Notify connection state change.
                if self._on_connection_state_changed:
                    self._on_connection_state_changed(True, "Reconnected")
            else:
                logger.warning(
                    "Auto-reconnect failed (%s/%s)",
                    self._reconnect_attempts,
                    self._max_reconnect_attempts,
                )
                # Don't error immediately if retries remain.
                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    if self._on_network_error:
                        self._on_network_error(
                            f"Reconnect failed; max retries reached: {original_reason}"
                        )
        except Exception as e:
            logger.error(f"Error during reconnect: {e}")
            if self._reconnect_attempts >= self._max_reconnect_attempts:
                if self._on_network_error:
                    self._on_network_error(f"Reconnect error: {str(e)}")

    def enable_auto_reconnect(self, enabled: bool = True, max_attempts: int = 5):
        """Enable or disable auto-reconnect.

        Args:
            enabled: Whether to enable auto-reconnect
            max_attempts: Maximum reconnect attempts
        """
        self._auto_reconnect_enabled = enabled
        if enabled:
            self._max_reconnect_attempts = max_attempts
            logger.info(f"Auto-reconnect enabled, max attempts: {max_attempts}")
        else:
            self._max_reconnect_attempts = 0
            logger.info("Auto-reconnect disabled.")

    def get_connection_info(self) -> dict:
        """Get connection information.

        Returns:
            dict: Connection state, reconnect attempts, etc.
        """
        return {
            "connected": self.connected,
            "websocket_closed": (
                self.websocket.close_code is not None if self.websocket else True
            ),
            "is_closing": self._is_closing,
            "auto_reconnect_enabled": self._auto_reconnect_enabled,
            "reconnect_attempts": self._reconnect_attempts,
            "max_reconnect_attempts": self._max_reconnect_attempts,
            "last_ping_time": self._last_ping_time,
            "last_pong_time": self._last_pong_time,
            "websocket_url": self.WEBSOCKET_URL,
        }

    async def _message_handler(self):
        """
        Handle incoming WebSocket messages.
        """
        try:
            async for message in self.websocket:
                if self._is_closing:
                    break

                try:
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type")
                            if msg_type == "hello":
                                # Handle server hello message.
                                await self._handle_server_hello(data)
                            else:
                                if self._on_incoming_json:
                                    self._on_incoming_json(data)
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Invalid JSON message: {message}, error: {e}"
                            )
                    elif isinstance(message, bytes):
                        # Binary message, likely audio.
                        if self._on_incoming_audio:
                            self._on_incoming_audio(message)
                except Exception as e:
                    # Handle single message errors, continue processing.
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    continue

        except asyncio.CancelledError:
            logger.debug("Message handler task canceled.")
            return
        except websockets.ConnectionClosed as e:
            if not self._is_closing:
                logger.info(f"WebSocket connection closed: {e}")
                await self._handle_connection_loss(
                    f"Connection closed: {e.code} {e.reason}"
                )
        except websockets.ConnectionClosedError as e:
            if not self._is_closing:
                logger.info(f"WebSocket connection closed with error: {e}")
                await self._handle_connection_loss(
                    f"Connection error: {e.code} {e.reason}"
                )
        except websockets.InvalidState as e:
            logger.error(f"WebSocket invalid state: {e}")
            await self._handle_connection_loss("Connection state error.")
        except ConnectionResetError:
            logger.warning("Connection reset.")
            await self._handle_connection_loss("Connection reset.")
        except OSError as e:
            logger.error(f"Network I/O error: {e}")
            await self._handle_connection_loss("Network I/O error.")
        except Exception as e:
            logger.error(f"Message handler loop error: {e}", exc_info=True)
            await self._handle_connection_loss(f"Message handler error: {str(e)}")

    async def send_audio(self, data: bytes):
        """
        Send audio data.
        """
        if not self.is_audio_channel_opened():
            return

        try:
            await self.websocket.send(data)
        except websockets.ConnectionClosed as e:
            logger.warning(f"Connection closed while sending audio: {e}")
            await self._handle_connection_loss(
                f"Send audio failed: {e.code} {e.reason}"
            )
        except websockets.ConnectionClosedError as e:
            logger.warning(f"Connection error while sending audio: {e}")
            await self._handle_connection_loss(
                f"Send audio error: {e.code} {e.reason}"
            )
        except Exception as e:
            logger.error(f"Failed to send audio data: {e}")
            # Let connection handler deal with network errors.
            await self._handle_connection_loss(f"Send audio error: {str(e)}")

    async def send_text(self, message: str):
        """
        Send a text message.
        """
        if not self.websocket or self._is_closing:
            logger.warning("WebSocket not connected or closing; cannot send message.")
            return

        try:
            await self.websocket.send(message)
        except websockets.ConnectionClosed as e:
            logger.warning(f"Connection closed while sending text: {e}")
            await self._handle_connection_loss(
                f"Send text failed: {e.code} {e.reason}"
            )
        except websockets.ConnectionClosedError as e:
            logger.warning(f"Connection error while sending text: {e}")
            await self._handle_connection_loss(
                f"Send text error: {e.code} {e.reason}"
            )
        except Exception as e:
            logger.error(f"Failed to send text message: {e}")
            await self._handle_connection_loss(f"Send text error: {str(e)}")

    def is_audio_channel_opened(self) -> bool:
        """Check whether the audio channel is open.

        Checks connection state including WebSocket state.
        """
        if not self.websocket or not self.connected or self._is_closing:
            return False

        # Check WebSocket actual state.
        try:
            return self.websocket.close_code is None
        except Exception:
            return False

    async def open_audio_channel(self) -> bool:
        """Establish WebSocket connection.

        Create a new WebSocket connection if not connected.
        Returns:
            bool: Whether connection succeeded
        """
        if not self.is_audio_channel_opened():
            return await self.connect()
        return True

    async def _handle_server_hello(self, data: dict):
        """
        Handle server hello message.
        """
        try:
            # Validate transport.
            transport = data.get("transport")
            if not transport or transport != "websocket":
                logger.error(f"Unsupported transport: {transport}")
                return

            # Set hello received event.
            self.hello_received.set()

            # Notify audio channel opened.
            if self._on_audio_channel_opened:
                await self._on_audio_channel_opened()

            logger.info("Handled server hello message successfully.")

        except Exception as e:
            logger.error(f"Error handling server hello message: {e}")
            if self._on_network_error:
                self._on_network_error(f"Failed to handle server response: {str(e)}")

    async def _cleanup_connection(self):
        """
        Clean up connection resources.
        """
        self.connected = False

        # Cancel message task to avoid pending waits after loop exit.
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Error waiting for message task cancel: {e}")
        self._message_task = None

        # Cancel heartbeat task.
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Cancel connection monitor task.
        if self._connection_monitor_task and not self._connection_monitor_task.done():
            self._connection_monitor_task.cancel()
            try:
                await self._connection_monitor_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connection.
        if self.websocket and self.websocket.close_code is None:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")

        self.websocket = None
        self._last_ping_time = None
        self._last_pong_time = None

    async def close_audio_channel(self):
        """
        Close the audio channel.
        """
        self._is_closing = True

        try:
            await self._cleanup_connection()

            if self._on_audio_channel_closed:
                await self._on_audio_channel_closed()

        except Exception as e:
            logger.error(f"Failed to close audio channel: {e}")
        finally:
            self._is_closing = False
