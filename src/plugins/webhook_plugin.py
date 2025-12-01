import asyncio
import json
from datetime import datetime
from typing import Any

import aiohttp

from .base import Plugin
from src.utils.logging_config import get_logger
from src.constants.constants import DeviceState


logger = get_logger(__name__)


class WebhookPlugin(Plugin):
    """Plugin yang memicu webhook saat device state berubah: LISTENING start dan IDLE stop.

    Konfigurasi:
      WEBHOOKS.on_listening_start: URL untuk dipanggil saat masuk LISTENING
      WEBHOOKS.on_listening_stop:  URL untuk dipanggil saat kembali ke IDLE
    """

    name = "webhook"
    priority = 70

    def __init__(self) -> None:
        super().__init__()
        self._app = None
        self._session: aiohttp.ClientSession | None = None

    async def setup(self, app: Any) -> None:
        self._app = app
        # reuse a single session for efficiency
        try:
            self._session = aiohttp.ClientSession()
        except Exception:
            self._session = None

    async def start(self) -> None:
        await super().start()

    async def on_device_state_changed(self, state: Any) -> None:
        try:
            if not self._app:
                return
            config = getattr(self._app, "config", None)
            if not config:
                return

            # read webhook urls from config manager (if available)
            try:
                cfg_mgr = self._app.config
                start_url = cfg_mgr.get_config("WEBHOOKS.on_listening_start", None)
                stop_url = cfg_mgr.get_config("WEBHOOKS.on_listening_stop", None)
            except Exception:
                start_url = None
                stop_url = None

            payload = {
                "state": str(state),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            # Non-blocking fire-and-forget tasks
            if state == DeviceState.LISTENING and start_url:
                asyncio.create_task(self._post_safe(start_url, payload), name="webhook:start")

            if state == DeviceState.IDLE and stop_url:
                asyncio.create_task(self._post_safe(stop_url, payload), name="webhook:stop")

        except Exception:
            logger.exception("WebhookPlugin failed handling device state change")

    async def _post_safe(self, url: str, payload: dict) -> None:
        try:
            headers = {"Content-Type": "application/json"}
            if self._session:
                async with self._session.post(url, data=json.dumps(payload), headers=headers, timeout=10) as resp:
                    # read to ensure connection completes; but don't raise on non-2xx
                    _ = await resp.text()
            else:
                # fallback: create a short-lived session
                async with aiohttp.ClientSession() as s:
                    async with s.post(url, data=json.dumps(payload), headers=headers, timeout=10) as resp:
                        _ = await resp.text()
            logger.debug("Webhook posted to %s", url)
        except Exception:
            logger.exception("Failed to post webhook to %s", url)

    async def stop(self) -> None:
        await super().stop()
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception:
            pass

    async def shutdown(self) -> None:
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        except Exception:
            pass
