"""System tool implementations.

Provide concrete system tool functionality.
"""

import asyncio
from typing import Any, Dict

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


async def set_volume(args: Dict[str, Any]) -> bool:
    """
    Set volume.
    """
    try:
        volume = args["volume"]
        logger.info(f"[SystemTools] Setting volume to {volume}")

        # Validate volume range.
        if not (0 <= volume <= 100):
            logger.warning(f"[SystemTools] Volume out of range: {volume}")
            return False

        # Use VolumeController directly.
        from src.utils.volume_controller import VolumeController

        # Check dependencies and create controller.
        if not VolumeController.check_dependencies():
            logger.warning(
                "[SystemTools] Volume control dependencies missing; cannot set volume."
            )
            return False

        volume_controller = VolumeController()
        await asyncio.to_thread(volume_controller.set_volume, volume)
        logger.info(f"[SystemTools] Volume set successfully: {volume}")
        return True

    except KeyError:
        logger.error("[SystemTools] Missing volume parameter.")
        return False
    except Exception as e:
        logger.error(f"[SystemTools] Failed to set volume: {e}", exc_info=True)
        return False


async def get_volume(args: Dict[str, Any]) -> int:
    """
    Get current volume.
    """
    try:
        logger.info("[SystemTools] Getting current volume.")

        # Use VolumeController directly.
        from src.utils.volume_controller import VolumeController

        # Check dependencies and create controller.
        if not VolumeController.check_dependencies():
            logger.warning(
                "[SystemTools] Volume control dependencies missing; returning default."
            )
            return VolumeController.DEFAULT_VOLUME

        volume_controller = VolumeController()
        current_volume = await asyncio.to_thread(volume_controller.get_volume)
        logger.info(f"[SystemTools] Current volume: {current_volume}")
        return current_volume

    except Exception as e:
        logger.error(f"[SystemTools] Failed to get volume: {e}", exc_info=True)
        from src.utils.volume_controller import VolumeController

        return VolumeController.DEFAULT_VOLUME


async def _get_audio_status() -> Dict[str, Any]:
    """
    Get audio status.
    """
    try:
        from src.utils.volume_controller import VolumeController

        if VolumeController.check_dependencies():
            volume_controller = VolumeController()
            # 使用线程池获取音量，避免阻塞
            current_volume = await asyncio.to_thread(volume_controller.get_volume)
            return {
                "volume": current_volume,
                "muted": current_volume == 0,
                "available": True,
            }
        else:
            return {
                "volume": 50,
                "muted": False,
                "available": False,
                "reason": "Dependencies not available",
            }

    except Exception as e:
        logger.warning(f"[SystemTools] Failed to get audio status: {e}")
        return {"volume": 50, "muted": False, "available": False, "error": str(e)}


def _get_application_status() -> Dict[str, Any]:
    """
    Get application status information.
    """
    try:
        from src.application import Application
        from src.iot.thing_manager import ThingManager

        app = Application.get_instance()
        thing_manager = ThingManager.get_instance()

        # DeviceState的值直接是字符串，不需要访问.name属性
        device_state = str(app.get_device_state())
        iot_count = len(thing_manager.things) if thing_manager else 0

        return {
            "device_state": device_state,
            "iot_devices": iot_count,
        }

    except Exception as e:
        logger.warning(f"[SystemTools] Failed to get application status: {e}")
        return {"device_state": "unknown", "iot_devices": 0, "error": str(e)}
