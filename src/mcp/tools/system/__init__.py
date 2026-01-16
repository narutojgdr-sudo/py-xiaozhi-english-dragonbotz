"""System tools package.

Provides system management functions including device status and audio control.
"""

from .manager import SystemToolsManager, get_system_tools_manager
from .tools import get_volume, set_volume

__all__ = [
    "SystemToolsManager",
    "get_system_tools_manager",
    "set_volume",
    "get_volume",
]
