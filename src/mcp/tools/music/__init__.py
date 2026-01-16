"""Music player tools.

Provides search, play, pause, stop, and seek controls.
"""

from .manager import MusicToolsManager, get_music_tools_manager
from .music_player import get_music_player_instance

__all__ = [
    "MusicToolsManager",
    "get_music_tools_manager",
    "get_music_player_instance",
]
