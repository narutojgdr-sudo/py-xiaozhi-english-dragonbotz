"""Music tools manager.

Initializes, configures, and registers MCP music tools.
"""

from typing import Any, Dict

from src.utils.logging_config import get_logger

from .music_player import get_music_player_instance

logger = get_logger(__name__)


class MusicToolsManager:
    """
    Music tools manager.
    """

    def __init__(self):
        """
        Initialize music tools manager.
        """
        self._initialized = False
        self._music_player = None
        logger.info("[MusicManager] Music tools manager initialized.")

    def init_tools(self, add_tool, PropertyList, Property, PropertyType):
        """
        Initialize and register all music tools.
        """
        try:
            logger.info("[MusicManager] Registering music tools.")

            # Get music player singleton.
            self._music_player = get_music_player_instance()

            # Register search and play tool.
            self._register_search_and_play_tool(
                add_tool, PropertyList, Property, PropertyType
            )

            # Register pause tool.
            self._register_pause_tool(add_tool, PropertyList)

            # Register resume tool.
            self._register_resume_tool(add_tool, PropertyList)

            # Register stop tool.
            self._register_stop_tool(add_tool, PropertyList)

            # Register seek tool.
            self._register_seek_tool(add_tool, PropertyList, Property, PropertyType)

            # Register get lyrics tool.
            self._register_get_lyrics_tool(add_tool, PropertyList)

            # Register local playlist tool.
            self._register_get_local_playlist_tool(
                add_tool, PropertyList, Property, PropertyType
            )

            self._initialized = True
            logger.info("[MusicManager] Music tools registered.")

        except Exception as e:
            logger.error(
                f"[MusicManager] Music tool registration failed: {e}", exc_info=True
            )
            raise

    def _register_search_and_play_tool(
        self, add_tool, PropertyList, Property, PropertyType
    ):
        """
        Register search and play tool.
        """

        async def search_and_play_wrapper(args: Dict[str, Any]) -> str:
            song_name = args.get("song_name", "")
            result = await self._music_player.search_and_play(song_name)
            return result.get("message", "Search and play completed.")

        search_props = PropertyList([Property("song_name", PropertyType.STRING)])

        add_tool(
            (
                "music_player.search_and_play",
                "Search for a song by name and play it. Stops current playback "
                "and starts the requested track.",
                search_props,
                search_and_play_wrapper,
            )
        )
        logger.debug("[MusicManager] Registered search/play tool.")

    def _register_pause_tool(self, add_tool, PropertyList):
        """
        Register pause tool.
        """

        async def pause_wrapper(args: Dict[str, Any]) -> str:
            result = await self._music_player.pause()
            return result.get("message", "Paused.")

        add_tool(
            (
                "music_player.pause",
                "Pause current playback and keep position. Call resume to continue. "
                "Only use when the user explicitly asks to pause.",
                PropertyList(),
                pause_wrapper,
            )
        )
        logger.debug("[MusicManager] Registered pause tool.")

    def _register_resume_tool(self, add_tool, PropertyList):
        """
        Register resume tool.
        """

        async def resume_wrapper(args: Dict[str, Any]) -> str:
            result = await self._music_player.resume()
            return result.get("message", "Resumed playback.")

        add_tool(
            (
                "music_player.resume",
                "Resume playback from the paused position when user requested pause.",
                PropertyList(),
                resume_wrapper,
            )
        )
        logger.debug("[MusicManager] Registered resume tool.")

    def _register_stop_tool(self, add_tool, PropertyList):
        """
        Register stop tool.
        """

        async def stop_wrapper(args: Dict[str, Any]) -> str:
            result = await self._music_player.stop()
            return result.get("message", "Stopped playback.")

        add_tool(
            (
                "music_player.stop",
                "Stop music playback and reset position to the start.",
                PropertyList(),
                stop_wrapper,
            )
        )
        logger.debug("[MusicManager] Registered stop tool.")

    def _register_seek_tool(self, add_tool, PropertyList, Property, PropertyType):
        """
        Register seek tool.
        """

        async def seek_wrapper(args: Dict[str, Any]) -> str:
            position = args.get("position", 0)
            result = await self._music_player.seek(float(position))
            return result.get("message", "Seek completed.")

        seek_props = PropertyList(
            [Property("position", PropertyType.INTEGER, min_value=0)]
        )

        add_tool(
            (
                "music_player.seek",
                "Seek to a specific position in seconds from the start.",
                seek_props,
                seek_wrapper,
            )
        )
        logger.debug("[MusicManager] Registered seek tool.")

    def _register_get_lyrics_tool(self, add_tool, PropertyList):
        """
        Register get lyrics tool.
        """

        async def get_lyrics_wrapper(args: Dict[str, Any]) -> str:
            result = await self._music_player.get_lyrics()
            if result.get("status") == "success":
                lyrics = result.get("lyrics", [])
                return "Lyrics:\n" + "\n".join(lyrics)
            else:
                return result.get("message", "Failed to fetch lyrics.")

        add_tool(
            (
                "music_player.get_lyrics",
                "Get lyrics for the current song, including timestamps.",
                PropertyList(),
                get_lyrics_wrapper,
            )
        )
        logger.debug("[MusicManager] Registered get lyrics tool.")

    def _register_get_local_playlist_tool(
        self, add_tool, PropertyList, Property, PropertyType
    ):
        """
        Register local playlist tool.
        """

        async def get_local_playlist_wrapper(args: Dict[str, Any]) -> str:
            force_refresh = args.get("force_refresh", False)
            result = await self._music_player.get_local_playlist(force_refresh)

            if result.get("status") == "success":
                playlist = result.get("playlist", [])
                total_count = result.get("total_count", 0)

                if playlist:
                    playlist_text = f"Local playlist ({total_count} tracks):\n"
                    playlist_text += "\n".join(playlist)
                    return playlist_text
                else:
                    return "No local music files in cache."
            else:
                return result.get("message", "Failed to fetch local playlist.")

        refresh_props = PropertyList(
            [Property("force_refresh", PropertyType.BOOLEAN, default_value=False)]
        )

        add_tool(
            (
                "music_player.get_local_playlist",
                "List locally cached songs (format: 'Song - Artist').",
                refresh_props,
                get_local_playlist_wrapper,
            )
        )
        logger.debug("[MusicManager] Registered local playlist tool.")

    def _format_time(self, seconds: float) -> str:
        """
        Format seconds as mm:ss.
        """
        minutes = int(seconds) // 60
        seconds = int(seconds) % 60
        return f"{minutes:02d}:{seconds:02d}"

    def is_initialized(self) -> bool:
        """
        Check whether manager is initialized.
        """
        return self._initialized


# Global manager instance.
_music_tools_manager = None


def get_music_tools_manager() -> MusicToolsManager:
    """
    Get the music tools manager singleton.
    """
    global _music_tools_manager
    if _music_tools_manager is None:
        _music_tools_manager = MusicToolsManager()
        logger.debug("[MusicManager] Created music tools manager instance.")
    return _music_tools_manager
