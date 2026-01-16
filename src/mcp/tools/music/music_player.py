import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests

from src.audio_codecs.music_decoder import MusicDecoder
from src.constants.constants import AudioConfig
from src.utils.logging_config import get_logger
from src.utils.resource_finder import get_user_cache_dir

# Try to import music metadata library.
try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3NoHeaderError

    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

logger = get_logger(__name__)


class MusicMetadata:
    """
    Music metadata class.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.filename = file_path.name
        self.file_id = file_path.stem  # File name without extension (song ID).
        self.file_size = file_path.stat().st_size

        # Metadata extracted from file.
        self.title = None
        self.artist = None
        self.album = None
        self.duration = None  # Seconds.

    def extract_metadata(self) -> bool:
        """
        Extract music file metadata.
        """
        if not MUTAGEN_AVAILABLE:
            return False

        try:
            audio_file = MutagenFile(self.file_path)
            if audio_file is None:
                return False

            # Basic info.
            if hasattr(audio_file, "info"):
                self.duration = getattr(audio_file.info, "length", None)

            # ID3 tag info.
            tags = audio_file.tags if audio_file.tags else {}

            # Title.
            self.title = self._get_tag_value(tags, ["TIT2", "TITLE", "\xa9nam"])

            # Artist.
            self.artist = self._get_tag_value(tags, ["TPE1", "ARTIST", "\xa9ART"])

            # Album.
            self.album = self._get_tag_value(tags, ["TALB", "ALBUM", "\xa9alb"])

            return True

        except ID3NoHeaderError:
            # No ID3 tags; not an error.
            return True
        except Exception as e:
            logger.debug(f"Failed to extract metadata {self.filename}: {e}")
            return False

    def _get_tag_value(self, tags: dict, tag_names: List[str]) -> Optional[str]:
        """
        Get value from multiple possible tag names.
        """
        for tag_name in tag_names:
            if tag_name in tags:
                value = tags[tag_name]
                if isinstance(value, list) and value:
                    return str(value[0])
                elif value:
                    return str(value)
        return None

    def format_duration(self) -> str:
        """
        Format duration.
        """
        if self.duration is None:
            return "Unknown"

        minutes = int(self.duration) // 60
        seconds = int(self.duration) % 60
        return f"{minutes:02d}:{seconds:02d}"


class MusicPlayer:
    def __init__(self):
        # FFmpeg decoder and playback queue.
        self.decoder: Optional[MusicDecoder] = None
        self._music_queue: Optional[asyncio.Queue] = None
        self._playback_task: Optional[asyncio.Task] = None

        # Core playback state.
        self.current_song = ""
        self.current_url = ""
        self.song_id = ""
        self.total_duration = 0
        self.is_playing = False
        self.paused = False
        self.current_position = 0
        self.start_play_time = 0
        self._pause_source: Optional[str] = None  # "tts" | "manual" | None

        # Current playback file path (for pause/resume).
        self._current_file_path: Optional[Path] = None

        # Deferred start: wait for TTS to finish.
        self._deferred_start_path: Optional[Path] = None
        self._deferred_start_position: float = 0.0

        # Lyrics.
        self.lyrics = []  # Lyrics list [(time, text), ...]
        self.current_lyric_index = -1  # Current lyric index.

        # Cache directory setup (use user cache dir for write access).
        user_cache_dir = get_user_cache_dir()
        self.cache_dir = user_cache_dir / "music"
        self.temp_cache_dir = self.cache_dir / "temp"
        self._init_cache_dirs()

        # API configuration.
        self.config = {
            "SEARCH_URL": "http://search.kuwo.cn/r.s",
            "PLAY_URL": "http://api.xiaodaokg.com/kuwo.php",
            "LYRIC_URL": "https://api.xiaodaokg.com/kw/kwlyric.php",
            "HEADERS": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " "AppleWebKit/537.36"
                ),
                "Accept": "*/*",
                "Connection": "keep-alive",
            },
        }

        # Clean temp cache.
        self._clean_temp_cache()

        # Get application instance and AudioCodec.
        self.app = None
        self.audio_codec = None
        self._initialize_app_reference()

        # Local playlist cache.
        self._local_playlist = None
        self._last_scan_time = 0

        logger.info("Music player singleton initialized (FFmpeg + AudioCodec).")

    def _initialize_app_reference(self):
        """
        Initialize application reference and AudioCodec.
        """
        try:
            from src.application import Application

            self.app = Application.get_instance()
            self.audio_codec = getattr(self.app, "audio_codec", None)

            if not self.audio_codec:
                logger.warning("AudioCodec not initialized; playback may be unavailable.")

        except Exception as e:
            logger.warning(f"Failed to get Application instance: {e}")
            self.app = None

    def _init_cache_dirs(self):
        """
        Initialize cache directories.
        """
        try:
            # Create main cache directory.
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Create temp cache directory.
            self.temp_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Music cache directory initialized: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            # Fallback to system temp directory.
            self.cache_dir = Path(tempfile.gettempdir()) / "xiaozhi_music_cache"
            self.temp_cache_dir = self.cache_dir / "temp"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.temp_cache_dir.mkdir(parents=True, exist_ok=True)

    def _clean_temp_cache(self):
        """
        Clean temp cache files.
        """
        try:
            # Clear all files in temp cache directory.
            for file_path in self.temp_cache_dir.glob("*"):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        logger.debug(f"Deleted temp cache file: {file_path.name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete temp cache file: {file_path.name}, {e}"
                    )

            logger.info("Temp music cache cleaned.")
        except Exception as e:
            logger.error(f"Failed to clean temp cache directory: {e}")

    def _scan_local_music(self, force_refresh: bool = False) -> List[MusicMetadata]:
        """
        Scan local music cache and return playlist.
        """
        current_time = time.time()

        # Return cached playlist if not forcing refresh and cache is fresh (5 min).
        if (
            not force_refresh
            and self._local_playlist is not None
            and (current_time - self._last_scan_time) < 300
        ):
            return self._local_playlist

        playlist = []

        if not self.cache_dir.exists():
            logger.warning(f"Cache directory does not exist: {self.cache_dir}")
            return playlist

        # Find all music files.
        music_files = []
        for pattern in ["*.mp3", "*.m4a", "*.flac", "*.wav", "*.ogg"]:
            music_files.extend(self.cache_dir.glob(pattern))

        logger.debug(f"Found {len(music_files)} music files.")

        # Scan each file.
        for file_path in music_files:
            try:
                metadata = MusicMetadata(file_path)

                # Try extracting metadata.
                if MUTAGEN_AVAILABLE:
                    metadata.extract_metadata()

                playlist.append(metadata)

            except Exception as e:
                logger.debug(f"Failed to process music file {file_path.name}: {e}")

        # Sort by artist and title.
        playlist.sort(key=lambda x: (x.artist or "Unknown", x.title or x.filename))

        # Update cache.
        self._local_playlist = playlist
        self._last_scan_time = current_time

        logger.info(f"Scan complete, found {len(playlist)} local tracks.")
        return playlist

    async def get_local_playlist(self, force_refresh: bool = False) -> dict:
        """
        Get local music playlist.
        """
        try:
            playlist = self._scan_local_music(force_refresh)

            if not playlist:
                return {
                    "status": "info",
                    "message": "No local music files in cache.",
                    "playlist": [],
                    "total_count": 0,
                }

            # Format playlist for easy reading.
            formatted_playlist = []
            for metadata in playlist:
                title = metadata.title or "Unknown title"
                artist = metadata.artist or "Unknown artist"
                song_info = f"{title} - {artist}"
                formatted_playlist.append(song_info)

            return {
                "status": "success",
                "message": f"Found {len(playlist)} local tracks.",
                "playlist": formatted_playlist,
                "total_count": len(playlist),
            }

        except Exception as e:
            logger.error(f"Failed to get local playlist: {e}")
            return {
                "status": "error",
                "message": f"Failed to get local playlist: {str(e)}",
                "playlist": [],
                "total_count": 0,
            }

    async def search_local_music(self, query: str) -> dict:
        """
        Search local music.
        """
        try:
            playlist = self._scan_local_music()

            if not playlist:
                return {
                    "status": "info",
                    "message": "No local music files in cache.",
                    "results": [],
                    "found_count": 0,
                }

            query = query.lower()
            results = []

            for metadata in playlist:
                # Search in title, artist, filename.
                searchable_text = " ".join(
                    filter(
                        None,
                        [
                            metadata.title,
                            metadata.artist,
                            metadata.album,
                            metadata.filename,
                        ],
                    )
                ).lower()

                if query in searchable_text:
                    title = metadata.title or "Unknown title"
                    artist = metadata.artist or "Unknown artist"
                    song_info = f"{title} - {artist}"
                    results.append(
                        {
                            "song_info": song_info,
                            "file_id": metadata.file_id,
                            "duration": metadata.format_duration(),
                        }
                    )

            return {
                "status": "success",
                "message": f"Found {len(results)} matching local tracks.",
                "results": results,
                "found_count": len(results),
            }

        except Exception as e:
            logger.error(f"Failed to search local music: {e}")
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "results": [],
                "found_count": 0,
            }

    async def play_local_song_by_id(self, file_id: str) -> dict:
        """
        Play a local song by file ID.
        """
        try:
            # Build file path.
            file_path = self.cache_dir / f"{file_id}.mp3"

            if not file_path.exists():
                # Try other formats.
                for ext in [".m4a", ".flac", ".wav", ".ogg"]:
                    alt_path = self.cache_dir / f"{file_id}{ext}"
                    if alt_path.exists():
                        file_path = alt_path
                        break
                else:
                    return {
                        "status": "error",
                        "message": f"Local file not found: {file_id}",
                    }

            # Get song info.
            metadata = MusicMetadata(file_path)
            if MUTAGEN_AVAILABLE:
                metadata.extract_metadata()

            # Update song info.
            title = metadata.title or "Unknown title"
            artist = metadata.artist or "Unknown artist"
            self.current_song = f"{title} - {artist}"
            self.song_id = file_id
            self.total_duration = metadata.duration or 0
            self.current_url = str(file_path)  # Local file path.
            self.lyrics = []  # Lyrics not supported for local files.

            # Start playback.
            success = await self._start_playback(file_path)

            if success:
                # Return song info with duration.
                duration_str = self._format_time(self.total_duration)
                return {
                    "status": "success",
                    "message": f"Now playing: {self.current_song}",
                    "song": self.current_song,
                    "duration": duration_str,
                    "total_seconds": self.total_duration,
                }
            else:
                return {"status": "error", "message": "Playback failed"}

        except Exception as e:
            logger.error(f"Failed to play local music: {e}")
            return {"status": "error", "message": f"Playback failed: {str(e)}"}

    # Internal helpers: position and progress.
    async def get_position(self):
        if not self.is_playing or self.paused:
            return self.current_position

        current_pos = min(self.total_duration, time.time() - self.start_play_time)

        # Check if playback finished.
        if current_pos >= self.total_duration and self.total_duration > 0:
            await self._handle_playback_finished()

        return current_pos

    async def get_progress(self):
        """
        Get playback progress percentage.
        """
        if self.total_duration <= 0:
            return 0
        position = await self.get_position()
        return round(position * 100 / self.total_duration, 1)

    async def _handle_playback_finished(self):
        """
        Handle playback completion.
        """
        if self.is_playing:
            logger.info(f"Song playback finished: {self.current_song}")
            # Stop decoder.
            if self.decoder:
                await self.decoder.stop()
                self.decoder = None

            self.is_playing = False
            self.paused = False
            self.current_position = self.total_duration

            # Update UI with completion status.
            if self.app and hasattr(self.app, "set_chat_message"):
                dur_str = self._format_time(self.total_duration)
                await self._safe_update_ui(
                    f"Playback completed: {self.current_song} [{dur_str}]"
                )

    # Core methods.
    async def search_and_play(self, song_name: str) -> dict:
        """
        Search and play a song.
        """
        try:
            # Search for song.
            song_id, url = await self._search_song(song_name)
            if not song_id or not url:
                return {"status": "error", "message": f"Song not found: {song_name}"}

            # Play song.
            success = await self._play_url(url)
            if success:
                # Return song info with duration.
                duration_str = self._format_time(self.total_duration)
                return {
                    "status": "success",
                    "message": f"Now playing: {self.current_song}",
                    "song": self.current_song,
                    "duration": duration_str,
                    "total_seconds": self.total_duration,
                }
            else:
                return {"status": "error", "message": "Playback failed"}

        except Exception as e:
            logger.error(f"Search and play failed: {e}")
            return {"status": "error", "message": f"Operation failed: {str(e)}"}

    async def stop(self) -> dict:
        """
        Stop playback.
        """
        try:
            if not self.is_playing and not self._pending_play:
                return {"status": "info", "message": "No song is playing."}

            current_song = self.current_song

            # Stop decoder.
            if self.decoder:
                await self.decoder.stop()
                self.decoder = None

            # Cancel playback task.
            if self._playback_task and not self._playback_task.done():
                self._playback_task.cancel()
                try:
                    await self._playback_task
                except asyncio.CancelledError:
                    pass

            # Clear queue.
            if self._music_queue:
                while not self._music_queue.empty():
                    try:
                        self._music_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            # Reset state.
            self.is_playing = False
            self.paused = False
            self._pause_source = None  # Clear pause source.
            self._pending_play = False
            self._pending_file_path = None
            self.current_position = 0

            # Update UI.
            if self.app and hasattr(self.app, "set_chat_message"):
                await self._safe_update_ui(f"Stopped: {current_song}")

            logger.info(f"Stopped playback: {current_song}")
            return {"status": "success", "message": "Stopped."}

        except Exception as e:
            logger.error(f"Failed to stop playback: {e}")
            return {"status": "error", "message": f"Stop failed: {str(e)}"}

    async def pause(self, source: str = "manual") -> dict:
        """Pause playback (stop decoder only, keep queue).

        Args:
            source: Pause source, "manual" user pause, "tts" TTS pause
        """
        try:
            if not self.is_playing:
                return {"status": "info", "message": "No song is playing."}

            if self.paused:
                # If paused with different source, update source (user intent wins).
                if self._pause_source != source:
                    old_source = self._pause_source
                    self._pause_source = source
                    logger.info(f"Updated pause source: {old_source} → {source}")
                return {"status": "info", "message": "Already paused."}

            # Set pause flag immediately to avoid duplicate calls.
            self.paused = True
            self._pause_source = source

            # Record playback position at pause.
            if self.start_play_time > 0:
                self.current_position = time.time() - self.start_play_time

            # Stop decoder (stop generating new data).
            if self.decoder:
                await self.decoder.stop()
                self.decoder = None

            # Wait for decoder to stop.
            await asyncio.sleep(0.05)

            # Clear internal music queue (keep AudioCodec shared queue).
            cleared_count = 0
            if self._music_queue:
                while not self._music_queue.empty():
                    try:
                        self._music_queue.get_nowait()
                        cleared_count += 1
                    except asyncio.QueueEmpty:
                        break

            logger.info(
                "Paused playback: %s at %s, source: %s, cleared %s internal frames",
                self.current_song,
                self._format_time(self.current_position),
                source,
                cleared_count,
            )

            return {"status": "success", "message": "Paused."}

        except Exception as e:
            logger.error(f"Failed to pause playback: {e}", exc_info=True)
            return {"status": "error", "message": f"Pause failed: {str(e)}"}

    async def resume(self) -> dict:
        """
        Resume playback (restart decode from pause position).
        """
        try:
            if not self.is_playing:
                return {"status": "info", "message": "No song is playing."}

            if not self.paused:
                return {"status": "info", "message": "Not paused."}

            if not self._current_file_path or not self._current_file_path.exists():
                return {"status": "error", "message": "Audio file not found."}

            # Restart decoding and playback from pause position.
            logger.info(
                f"Resuming playback: {self.current_song} from {self._format_time(self.current_position)}"
            )

            # Recreate music queue.
            self._music_queue = asyncio.Queue(maxsize=100)

            # Restart FFmpeg decoder from pause position.
            self.decoder = MusicDecoder(
                sample_rate=AudioConfig.OUTPUT_SAMPLE_RATE,
                channels=AudioConfig.CHANNELS,
            )

            success = await self.decoder.start_decode(
                self._current_file_path, self._music_queue, self.current_position
            )
            if not success:
                logger.error("Failed to restart decoder.")
                return {"status": "error", "message": "Resume failed."}

            # Cancel old playback task if present.
            if self._playback_task and not self._playback_task.done():
                self._playback_task.cancel()
                try:
                    await self._playback_task
                except asyncio.CancelledError:
                    pass

            # Start new playback task.
            self._playback_task = asyncio.create_task(self._playback_loop())

            # Restore state.
            self.paused = False
            self._pause_source = None  # Clear pause source.
            self.start_play_time = time.time() - self.current_position  # Adjust baseline.

            # Update UI.
            if self.app and hasattr(self.app, "set_chat_message"):
                await self._safe_update_ui(f"Resumed: {self.current_song}")

            return {"status": "success", "message": "Resumed playback."}

        except Exception as e:
            logger.error(f"Failed to resume playback: {e}")
            return {"status": "error", "message": f"Resume failed: {str(e)}"}

    async def seek(self, position: float) -> dict:
        """
        Seek to a position (by re-decoding).
        """
        try:
            if not self.is_playing:
                return {"status": "error", "message": "No song is playing."}

            if not self._current_file_path or not self._current_file_path.exists():
                return {"status": "error", "message": "Audio file not found."}

            # Clamp seek range.
            if position < 0:
                position = 0
            elif position >= self.total_duration:
                position = max(0, self.total_duration - 1)

            # Stop current playback immediately.
            # Stop decoder.
            if self.decoder:
                await self.decoder.stop()
                self.decoder = None

            # Wait for decoder to stop.
            await asyncio.sleep(0.05)

            # Clear music queue.
            cleared_count = 0
            if self._music_queue:
                while not self._music_queue.empty():
                    try:
                        self._music_queue.get_nowait()
                        cleared_count += 1
                    except asyncio.QueueEmpty:
                        break

            # Clear AudioCodec playback queue.
            if self.audio_codec:
                await self.audio_codec.clear_audio_queue()

            logger.info(
                f"Seek to {self._format_time(position)}, cleared {cleared_count} frames"
            )

            # Restart playback from new position.
            success = await self._start_playback(self._current_file_path, position)

            if success:
                return {
                    "status": "success",
                    "message": f"Seeked to {self._format_time(position)}",
                }
            else:
                return {"status": "error", "message": "Seek failed"}

        except Exception as e:
            logger.error(f"Seek failed: {e}", exc_info=True)
            return {"status": "error", "message": f"Seek failed: {str(e)}"}

    async def get_lyrics(self) -> dict:
        """
        Get current song lyrics.
        """
        if not self.lyrics:
            return {"status": "info", "message": "No lyrics for this song.", "lyrics": []}

        # Extract lyrics text into list.
        lyrics_text = []
        for time_sec, text in self.lyrics:
            time_str = self._format_time(time_sec)
            lyrics_text.append(f"[{time_str}] {text}")

        return {
            "status": "success",
            "message": f"Retrieved {len(self.lyrics)} lines of lyrics.",
            "lyrics": lyrics_text,
        }

    async def get_status(self) -> dict:
        """Get player status.

        Note: Return user-visible status only. TTS pauses should not be reported as
        user pauses.
        """
        position = await self.get_position()
        progress = await self.get_progress()

        # Determine actual playback state (exclude TTS pauses).
        if not self.is_playing:
            playing_state = "Not playing"
        elif self.paused and self._pause_source == "manual":
            # Only report pause if user initiated.
            playing_state = "Paused"
        elif self.is_playing:
            playing_state = "Playing"
        else:
            playing_state = "Unknown"

        duration_str = self._format_time(self.total_duration)
        position_str = self._format_time(position)

        return {
            "status": "success",
            "message": (
                f"Current song: {self.current_song}\n"
                f"Playback state: {playing_state}\n"
                f"Pause source: {self._pause_source} (tts = temporary TTS pause)\n"
                f"Duration: {duration_str}\n"
                f"Position: {position_str}\n"
                f"Progress: {progress}%\n"
                f"Lyrics available: {'yes' if len(self.lyrics) > 0 else 'no'}"
            ),
        }

    # Internal methods.
    async def _search_song(self, song_name: str) -> Tuple[str, str]:
        """
        Search song and get ID/URL.
        """
        try:
            # Build search params.
            params = {
                "all": song_name,
                "ft": "music",
                "newsearch": "1",
                "alflac": "1",
                "itemset": "web_2013",
                "client": "kt",
                "cluster": "0",
                "pn": "0",
                "rn": "1",
                "vermerge": "1",
                "rformat": "json",
                "encoding": "utf8",
                "show_copyright_off": "1",
                "pcmp4": "1",
                "ver": "mbox",
                "vipver": "MUSIC_8.7.6.0.BCS31",
                "plat": "pc",
                "devid": "0",
            }

            # Search for song.
            response = await asyncio.to_thread(
                requests.get,
                self.config["SEARCH_URL"],
                params=params,
                headers=self.config["HEADERS"],
                timeout=10,
            )
            response.raise_for_status()

            # Parse response.
            text = response.text.replace("'", '"')

            # Extract song ID.
            song_id = self._extract_value(text, '"DC_TARGETID":"', '"')
            if not song_id:
                return "", ""

            # Extract song info.
            title = self._extract_value(text, '"NAME":"', '"') or song_name
            artist = self._extract_value(text, '"ARTIST":"', '"')
            album = self._extract_value(text, '"ALBUM":"', '"')
            duration_str = self._extract_value(text, '"DURATION":"', '"')

            if duration_str:
                try:
                    self.total_duration = int(duration_str)
                except ValueError:
                    self.total_duration = 0

            # Set display name.
            display_name = title
            if artist:
                display_name = f"{title} - {artist}"
                if album:
                    display_name += f" ({album})"
            self.current_song = display_name
            self.song_id = song_id

            # Get playback URL.
            play_url = f"{self.config['PLAY_URL']}?ID={song_id}"
            url_response = await asyncio.to_thread(
                requests.get, play_url, headers=self.config["HEADERS"], timeout=10
            )
            url_response.raise_for_status()

            play_url_text = url_response.text.strip()
            if play_url_text and play_url_text.startswith("http"):
                # Fetch lyrics.
                await self._fetch_lyrics(song_id)
                return song_id, play_url_text

            return song_id, ""

        except Exception as e:
            logger.error(f"Failed to search song: {e}")
            return "", ""

    async def _play_url(self, url: str) -> bool:
        """
        Play a URL (FFmpeg + AudioCodec).
        """
        try:
            # Check AudioCodec availability.
            if not self.audio_codec:
                logger.error("AudioCodec not initialized; cannot play music.")
                return False

            # Stop current playback.
            if self.is_playing:
                await self.stop()

            # Check cache or download.
            file_path = await self._get_or_download_file(url)
            if not file_path:
                return False

            # Start playback.
            return await self._start_playback(file_path)

        except Exception as e:
            logger.error(f"Playback failed: {e}")
            return False

    async def _start_playback(
        self, file_path: Path, start_position: float = 0.0
    ) -> bool:
        """Start playback (internal).

        Args:
            file_path: Audio file path
            start_position: Start position in seconds
        """
        try:
            # Check TTS state: defer start while TTS is speaking.
            if self.app and self.app.is_speaking():
                logger.info("TTS speaking; deferring music start.")
                self._deferred_start_path = file_path
                self._deferred_start_position = start_position
                # Mark as ready to play (AudioPlugin will resume later).
                self.is_playing = True
                self.paused = True
                return True

            # Clear deferred start flags.
            self._deferred_start_path = None
            self._deferred_start_position = 0.0

            # Save current file path (pause/resume).
            self._current_file_path = file_path

            # Create music queue.
            self._music_queue = asyncio.Queue(maxsize=100)

            # Start FFmpeg decoder (supports start position).
            self.decoder = MusicDecoder(
                sample_rate=AudioConfig.OUTPUT_SAMPLE_RATE,  # 24000Hz
                channels=AudioConfig.CHANNELS,  # 1 channel
            )

            success = await self.decoder.start_decode(
                file_path, self._music_queue, start_position
            )
            if not success:
                logger.error("Failed to start audio decoder.")
                return False

            # Start playback task.
            self._playback_task = asyncio.create_task(self._playback_loop())

            # Update playback state.
            self.is_playing = True
            self.paused = False
            self._pending_play = False
            self.current_position = start_position  # Start from position.
            self.start_play_time = time.time() - start_position  # Adjust baseline.
            self.current_lyric_index = -1

            position_info = f" from {start_position:.1f}s" if start_position > 0 else ""
            logger.info(f"Starting playback: {self.current_song}{position_info}")

            # Update UI.
            if self.app and hasattr(self.app, "set_chat_message"):
                await self._safe_update_ui(f"Now playing: {self.current_song}")

            # Start lyrics update task.
            asyncio.create_task(self._lyrics_update_task())

            return True

        except Exception as e:
            logger.error(f"Failed to start playback: {e}")
            return False

    async def _playback_loop(self):
        """
        Playback loop: read PCM from queue and write to AudioCodec.
        """
        try:
            while self.is_playing:
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue

                # Read from music queue.
                try:
                    audio_data = await asyncio.wait_for(
                        self._music_queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Music queue read timed out.")
                    continue

                if audio_data is None:
                    # EOF, playback finished.
                    logger.info("Music playback finished.")
                    await self._handle_playback_finished()
                    break

                # Write to AudioCodec playback queue.
                await self._write_to_audio_codec(audio_data)

        except asyncio.CancelledError:
            logger.debug("Playback loop canceled.")
        except Exception as e:
            logger.error(f"Playback loop error: {e}", exc_info=True)

    async def _write_to_audio_codec(self, pcm_data: np.ndarray):
        """
        Write PCM data to AudioCodec playback queue.
        """
        try:
            if not self.audio_codec:
                logger.error("AudioCodec not initialized.")
                return

            # Ensure mono data.
            if pcm_data.ndim > 1:
                # Stereo to mono (average).
                pcm_data = pcm_data.mean(axis=1).astype(np.int16)

            # Call AudioCodec write_pcm_direct.
            await self.audio_codec.write_pcm_direct(pcm_data)

        except Exception as e:
            logger.error(f"Failed to write to AudioCodec: {e}", exc_info=True)

    async def _get_or_download_file(self, url: str) -> Optional[Path]:
        """Get or download file.

        Check cache first, download if missing.
        """
        try:
            # Use song ID as cache filename.
            cache_filename = f"{self.song_id}.mp3"
            cache_path = self.cache_dir / cache_filename

            # Check cache.
            if cache_path.exists():
                logger.info(f"Using cache: {cache_path}")
                return cache_path

            # Download if not cached.
            return await self._download_file(url, cache_filename)

        except Exception as e:
            logger.error(f"Failed to get file: {e}")
            return None

    async def _download_file(self, url: str, filename: str) -> Optional[Path]:
        """Download file to cache directory.

        Download to temp directory then move to cache.
        """
        temp_path = None
        try:
            # Create temp file path.
            temp_path = self.temp_cache_dir / f"temp_{int(time.time())}_{filename}"

            # Download asynchronously.
            response = await asyncio.to_thread(
                requests.get,
                url,
                headers=self.config["HEADERS"],
                stream=True,
                timeout=30,
            )
            response.raise_for_status()

            # Write temp file.
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Move to cache after download.
            cache_path = self.cache_dir / filename
            shutil.move(str(temp_path), str(cache_path))

            logger.info(f"Music downloaded and cached: {cache_path}")
            return cache_path

        except Exception as e:
            logger.error(f"Download failed: {e}")
            # Clean up temp file.
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned temp download file: {temp_path}")
                except Exception:
                    pass
            return None

    async def _fetch_lyrics(self, song_id: str):
        """
        Fetch lyrics.
        """
        try:
            # Reset lyrics.
            self.lyrics = []

            # Build lyric API request.
            lyric_url = self.config.get("LYRIC_URL")
            lyric_api_url = f"{lyric_url}?id={song_id}"
            logger.info(f"Lyric URL: {lyric_api_url}")

            response = await asyncio.to_thread(
                requests.get, lyric_api_url, headers=self.config["HEADERS"], timeout=10
            )
            response.raise_for_status()

            # Parse JSON.
            data = response.json()

            # Parse lyrics.
            if (
                data.get("code") == 200
                and data.get("data")
                and data["data"].get("content")
            ):
                lrc_content = data["data"]["content"]

                # Parse LRC lyrics.
                lines = lrc_content.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Match timestamp format [mm:ss.xx].
                    import re

                    time_match = re.match(r"\[(\d{2}):(\d{2})\.(\d{2})\](.+)", line)
                    if time_match:
                        minutes = int(time_match.group(1))
                        seconds = int(time_match.group(2))
                        centiseconds = int(time_match.group(3))
                        text = time_match.group(4).strip()

                        # Convert to total seconds.
                        time_sec = minutes * 60 + seconds + centiseconds / 100.0

                        # TODO(i18n): Keep Chinese lyric metadata prefixes for filtering.
                        # Skip empty and metadata lines.
                        if (
                            text
                            and not text.startswith("作词")
                            and not text.startswith("作曲")
                            and not text.startswith("编曲")
                            and not text.startswith("ti:")
                            and not text.startswith("ar:")
                            and not text.startswith("al:")
                            and not text.startswith("by:")
                            and not text.startswith("offset:")
                        ):
                            self.lyrics.append((time_sec, text))

                logger.info(f"Fetched lyrics successfully, {len(self.lyrics)} lines.")
            else:
                logger.warning(
                    f"No lyrics or invalid format: {data.get('msg', '')}"
                )

        except Exception as e:
            logger.error(f"Failed to fetch lyrics: {e}")

    async def _lyrics_update_task(self):
        """
        Lyrics update task.
        """
        if not self.lyrics:
            return

        try:
            while self.is_playing:
                if self.paused:
                    await asyncio.sleep(0.5)
                    continue

                current_time = time.time() - self.start_play_time

                # Check for completion.
                if current_time >= self.total_duration:
                    await self._handle_playback_finished()
                    break

                # Find lyric for current time.
                current_index = self._find_current_lyric_index(current_time)

                # Update display on lyric change.
                if current_index != self.current_lyric_index:
                    await self._display_current_lyric(current_index)

                await asyncio.sleep(0.2)
        except Exception as e:
            logger.error(f"Lyrics update task error: {e}")

    def _find_current_lyric_index(self, current_time: float) -> int:
        """
        Find lyric index for current time.
        """
        # Find next lyric line.
        next_lyric_index = None
        for i, (time_sec, _) in enumerate(self.lyrics):
            # Add small offset (0.5s) for accuracy.
            if time_sec > current_time - 0.5:
                next_lyric_index = i
                break

        # Determine current lyric index.
        if next_lyric_index is not None and next_lyric_index > 0:
            # Current lyric is the previous line.
            return next_lyric_index - 1
        elif next_lyric_index is None and self.lyrics:
            # No next line; we're at the last lyric.
            return len(self.lyrics) - 1
        else:
            # Other cases (e.g., playback start).
            return 0

    async def _display_current_lyric(self, current_index: int):
        """
        Display current lyric.
        """
        self.current_lyric_index = current_index

        if current_index < len(self.lyrics):
            time_sec, text = self.lyrics[current_index]

            # Add time and progress info.
            position_str = self._format_time(time.time() - self.start_play_time)
            duration_str = self._format_time(self.total_duration)
            display_text = f"[{position_str}/{duration_str}] {text}"

            # Update UI.
            if self.app and hasattr(self.app, "set_chat_message"):
                await self._safe_update_ui(display_text)
                logger.debug(f"Displaying lyric: {text}")

    def _extract_value(self, text: str, start_marker: str, end_marker: str) -> str:
        """
        Extract value from text.
        """
        start_pos = text.find(start_marker)
        if start_pos == -1:
            return ""

        start_pos += len(start_marker)
        end_pos = text.find(end_marker, start_pos)

        if end_pos == -1:
            return ""

        return text[start_pos:end_pos]

    def _format_time(self, seconds: float) -> str:
        """
        Format seconds as mm:ss.
        """
        minutes = int(seconds) // 60
        seconds = int(seconds) % 60
        return f"{minutes:02d}:{seconds:02d}"

    async def _safe_update_ui(self, message: str):
        """
        Safely update UI.
        """
        if not self.app or not hasattr(self.app, "set_chat_message"):
            return

        try:
            self.app.set_chat_message("assistant", message)
        except Exception as e:
            logger.error(f"Failed to update UI: {e}")

    def __del__(self):
        """
        Clean up resources.
        """
        try:
            # Clean temp cache on normal exit.
            self._clean_temp_cache()
        except Exception:
            # Ignore errors during destruction.
            pass


# Global music player instance.
_music_player_instance = None


def get_music_player_instance() -> MusicPlayer:
    """
    Get music player singleton.
    """
    global _music_player_instance
    if _music_player_instance is None:
        _music_player_instance = MusicPlayer()
        logger.info("[MusicPlayer] Created music player singleton.")
    return _music_player_instance
