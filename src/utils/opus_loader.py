# Handle the Opus dynamic library before importing opuslib.
import ctypes
import os
import platform
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union, cast

# Get logger.
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# Platform constants.
class PLATFORM(Enum):
    WINDOWS = "windows"
    MACOS = "darwin"
    LINUX = "linux"


# Architecture constants.
class ARCH(Enum):
    WINDOWS = {"arm": "x64", "intel": "x64"}
    MACOS = {"arm": "arm64", "intel": "x64"}
    LINUX = {"arm": "arm64", "intel": "x64"}


# Dynamic library path constants.
class LIB_PATH(Enum):
    WINDOWS = "libs/libopus/win/x64"
    MACOS = "libs/libopus/mac/{arch}"
    LINUX = "libs/libopus/linux/{arch}"


# Dynamic library name constants.
class LIB_INFO(Enum):
    WINDOWS = {"name": "opus.dll", "system_name": ["opus"]}
    MACOS = {"name": "libopus.dylib", "system_name": ["libopus.dylib"]}
    LINUX = {"name": "libopus.so", "system_name": ["libopus.so.0", "libopus.so"]}


def get_platform() -> str:
    system = platform.system().lower()
    if system == "windows" or system.startswith("win"):
        system = PLATFORM.WINDOWS
    elif system == "darwin":
        system = PLATFORM.MACOS
    else:
        system = PLATFORM.LINUX
    return system


def get_arch(system: PLATFORM) -> str:
    architecture = platform.machine().lower()
    is_arm = "arm" in architecture or "aarch64" in architecture
    if system == PLATFORM.WINDOWS:
        arch_name = ARCH.WINDOWS.value["arm" if is_arm else "intel"]
    elif system == PLATFORM.MACOS:
        arch_name = ARCH.MACOS.value["arm" if is_arm else "intel"]
    else:
        arch_name = ARCH.LINUX.value["arm" if is_arm else "intel"]
    return architecture, arch_name


def get_lib_path(system: PLATFORM, arch_name: str):
    if system == PLATFORM.WINDOWS:
        lib_name = LIB_PATH.WINDOWS.value
    elif system == PLATFORM.MACOS:
        lib_name = LIB_PATH.MACOS.value.format(arch=arch_name)
    else:
        lib_name = LIB_PATH.LINUX.value.format(arch=arch_name)
    return lib_name


def get_lib_name(system: PLATFORM, local: bool = True) -> Union[str, List[str]]:
    """Get the library name.

    Args:
        system (PLATFORM): Platform
        local (bool, optional): Whether to get local name (str). If False, return a
            list of system names.

    Returns:
        str | List: Library name(s)
    """
    key = "name" if local else "system_name"
    if system == PLATFORM.WINDOWS:
        lib_name = LIB_INFO.WINDOWS.value[key]
    elif system == PLATFORM.MACOS:
        lib_name = LIB_INFO.MACOS.value[key]
    else:
        lib_name = LIB_INFO.LINUX.value[key]
    return lib_name


def get_system_info() -> Tuple[str, str]:
    """
    Get current system information.
    """
    # Normalize system name.
    system = get_platform()

    # Normalize architecture name.
    _, arch_name = get_arch(system)
    logger.info(f"Detected system: {system}, arch: {arch_name}")

    return system, arch_name


def get_search_paths(system: PLATFORM, arch_name: str) -> List[Tuple[Path, str]]:
    """
    Get library search paths (using the unified resource finder).
    """
    from .resource_finder import find_libs_dir, get_project_root

    lib_name = cast(str, get_lib_name(system))

    search_paths: List[Tuple[Path, str]] = []

    # Map system names to directory names.
    system_dir_map = {
        PLATFORM.WINDOWS: "win",
        PLATFORM.MACOS: "mac",
        PLATFORM.LINUX: "linux",
    }

    system_dir = system_dir_map.get(system)

    # First, look for platform- and arch-specific libs directory.
    if system_dir:
        specific_libs_dir = find_libs_dir(f"libopus/{system_dir}", arch_name)
        if specific_libs_dir:
            search_paths.append((specific_libs_dir, lib_name))
            logger.debug(
                f"Found platform/arch-specific libs directory: {specific_libs_dir}"
            )

    # Then look for platform-specific libs directory.
    if system_dir:
        platform_libs_dir = find_libs_dir(f"libopus/{system_dir}")
        if platform_libs_dir:
            search_paths.append((platform_libs_dir, lib_name))
            logger.debug(f"Found platform-specific libs directory: {platform_libs_dir}")

    # Look for general libs directory.
    general_libs_dir = find_libs_dir()
    if general_libs_dir:
        search_paths.append((general_libs_dir, lib_name))
        logger.debug(f"Added general libs directory: {general_libs_dir}")

    # Add project root as final fallback.
    project_root = get_project_root()
    search_paths.append((project_root, lib_name))

    # Log all search paths for debugging.
    for dir_path, filename in search_paths:
        full_path = dir_path / filename
        logger.debug(f"Search path: {full_path} (exists: {full_path.exists()})")
    return search_paths


def find_system_opus() -> str:
    """
    Find the Opus library from system paths.
    """
    system, _ = get_system_info()
    lib_path = ""

    try:
        # Get possible system library names.
        lib_names = cast(List[str], get_lib_name(system, False))

        # Try loading each possible name.
        for lib_name in lib_names:
            try:
                # Import ctypes.util for find_library.
                import ctypes.util

                system_lib_path = ctypes.util.find_library(lib_name)

                if system_lib_path:
                    lib_path = system_lib_path
                    logger.info(f"Found Opus library in system path: {lib_path}")
                    break
                else:
                    # Try loading by library name.
                    ctypes.cdll.LoadLibrary(lib_name)
                    lib_path = lib_name
                    logger.info(f"Loaded system Opus library directly: {lib_name}")
                    break
            except Exception as e:
                logger.debug(f"Failed to load system library {lib_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to locate system Opus library: {e}")

    return lib_path


def copy_opus_to_project(system_lib_path):
    """
    Copy the system library into the project directory.
    """
    from .resource_finder import get_project_root

    system, arch_name = get_system_info()

    if not system_lib_path:
        logger.error("Cannot copy Opus library: system library path is empty.")
        return None

    try:
        # Use resource_finder to get project root.
        project_root = get_project_root()

        # Get target directory path using actual structure.
        target_path = get_lib_path(system, arch_name)
        target_dir = project_root / target_path

        # Create target directory (if missing).
        target_dir.mkdir(parents=True, exist_ok=True)

        # Determine target filename.
        lib_name = cast(str, get_lib_name(system))
        target_file = target_dir / lib_name

        # Copy file.
        shutil.copy2(system_lib_path, target_file)
        logger.info(f"Copied Opus library from {system_lib_path} to {target_file}")

        return str(target_file)

    except Exception as e:
        logger.error(f"Failed to copy Opus library to project directory: {e}")
        return None


def setup_opus() -> bool:
    """
    Set up the Opus dynamic library.
    """
    # Check if runtime_hook already loaded it.
    if hasattr(sys, "_opus_loaded"):
        logger.info("Opus library already loaded by runtime hook.")
        return True

    # Get current system info.
    system, arch_name = get_system_info()
    logger.info(f"Current system: {system}, arch: {arch_name}")

    # Build search paths.
    search_paths = get_search_paths(system, arch_name)

    # Search for local library file.
    lib_path = ""
    lib_dir = ""

    for dir_path, file_name in search_paths:
        full_path = dir_path / file_name
        if full_path.exists():
            lib_path = str(full_path)
            lib_dir = str(dir_path)
            logger.info(f"Found Opus library file: {lib_path}")
            break

    # If not found locally, try system lookup.
    if not lib_path:
        logger.warning("Local Opus library not found; trying system path.")
        system_lib_path = find_system_opus()

        if system_lib_path:
            # First try using the system library directly.
            try:
                _ = ctypes.cdll.LoadLibrary(system_lib_path)
                logger.info(f"Loaded Opus library from system path: {system_lib_path}")
                sys._opus_loaded = True
                return True
            except Exception as e:
                logger.warning(
                    f"Failed to load system Opus library: {e}; copying to project."
                )

            # If direct load fails, copy to project directory.
            lib_path = copy_opus_to_project(system_lib_path)
            if lib_path:
                lib_dir = str(Path(lib_path).parent)
            else:
                logger.error("Unable to find or copy the Opus library file.")
                return False
        else:
            logger.error("Opus library not found on the system.")
            return False

    # Windows-specific handling.
    if system == PLATFORM.WINDOWS and lib_dir:
        # Add DLL search path.
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_dir)
                logger.debug(f"Added DLL search path: {lib_dir}")
            except Exception as e:
                logger.warning(f"Failed to add DLL search path: {e}")

        # Set environment variable.
        os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")

    # Patch library path.
    _patch_find_library("opus", lib_path)

    # Attempt to load library.
    try:
        # Load DLL and keep reference to prevent GC.
        _ = ctypes.CDLL(lib_path)
        logger.info(f"Successfully loaded Opus library: {lib_path}")
        sys._opus_loaded = True
        return True
    except Exception as e:
        logger.error(f"Failed to load Opus library: {e}")
        return False


def _patch_find_library(lib_name: str, lib_path: str):
    """
    Patch ctypes.util.find_library.
    """
    import ctypes.util

    original_find_library = ctypes.util.find_library

    def patched_find_library(name):
        if name == lib_name:
            return lib_path
        return original_find_library(name)

    ctypes.util.find_library = patched_find_library
