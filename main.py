import argparse
import asyncio
import signal
import sys

from src.application import Application
from src.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Xiaozhi AI Client")
    parser.add_argument(
        "--mode",
        choices=["gui", "cli"],
        default="gui",
        help="Run mode: gui (graphical) or cli (command line)",
    )
    parser.add_argument(
        "--protocol",
        choices=["mqtt", "websocket"],
        default="websocket",
        help="Communication protocol: mqtt or websocket",
    )
    parser.add_argument(
        "--skip-activation",
        action="store_true",
        help="Skip activation flow and start the app (debug only)",
    )
    return parser.parse_args()


async def handle_activation(mode: str) -> bool:
    """Handle the device activation flow with an existing event loop.

    Args:
        mode: Run mode, "gui" or "cli"

    Returns:
        bool: Whether activation succeeded
    """
    try:
        from src.core.system_initializer import SystemInitializer

        logger.info("Starting device activation flow check...")

        system_initializer = SystemInitializer()
        # Use SystemInitializer activation handling with GUI/CLI adaptation.
        result = await system_initializer.handle_activation_process(mode=mode)
        success = bool(result.get("is_activated", False))
        logger.info(f"Activation flow completed, result: {success}")
        return success
    except Exception as e:
        logger.error(f"Activation flow error: {e}", exc_info=True)
        return False


async def start_app(mode: str, protocol: str, skip_activation: bool) -> int:
    """
    Unified entry to start the app (runs in an existing event loop).
    """
    logger.info("Starting Xiaozhi AI client")

    # Handle activation flow.
    if not skip_activation:
        activation_success = await handle_activation(mode)
        if not activation_success:
            logger.error("Device activation failed; exiting.")
            return 1
    else:
        logger.warning("Skipping activation flow (debug mode).")

    # Create and start the application.
    app = Application.get_instance()
    return await app.run(mode=mode, protocol=protocol)


if __name__ == "__main__":
    exit_code = 1
    try:
        args = parse_args()
        setup_logging()

        # Detect Wayland and set Qt platform plugin configuration.
        import os

        is_wayland = (
            os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("XDG_SESSION_TYPE") == "wayland"
        )

        if args.mode == "gui" and is_wayland:
            # Ensure Qt uses the correct platform plugin in Wayland.
            if "QT_QPA_PLATFORM" not in os.environ:
                # Prefer the Wayland plugin, fallback to xcb (X11 compatibility).
                os.environ["QT_QPA_PLATFORM"] = "wayland;xcb"
                logger.info("Wayland environment: set QT_QPA_PLATFORM=wayland;xcb")

            # Disable unstable Qt features on Wayland.
            os.environ.setdefault("QT_WAYLAND_DISABLE_WINDOWDECORATION", "1")
            logger.info("Wayland environment detected; applied compatibility settings.")

        # Set signal handling: ignore SIGTRAP on macOS to avoid "trace trap" exits.
        try:
            if hasattr(signal, "SIGINT"):
                # Let qasync/Qt handle Ctrl+C; keep default or GUI layer handling.
                pass
            if hasattr(signal, "SIGTERM"):
                # Allow normal shutdown on terminate signal.
                pass
            if hasattr(signal, "SIGTRAP"):
                signal.signal(signal.SIGTRAP, signal.SIG_IGN)
        except Exception:
            # Ignore unsupported signal handling on some platforms/environments.
            pass

        if args.mode == "gui":
            # In GUI mode, main creates the QApplication and qasync event loop.
            try:
                import qasync
                from PyQt5.QtWidgets import QApplication
            except ImportError as e:
                logger.error(f"GUI mode requires qasync and PyQt5: {e}")
                sys.exit(1)

            qt_app = QApplication.instance() or QApplication(sys.argv)

            loop = qasync.QEventLoop(qt_app)
            asyncio.set_event_loop(loop)
            logger.info("Created qasync event loop in main.")

            # Prevent closing the last window from exiting the app prematurely.
            try:
                qt_app.setQuitOnLastWindowClosed(False)
            except Exception:
                pass

            with loop:
                exit_code = loop.run_until_complete(
                    start_app(args.mode, args.protocol, args.skip_activation)
                )
        else:
            # CLI mode uses the standard asyncio event loop.
            exit_code = asyncio.run(
                start_app(args.mode, args.protocol, args.skip_activation)
            )

    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
        exit_code = 0
    except Exception as e:
        logger.error(f"Program exited with error: {e}", exc_info=True)
        exit_code = 1
    finally:
        sys.exit(exit_code)
