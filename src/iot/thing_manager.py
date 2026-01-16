import asyncio
import json
from typing import Any, Dict, Optional, Tuple

from src.iot.thing import Thing
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ThingManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ThingManager()
        return cls._instance

    def __init__(self):
        self.things = []
        self.last_states = {}  # Cache the last state.

    async def initialize_iot_devices(self, config):
        """Initialize IoT devices.

        Note: Countdown timer moved to MCP tools for better AI integration and feedback.
        """
        # Load configuration to decide whether to expose virtual devices to the AI
        try:
            from src.utils.config_manager import ConfigManager

            cfg = ConfigManager.get_instance()
            enable_virtual = bool(cfg.get_config("IOT.ENABLE_VIRTUAL_DEVICES", False))
        except Exception:
            enable_virtual = False

        if not enable_virtual:
            # Virtual/testing devices disabled by config; do not register them.
            logger.info(
                "Virtual IoT devices disabled (IOT.ENABLE_VIRTUAL_DEVICES=False)."
            )
            return

        from src.iot.things.lamp import Lamp

        # Add devices.
        self.add_thing(Lamp())

    def add_thing(self, thing: Thing) -> None:
        self.things.append(thing)

    async def get_descriptors_json(self) -> str:
        """
        Get descriptor JSON for all devices.
        """
        # get_descriptor_json() is synchronous (static data).
        descriptors = [thing.get_descriptor_json() for thing in self.things]
        return json.dumps(descriptors)

    async def get_states_json(self, delta=False) -> Tuple[bool, str]:
        """Get state JSON for all devices.

        Args:
            delta: True to return only changed states

        Returns:
            Tuple[bool, str]: Whether states changed and JSON string
        """
        if not delta:
            self.last_states.clear()

        changed = False

        tasks = [thing.get_state_json() for thing in self.things]
        states_results = await asyncio.gather(*tasks)

        states = []
        for i, thing in enumerate(self.things):
            state_json = states_results[i]

            if delta:
                # Check if state changed.
                is_same = (
                    thing.name in self.last_states
                    and self.last_states[thing.name] == state_json
                )
                if is_same:
                    continue
                changed = True
                self.last_states[thing.name] = state_json

            # Ensure state_json is a dict.
            if isinstance(state_json, dict):
                states.append(state_json)
            else:
                states.append(json.loads(state_json))  # Convert JSON string to dict.

        return changed, json.dumps(states)

    async def get_states_json_str(self) -> str:
        """
        Keep legacy method name/return type for backward compatibility.
        """
        _, json_str = await self.get_states_json(delta=False)
        return json_str

    async def invoke(self, command: Dict) -> Optional[Any]:
        """Invoke device methods.

        Args:
            command: Command dict with name and method

        Returns:
            Optional[Any]: Invocation result if found; otherwise raises
        """
        thing_name = command.get("name")
        for thing in self.things:
            if thing.name == thing_name:
                return await thing.invoke(command)

        # Log error.
        logger.error(f"Device not found: {thing_name}")
        raise ValueError(f"Device not found: {thing_name}")
