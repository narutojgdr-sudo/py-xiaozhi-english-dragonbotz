from src.iot.thing import Thing


class Lamp(Thing):
    def __init__(self):
        super().__init__("Lamp", "A test lamp")
        self.power = False

        # Define properties - use async getter.
        self.add_property("power", "Whether the lamp is on", self.get_power)

        # Define methods - use async handlers.
        self.add_method("TurnOn", "Turn on the lamp", [], self._turn_on)

        self.add_method("TurnOff", "Turn off the lamp", [], self._turn_off)

    async def get_power(self):
        return self.power

    async def _turn_on(self, params):
        self.power = True
        return {"status": "success", "message": "Lamp turned on"}

    async def _turn_off(self, params):
        self.power = False
        return {"status": "success", "message": "Lamp turned off"}
