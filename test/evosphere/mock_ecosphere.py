from revolve.evosphere.biosphere import Ecosphere
from simulation.simulator.simulator_helper import SimulatorType


class MockEcosphere(Ecosphere):

    def __init__(self, name: str = "test"):
        super().__init__(name)
        self.simulator_type = SimulatorType.NONE


class MockEcosphereA(MockEcosphere):

    def __init__(self):
        super().__init__("A")


class MockEcosphereB(MockEcosphere):

    def __init__(self):
        super().__init__("B")


class MockEcosphereC(MockEcosphere):

    def __init__(self):
        super().__init__("C")