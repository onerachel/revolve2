
import unittest

from evosphere.robot.test_birth_clinic import MockBirthClinic
from nca.core.actor.individual_factory import ActorFactory
from simulation.simulator.simulator_command import SimulateCommand, TaskPriority
from src.simulation.simulation_manager import SimulationManager


from evosphere.mock_ecosphere import MockEcosphereA


class SimulationManagerTest(unittest.TestCase):

    def test_supervisor(self):

        manager = SimulationManager()
        agents = ActorFactory().create(3)
        ecosphere = MockEcosphereA()
        birth_clinic = MockBirthClinic()
        manager.simulate(SimulateCommand(agents, ecosphere, birth_clinic, TaskPriority.MEDIUM))
