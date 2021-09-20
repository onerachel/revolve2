import unittest

from nca.core.genome.grammar.grammar_initialization import LSystemInitialization
from nca.core.genome.grammar.lindenmayer_system import LSystemGenotype


class LSystemTest(unittest.TestCase):

    def test_lsystem(self):
        representation_1 = LSystemGenotype(LSystemInitialization())
        representation_2 = LSystemGenotype(LSystemInitialization())

        self.assertNotEqual(representation_1, representation_2)
