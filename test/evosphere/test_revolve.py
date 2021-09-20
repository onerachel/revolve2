import unittest

from test.evosphere.mock_evosphere import MockEvosphere


class TestEvosphere(unittest.TestCase):

    def test_create(self):

        evosphere = MockEvosphere()
        evosphere.evolve()
        self.assertTrue(True)
