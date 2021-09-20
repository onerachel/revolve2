from typing import List

from abstract.configurations import SpeciationConfiguration
from nca.core.actor.actors import Actors
from nca.core.actor.individual import Individual
from nca.core.ecology.population import Population
from nca.core.ecology.population_management import PopulationManagement
from nca.core.ecology.speciation.genus import Genus


class GenusManagement(PopulationManagement):

    def __init__(self, configuration=SpeciationConfiguration(), compatibility: Compatibility = Compatibility()):
        super().__init__(configuration)

        self.genus: Genus = Genus(compatibility)

    def initialize(self, agents: Actors):
        self.genus = Genus(self.genus.compatibility)

        for agent in agents:
            self.assign(agent)
        print(len(self.populations()))

    def assign(self, individual: Individual):
        inserted = self.genus.insert(individual)
        if not inserted:
            self.genus.add(Population(Actors([individual])))

    def speciate(self):
        self.initialize(self.agents())

    def populations(self) -> List[Population]:
        if self.genus is None:
            raise Exception("Genus uninitialized")
        return [population for population in self.genus.species]

    def agents(self) -> Actors:
        all_agents: Actors = Actors()

        for population in self.populations():
            for agent in population.individuals:
                all_agents.append(agent)

        return all_agents

    def to_json(self):
        return self.genus.to_json()
