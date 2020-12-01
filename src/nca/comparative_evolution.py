from typing import List
import matplotlib.pyplot as plt

from nca.core.actor.fitness import Fitness
from nca.core.actor.fitnesses import OnesFitness, FitnessEvaluation
from nca.core.actor.individual_factory import ActorFactory
from nca.core.analysis.statistics import Statistics
from nca.core.analysis.statistics_plotter import plot_statistics_measures_list
from nca.core.evolution.evolutionary_configurations import EvolutionaryConfiguration
from nca.repeated_evolution import RepeatedEvolution


class ComparativeEvolution:

    def __init__(self, evolutionary_configurations: List[EvolutionaryConfiguration],
                 fitness_evaluation: FitnessEvaluation = OnesFitness(), individual_factory: ActorFactory = ActorFactory(),
                 algorithm_names: List[str] = None, repeat: int = 1):
        self.evolutionary_configurations: List[EvolutionaryConfiguration] = evolutionary_configurations
        self.fitness_evaluation: FitnessEvaluation = fitness_evaluation
        self.individual_factory: ActorFactory = individual_factory

        self.statistics_measures: List[Statistics] = []
        if algorithm_names is None:
            algorithm_names = []
        self.algorithm_names: List[str] = algorithm_names
        self.repeat = repeat

    def gather(self):
        for index, evolutionary_configuration in enumerate(self.evolutionary_configurations):
            evolution = RepeatedEvolution(evolutionary_configuration, self.fitness_evaluation, self.individual_factory,
                                          False, repetitions=self.repeat)
            self.statistics_measures.append(evolution.evolve())
            print(self.algorithm_names[index] + " algorithm done...")

    def plot(self):
        plot_statistics_measures_list(self.statistics_measures, self.algorithm_names)
