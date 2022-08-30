from __future__ import annotations

from dataclasses import dataclass

from revolve2.core.modular_robot import ModularRobot

from jlo.drl_fixed_body.rl_brain import RLbrain

from jlo.direct_tree.direct_tree_genotype import DirectTreeGenotype
from jlo.direct_tree.direct_tree_config import DirectTreeGenotypeConfig
from jlo.direct_tree.direct_tree_genotype import develop as body_develop

from jlo.direct_tree import robot_zoo


def _make_direct_tree_config() -> DirectTreeGenotypeConfig:
    """
    Parameters to evolve the body of the agents
    """
    morph_single_mutation_prob = 0.2
    morph_no_single_mutation_prob = 1 - morph_single_mutation_prob  # 0.8
    morph_no_all_mutation_prob = morph_no_single_mutation_prob ** 4  # 0.4096
    morph_at_least_one_mutation_prob = 1 - morph_no_all_mutation_prob  # 0.5904

    brain_single_mutation_prob = 0.5

    tree_genotype_conf: DirectTreeGenotypeConfig = DirectTreeGenotypeConfig(
        max_parts=50,
        min_parts=10,
        max_oscillation=5,
        init_n_parts_mu=10,
        init_n_parts_sigma=4,
        init_prob_no_child=0.1,
        init_prob_child_block=0.4,
        init_prob_child_active_joint=0.5,
        mutation_p_duplicate_subtree=morph_single_mutation_prob,
        mutation_p_delete_subtree=morph_single_mutation_prob,
        mutation_p_generate_subtree=morph_single_mutation_prob,
        mutation_p_swap_subtree=morph_single_mutation_prob,
        mutation_p_mutate_oscillators=brain_single_mutation_prob,
        mutation_p_mutate_oscillator=0.5,
        mutate_oscillator_amplitude_sigma=0.3,
        mutate_oscillator_period_sigma=0.3,
        mutate_oscillator_phase_sigma=0.3,
    )

    return tree_genotype_conf


@dataclass
class Agent:
    body: DirectTreeGenotype
    brain: RLbrain


def make_agent() -> Agent:
    body = robot_zoo.make_ant_body()
    brain = None

    return Agent(body, brain)

def develop(agent: Agent) -> ModularRobot:
    body = body_develop(agent.body)
    brain = agent.brain
    return ModularRobot(body, brain)
