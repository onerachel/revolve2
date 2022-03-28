from random import Random, sample
from revolve2.core.database import open_async_database_sqlite
import logging
from revolve2.core.optimization import ProcessIdGen
from optimizer import Optimizer
from revolve2.core.modular_robot import Body, Brick, Core, ActiveHinge
import math


def make_body() -> Body:
    body = Body()
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2.0)
    body.core.left.attachment.attachment = Brick(0.0)
    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = ActiveHinge(math.pi / 2.0)
    body.core.right.attachment.attachment = Brick(0.0)
    return body


async def main() -> None:
    POPULATION_SIZE = 100
    SIGMA = 0.1
    LEARNING_RATE = 0.05
    NUM_GENERATIONS = 100

    SIMULATION_TIME = 30
    SAMPLING_FREQUENCY = 5
    CONTROL_FREQUENCY = 60

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
    )

    # random number generator
    rng = Random()
    rng.seed(0)

    # database
    database = open_async_database_sqlite("./database")

    # process id generator
    process_id_gen = ProcessIdGen()

    body = make_body()

    process_id = process_id_gen.gen()
    # maybe_optimizer = await Optimizer.from_database(
    #    database, process_id, process_id_gen, rng, POPULATION_SIZE, SIGMA, LEARNING_RATE
    # )
    # if maybe_optimizer is not None:
    #    optimizer = maybe_optimizer
    # else:
    optimizer = await Optimizer.new(
        database,
        process_id,
        process_id_gen,
        rng,
        POPULATION_SIZE,
        SIGMA,
        LEARNING_RATE,
        body,
        simulation_time=SIMULATION_TIME,
        sampling_frequency=SAMPLING_FREQUENCY,
        control_frequency=CONTROL_FREQUENCY,
        num_generations=NUM_GENERATIONS,
    )

    logging.info("Starting optimization process..")

    await optimizer.run()

    logging.info(f"Finished optimizing.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
