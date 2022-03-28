from revolve2.core.optimization import Process, ProcessIdGen
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from typing import List
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import Ndarray1xnSerializer, DbNdarray1xn
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from abc import ABC, abstractmethod
from random import Random
import numpy as np
import numpy.typing as npt
from typing import Optional


class OpenaiESOptimizer(ABC, Process):
    """
    OpenAI ES optimizer based on:
    https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d
    https://openai.com/blog/evolution-strategies/
    """

    @abstractmethod
    async def _evaluate_population(
        self,
        database: AsyncEngine,
        process_id: int,
        process_id_gen: ProcessIdGen,
        population: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """
        Evaluate all individuals in the population, returning their fitnesses.

        :population: MxN array with M the population size and N the size of an individual.
        :return: M long vector with M the population size, representing the fitness of each individual in `population`.
        """

    @abstractmethod
    def _must_do_next_gen(self) -> bool:
        """
        Decide if the optimizer must do another generation.
        :return: True if it must.
        """

    __database: AsyncEngine
    __process_id: int
    __process_id_gen: ProcessIdGen

    __rng: Random

    __population_size: int
    __sigma: float
    __learning_rate: float

    __gen_num: int
    __mean: npt.NDArray[np.float_]  # Nx1 array

    async def ainit_new(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
        rng: Random,
        population_size: int,
        sigma: float,
        learning_rate: float,
        initial_mean: npt.NDArray[np.float_],
    ) -> None:
        """
        :sigma: standard deviation
        :initial_mean: Nx1 array
        """
        self.__database = database
        self.__process_id = process_id
        self.__process_id_gen = process_id_gen

        self.__rng = rng

        self.__population_size = population_size
        self.__sigma = sigma
        self.__learning_rate = learning_rate

        self.__gen_num = 0
        self.__mean = initial_mean

        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await Ndarray1xnSerializer.create_tables(session)

        dbmeanid = (await Ndarray1xnSerializer.to_database(session, [self.__mean]))[0]
        dbopt = DBOpenaiESOptimizer(
            process_id=self.__process_id,
            population_size=self.__population_size,
            sigma=self.__sigma,
            learning_rate=self.__learning_rate,
            initial_mean=dbmeanid,
        )
        session.add(dbopt)

    async def ainit_from_database(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        process_id: int,
        process_id_gen: ProcessIdGen,
    ) -> bool:
        pass  # TODO

    async def run(self) -> None:
        while self.__safe_must_do_next_gen():
            rng = np.random.Generator(
                np.random.PCG64(self.__rng.randint(0, 2**63))
            )  # rng is currently not numpy, but this would be very convenient. do this until that is resolved.
            pertubations = rng.standard_normal(
                (self.__population_size, len(self.__mean))
            )
            population = self.__sigma * pertubations + self.__mean

            fitnesses = await self._evaluate_population(
                self.__database,
                self.__process_id_gen.gen(),
                self.__process_id_gen,
                population,
            )

            assert fitnesses.shape == (len(population),)
            fitnesses_gaussian = (fitnesses - np.mean(fitnesses)) / np.std(fitnesses)
            self.__mean = self.__mean + self.__learning_rate / (
                self.__population_size * self.__sigma
            ) * np.dot(pertubations.T, fitnesses_gaussian)

            self.__gen_num += 1

            async with AsyncSession(self.__database) as session:
                async with session.begin():
                    db_individual_ids = await Ndarray1xnSerializer.to_database(
                        session, population
                    )

                    dbgens = [
                        DBOpenaiESOptimizerGeneration(
                            process_id=self.__process_id,
                            gen_num=self.__gen_num,
                            gen_index=index,
                            individual=id,
                            fitness=fitness,
                        )
                        for index, id, fitness in zip(
                            range(len(population)), db_individual_ids, fitnesses
                        )
                    ]

                    session.add_all(dbgens)

    @property
    def generation_number(self) -> Optional[int]:
        """
        Get the current generation.
        The initial generation is numbered 0.
        """

        return self.__gen_num

    def __safe_must_do_next_gen(self) -> bool:
        must_do = self._must_do_next_gen()
        assert type(must_do) == bool
        return must_do


DbBase = declarative_base()


class DBOpenaiESOptimizer(DbBase):
    __tablename__ = "openaies_optimizer"

    process_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        primary_key=True,
    )
    population_size = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    sigma = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    learning_rate = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    initial_mean = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
    )


class DBOpenaiESOptimizerGeneration(DbBase):
    __tablename__ = "openaies_optimizer_generation"

    process_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_num = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    gen_index = sqlalchemy.Column(sqlalchemy.Integer, nullable=False, primary_key=True)
    individual = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey(DbNdarray1xn.id), nullable=False
    )
    fitness = sqlalchemy.Column(sqlalchemy.Float, nullable=True)
