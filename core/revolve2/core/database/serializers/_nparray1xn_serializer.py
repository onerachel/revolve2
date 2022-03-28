from __future__ import annotations

from typing import List

from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select

from revolve2.core.database import IncompatibleError, Tableable

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from dataclasses import dataclass
from .._serializer import Serializer
import numpy.typing as npt
import numpy as np


@dataclass
class Ndarray1xnSerializer(Serializer[npt.NDArray[np.float_]]):
    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[npt.NDArray[np.float_]]
    ) -> List[int]:
        dblists = [DbNdarray1xn() for _ in objects]
        session.add_all(dblists)
        await session.flush()
        ids = [
            dbfitness.id for dbfitness in dblists if dbfitness.id is not None
        ]  # cannot be none because not nullable but adding check for mypy
        assert len(ids) == len(objects)  # just to be sure because of check above

        items = [
            DbNdarray1xnItem(nparray1xn_id=id, array_index=i, value=v)
            for id, values in zip(ids, objects)
            for i, v in enumerate(values)
        ]

        session.add_all(items)

        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[npt.NDArray[np.float_]]:
        asd = await session.execute(
            select(DbNdarray1xnItem).filter(DbNdarray1xnItem.nparray1xn_id.in_(ids))
        )

        # id_map = {t.id: t for t in rows}

        # return [FitnessFloat(id_map[id].fitness) for id in ids]
        raise NotImplementedError()


DbBase = declarative_base()


class DbNdarray1xn(DbBase):
    __tablename__ = "nparray1xn"

    id = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True, autoincrement=True
    )


class DbNdarray1xnItem(DbBase):
    __tablename__ = "nparray1xn_item"

    nparray1xn_id = sqlalchemy.Column(
        sqlalchemy.Integer,
        sqlalchemy.ForeignKey(DbNdarray1xn.id),
        nullable=False,
        primary_key=True,
    )
    array_index = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        primary_key=True,
    )
    value = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
