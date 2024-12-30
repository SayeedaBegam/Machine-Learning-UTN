"""Compute graph data structure."""
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from graphviz import Digraph


@dataclass(eq=False)
class DataNode:
    """A value in the compute graph."""

    data: float
    gradient: float = 0


class Op(Enum):
    """The defined operations."""

    ADD = 0
    MUL = 1
    TANH = 2


LABELS = {Op.ADD: "+", Op.MUL: "*", Op.TANH: "tanh"}


@dataclass(eq=False)
class OpNode:
    """An operation in the compute graph."""

    op: Op


Node = DataNode | OpNode


class Edge(NamedTuple):
    """A directed edge from u to v."""

    u: Node
    v: Node


class Graph(NamedTuple):
    """A compute Graph."""

    V: set[Node] = None
    E: set[tuple[Node]] = None


def draw_dot(g: Graph):
    """Draw graphviz graph."""
    dot = Digraph()
    for v in g.V:
        if type(v) is DataNode:
            dot.node(
                name=str(id(v)),
                label="{ data %.4f | grad %.4f }" % (v.data, v.gradient),
                shape="record",
            )
        elif type(v) is OpNode:
            dot.node(name=str(id(v)), label=LABELS[v.op])
        else:
            raise TypeError(
                f"""
                Oh no, v was supposed to be either a DataNode or an OpNode,
                but it is a {type(v)}.
                """
            )
    for e in g.E:
        dot.edge(str(id(e.u)), str(id(e.v)))
    return dot


def topological_sort(g: Graph):
    """Topological sort."""
    L = []
    S = set(v for v in g.V if not set(e for e in g.E if e.v is v))
    while S:
        n = S.pop()
        L.append(n)
        for m in (e.v for e in g.E if e.u is n):
            g = Graph(g.V, g.E - {Edge(n, m)})
            if not set(e for e in g.E if e.v is m):
                S.add(m)
    if g.E:
        raise ValueError("Graph has at least one cycle")
    else:
        return L
