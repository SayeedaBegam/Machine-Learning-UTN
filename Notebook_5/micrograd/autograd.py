"""Functional autograd implementation."""
import math

from micrograd.graph import Graph, \
    Node, \
    DataNode, \
    Edge, \
    Op, \
    OpNode, \
    topological_sort


def add(g: Graph, x: DataNode, y: DataNode):
    """Add x and y and update the graph."""

    ###########################################################################
    # YOUR CODE
    ###########################################################################
    pass
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################


def mul(g: Graph, x: DataNode, y: DataNode):
    """Multiply x and y and update the graph."""

    ###########################################################################
    # YOUR CODE
    ###########################################################################
    pass
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################


def tanh(g: Graph, x: DataNode):
    """Hyperbolic tangent backward."""
    out = DataNode(math.tanh(x.data))
    op = OpNode(Op.TANH)
    g.V.update((out, op))
    g.E.update((Edge(x, op), Edge(op, out)))
    return g, out


def add_back(x: DataNode, y: DataNode, out: DataNode):
    """Add backward."""

    ###########################################################################
    # YOUR CODE
    ###########################################################################
    pass
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################


def mul_back(x: DataNode, y: DataNode, out: DataNode):
    """Mul backward."""

    ###########################################################################
    # YOUR CODE
    ###########################################################################
    pass
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################


def tanh_back(x: DataNode, out: DataNode):
    """Hyperbolic tangent backward."""
    x.gradient += 1 / math.cosh(x.data) ** 2 * out.gradient


backwards = {Op.MUL: mul_back, Op.ADD: add_back, Op.TANH: tanh_back}


def backward(g: Graph):
    """Backpropagation.

    Iterates through each triple (input nodes, operation node, output node) and
    uses the `backwards` dictionary to apply the corresponding function to each
    triple.
    """
    topo = tuple(reversed(topological_sort(g)))
    topo[0].gradient = 1
    for v in topo[1:]:
        if isinstance(v, OpNode):
            op_node = v
            (output_node,) = tuple(e.v for e in g.E if e.u is op_node)
            input_nodes = tuple(e.u for e in g.E if e.v is op_node)

    ###########################################################################
    # YOUR CODE
    ###########################################################################
            pass
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################


def neuron(
    g: Graph,
    x1: DataNode,
    x2: DataNode,
    w1: DataNode,
    w2: DataNode,
    b: DataNode,
):
    """Compute x^Tw + b where x = (x1, x2)^T and w = (w1, w2)^T."""
    g, w1x1 = mul(g, x1, w1)
    g, w2x2 = mul(g, x2, w2)
    g, w1x1_plus_w2x2 = add(g, w1x1, w2x2)
    return tanh(*add(g, w1x1_plus_w2x2, b))
