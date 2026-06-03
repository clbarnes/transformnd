from transformnd.base import TransformSequence
from transformnd.graph import TransformGraph
from transformnd.types import Spaces
from transformnd.transforms.simple import Translate, Scale
from transformnd.transforms.affine import Affine

import numpy as np

def test_graph_traversal():
    t1 = Translate([1, 2], spaces=Spaces("A", "B"))
    t2 = Scale([2, 3], spaces=Spaces("B", "C"))
    t3 = Affine(np.eye(3), spaces=Spaces("C", "D"))

    graph = TransformGraph()
    graph.add_transforms([t1])
    graph.add_transforms([t2])
    graph.add_transforms([t3])

    seq = graph.get_sequence("A", "D")
    assert isinstance(seq, TransformSequence)
    assert len(seq) == 1

    simplified_seq = seq.simplify()
    assert simplified_seq.spaces.source == "A"
    assert simplified_seq.spaces.target == "D"

    expected_affine = np.array([[2, 0, 2], [0, 3, 6], [0, 0, 1]])
    assert np.array_equal(simplified_seq.to_affine().matrix, expected_affine)

if __name__ == "__main__":
    test_graph_traversal()
    print("All tests passed!")