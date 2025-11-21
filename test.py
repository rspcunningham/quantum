from quantum.system import Circuit
from quantum.gates import H, X, CX

def test_circuit_uncompute():
    c0 = Circuit([H(0), X(1), CX(0, 1)])
    c1 = Circuit([H(0), c0])
    c2 = c1.uncomputed()

    print(c1.operations)
    print(c2.operations)
    assert c2.operations == [Circuit([CX(0, 1), X(1), H(0)]), H(0)]

test_circuit_uncompute()
