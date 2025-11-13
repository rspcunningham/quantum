import math
from src.quantum.gates import H, CCX, RY

h = H(0, 1)
print(h)

ccx = CCX(0, 1, 2)
print(ccx)

ry = RY(math.pi)(0)
print(ry)
