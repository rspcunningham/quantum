from quantum import run_simulation, registers, H, X, CCX, ControlledGateType, measure_all, plot_results

search, ancilla = registers(4, 1)
anc = ancilla[0]

CCCX = ControlledGateType(CCX)
CCCCX = ControlledGateType(CCCX)

init = H.on(search) + X(anc) + H(anc)
oracle = CCCCX(*search, anc)
diffuser = H.on(search) + X.on(search) + CCCCX(*search, anc) + X.on(search) + H.on(search)

circuit = init + (oracle + diffuser) * 3 + measure_all(search)
result = run_simulation(circuit, 100)
_ = plot_results(result)
