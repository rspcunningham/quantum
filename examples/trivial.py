from quantum import run_simulation, QuantumRegister, H, CX, measure_all, plot_results

qr = QuantumRegister(2)
circuit = H(qr[0]) + CX(qr[0], qr[1]) + measure_all(qr)
result = run_simulation(circuit, 1000)
_ = plot_results(result)
