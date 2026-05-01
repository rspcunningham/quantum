from quantum import QuantumRegister, H, CX, compile, measure_all

try:
    from quantum.visualization import plot_results
except ImportError:
    plot_results = None

qr = QuantumRegister(2)
circuit = H(qr[0]) + CX(qr[0], qr[1]) + measure_all(qr)
with compile(circuit) as compiled:
    result = compiled.run(1000)
print(result)
if plot_results is not None:
    _ = plot_results(result)
