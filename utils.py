from rich import print
from rich.table import Table

def display(self):
    """Display the quantum state in a nice table format"""
    dist = self._get_distribution()
    n = self.state_vector.shape[0]
    num_qubits = n.bit_length() - 1

    table = Table(title="Quantum State", show_header=True, header_style="bold magenta")
    table.add_column("Basis", style="cyan", justify="center")
    table.add_column("Amplitude", style="yellow", justify="right")
    table.add_column("Probability", style="green", justify="right")

    for i in range(n):
        basis_bits = format(i, f"0{num_qubits}b")
        basis = f"|{basis_bits}⟩"
        amplitude = f"{self.state_vector[i].item():.4f}"
        probability = f"{dist[i].item():.1%}"
        table.add_row(basis, amplitude, probability)

    print(table)
