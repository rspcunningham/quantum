from rich import print
from rich.table import Table
from rich.text import Text
from system import QuantumSystem

def format_basis(num: int, n_qubits):
    basis_bits = format(num, f"0{n_qubits}b")
    basis = f"|{basis_bits}⟩"
    return basis

def _render_probability_bar(probability: float, width: int = 12) -> Text:
    """Render a miniature bar proportional to the probability."""
    probability = max(0.0, min(1.0, probability))
    filled = int(probability * width)
    remainder = (probability * width) - filled

    partial_steps = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    partial_index = min(len(partial_steps) - 1, int(remainder * len(partial_steps)))
    bar = "█" * filled
    if partial_index and filled < width:
        bar += partial_steps[partial_index]

    if len(bar) < width:
        bar = bar.ljust(width)

    return Text(bar)

def display(program_state: ProgramState):
    """Display the quantum state in a nice table format"""
    dist = program_state._get_distribution()
    n = program_state.state_vector.shape[0]

    table = Table(show_header=True)
    table.add_column("Basis", justify="left")
    table.add_column("Amplitude", justify="right")
    table.add_column("Probability", justify="right")
    table.add_column("Distribution", justify="left")

    for i in range(n):
        basis = format_basis(i, program_state.n_qbits)
        amplitude = f"{program_state.state_vector[i].item():.4f}"
        probability_value = dist[i].item()
        probability = f"{probability_value:.4f}"
        probability_bar = _render_probability_bar(probability_value)
        table.add_row(basis, amplitude, probability, probability_bar)

    print(table)
