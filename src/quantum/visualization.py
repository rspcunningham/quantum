import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from quantum.system import QuantumSystem

def plot_results(results: dict[str, int], title: str = "Quantum Measurement Results", show: bool = True) -> tuple[Figure, Axes]:
    """Plot simulation results as a bar chart.

    Args:
        results: Dictionary mapping bit strings to their counts (from run_simulation)
        title: Title for the plot
        show: Whether to display the plot immediately (default: True)

    Returns:
        Tuple of (figure, axes) for further customization if needed
    """
    # Sort results by bit string for consistent ordering
    sorted_results = dict(sorted(results.items()))

    states = list(sorted_results.keys())
    counts = list(sorted_results.values())

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(states) * 0.5), 6)) # pyright: ignore[reportUnknownMemberType]

    # Create bar plot with seaborn
    _ = sns.barplot(x=states, y=counts, ax=ax)

    # Customize
    _ = ax.set_xlabel("Measurement Outcome (Bit String)", fontsize=12) # pyright: ignore[reportUnknownMemberType]
    _ = ax.set_ylabel("Count", fontsize=12) # pyright: ignore[reportUnknownMemberType]
    _ = ax.set_title(title, fontsize=14, fontweight='bold') # pyright: ignore[reportUnknownMemberType]

    # Rotate x-axis labels if there are many states
    if len(states) > 8:
        _ = plt.xticks(rotation=45, ha='right') # pyright: ignore[reportUnknownMemberType]

    # Add count labels on top of bars
    for i, count in enumerate(counts):
        _ = ax.text(i, count, str(count), ha='center', va='bottom', fontsize=10) # pyright: ignore[reportUnknownMemberType]

    plt.tight_layout()

    if show:
        plt.show() # pyright: ignore[reportUnknownMemberType]

    return fig, ax


def plot_probs(system: QuantumSystem, title: str = "Quantum State Probability Distribution", show: bool = True) -> tuple[Figure, Axes]:
    """Plot the probability distribution of a quantum system's state vector.

    Args:
        system: QuantumSystem instance to visualize
        title: Title for the plot
        show: Whether to display the plot immediately (default: True)

    Returns:
        Tuple of (figure, axes) for further customization if needed
    """
    # Get probability distribution from state vector
    probs = system.get_distribution().cpu().numpy().flatten()

    # Create basis state labels (bit strings)
    n_states = len(probs)
    states = [format(i, f'0{system.n_qubits}b') for i in range(n_states)]

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, n_states * 0.5), 6)) # pyright: ignore[reportUnknownMemberType]

    # Create bar plot with seaborn
    _ = sns.barplot(x=states, y=probs, ax=ax)

    # Customize
    _ = ax.set_xlabel("Basis State (Bit String)", fontsize=12) # pyright: ignore[reportUnknownMemberType]
    _ = ax.set_ylabel("Probability", fontsize=12) # pyright: ignore[reportUnknownMemberType]
    _ = ax.set_title(title, fontsize=14, fontweight='bold') # pyright: ignore[reportUnknownMemberType]
    _ = ax.set_ylim(0, 1.0)

    # Rotate x-axis labels if there are many states
    if n_states > 8:
        _ = plt.xticks(rotation=45, ha='right') # pyright: ignore[reportUnknownMemberType]

    # Add probability labels on top of bars (only for non-negligible probabilities)
    for i, prob in enumerate(probs):
        if prob > 0.01:  # Only show labels for probabilities > 1%
            _ = ax.text(i, prob, f'{prob:.3f}', ha='center', va='bottom', fontsize=10) # pyright: ignore[reportUnknownMemberType]

    plt.tight_layout()

    if show:
        plt.show() # pyright: ignore[reportUnknownMemberType]

    return fig, ax
