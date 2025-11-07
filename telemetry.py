import matplotlib.pyplot as plt
import numpy as np
import torch


class DistributionPlotter:
    def __init__(self, distribution):
        dist = self._to_1d(distribution)

        self.n = dist.shape[0]
        self.prev_max_idx = None

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)

        x = np.arange(self.n)
        self.bars = self.ax.bar(x, dist, width=1.0)

        self.ax.set_xlim(-0.5, self.n - 0.5)
        self.ax.set_xlabel("index")
        self.ax.set_ylabel("p")

        # initial style
        self._style_bars(dist)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show(block=False)

    def _to_1d(self, distribution):
        if isinstance(distribution, torch.Tensor):
            return distribution.detach().view(-1).cpu().numpy()
        return np.asarray(distribution).reshape(-1)

    def _style_bars(self, dist):
        # y-limit based on max
        ymax = max(1e-12, dist.max())
        self.ax.set_ylim(0, ymax * 1.05)

        for i, bar in enumerate(self.bars):
            bar.set_color("tab:blue")
            bar.set_alpha(0.5)
            bar.set_width(10.0)

    def update(self, distribution):
        dist = self._to_1d(distribution)
        if dist.shape[0] != self.n:
            raise ValueError(f"Expected length {self.n}, got {dist.shape[0]}")

        for bar, h in zip(self.bars, dist):
            bar.set_height(float(h))

        self._style_bars(dist)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)
