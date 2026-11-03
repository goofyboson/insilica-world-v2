"""Many worlds at once (V2-5, lever 4): each world keeps its own state, stream and captured tick graphs, and the
ensemble queues every member's tick before waiting, so the GPU runs the members' small kernels side by side. A member
does exactly what it does alone (its own graph on its own stream): results are identical to single runs (test ENSEMBLE).
At n = 4 eight members fill the GPU: about ×2.9 the worlds per second of one (docs/v2-5-speed.md)."""
from .world import World


class Ensemble:
    def __init__(self, worlds):
        self.worlds = list(worlds)

    @classmethod
    def build(cls, n, preset, params_list, **kw):
        """One world per entry of params_list (a calibration sweep's settings), same grid and preset."""
        return cls(World(n, preset, params=p, **kw) for p in params_list)

    def step(self, k=1, graph=True):
        for _ in range(k):
            for w in self.worlds:
                w.launch_tick(graph)
        for w in self.worlds:
            w._stream.synchronize()
        return self

    def __iter__(self):
        return iter(self.worlds)

    def __len__(self):
        return len(self.worlds)
