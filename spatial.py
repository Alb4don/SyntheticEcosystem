import math
import numpy as np

_SPAN = 1 << 20
_HALF_SPAN = (_SPAN >> 1)


class SpatialGrid:
    def __init__(self, positions, cell):
        if cell <= 0 or not math.isfinite(cell):
            raise ValueError('Tamanho de célula inválido')
        self.cell = float(cell)
        self.positions = np.array(positions, dtype=float, copy=True)
        self.buckets = {}
        if not len(self.positions):
            return
        keys = np.floor(self.positions / self.cell).astype(np.int64)
        flat = (keys[:, 0] + _HALF_SPAN) * _SPAN + (keys[:, 1] + _HALF_SPAN)
        order = np.argsort(flat, kind='stable')
        sorted_flat = flat[order]
        uniques, starts = np.unique(sorted_flat, return_index=True)
        ends = np.append(starts[1:], len(order))
        for value, start, end in zip(uniques.tolist(), starts.tolist(), ends.tolist()):
            kx = value // _SPAN - _HALF_SPAN
            ky = value % _SPAN - _HALF_SPAN
            self.buckets[(kx, ky)] = order[start:end]

    def _cell_of(self, point):
        return int(math.floor(point[0] / self.cell)), int(math.floor(point[1] / self.cell))

    def query(self, point, radius):
        empty = np.empty(0, dtype=np.int64)
        if not self.buckets or radius <= 0:
            return empty, np.empty(0, dtype=float)
        reach = int(math.ceil(radius / self.cell))
        cx, cy = self._cell_of(point)
        parts = [self.buckets[(x, y)] for x in range(cx - reach, cx + reach + 1) for y in range(cy - reach, cy + reach + 1) if (x, y) in self.buckets]
        if not parts:
            return empty, np.empty(0, dtype=float)
        candidates = np.concatenate(parts)
        distances = np.linalg.norm(self.positions[candidates] - np.asarray(point, dtype=float), axis=1)
        mask = distances <= radius
        selected = candidates[mask]
        order = np.argsort(selected, kind='stable')
        return selected[order], distances[mask][order]


def pairs_within(positions, radius):
    points = np.asarray(positions, dtype=float)
    count = len(points)
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=float))
    if count < 2 or radius <= 0:
        return empty
    order = np.argsort(points[:, 0], kind='stable')
    ordered = points[order]
    xs = ordered[:, 0]
    ends = np.searchsorted(xs, xs + radius, side='right')
    starts = np.arange(1, count + 1)
    counts = np.maximum(ends - starts, 0)
    total = int(counts.sum())
    if not total:
        return empty
    left = np.repeat(np.arange(count), counts)
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    right = np.repeat(starts, counts) + offsets
    delta = ordered[left] - ordered[right]
    distance = np.hypot(delta[:, 0], delta[:, 1])
    mask = distance <= radius
    return order[left[mask]], order[right[mask]], distance[mask]
