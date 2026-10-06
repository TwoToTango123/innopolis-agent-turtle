"""Occupancy grid loaded from a ROS map_server map (map.yaml + .pgm). No ROS dependencies.

Indexing convention: occ[iy, ix], iy = 0 is the BOTTOM row (smallest world y),
so cell (ix, iy) covers world [ox + ix*res, ox + (ix+1)*res) x [oy + iy*res, ...).
The PGM image is stored top row first, so it is flipped on load.
"""
import math
import os
from dataclasses import dataclass

import numpy as np
import yaml

FREE = 0
OCCUPIED = 100
UNKNOWN = -1


def read_pgm(path: str) -> np.ndarray:
    """Read a binary (P5) or ASCII (P2) PGM. Returns rows top-to-bottom."""
    with open(path, 'rb') as f:
        data = f.read()
    tokens = []
    pos = 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b'#':
            pos = data.index(b'\n', pos) + 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    pos += 1  # single whitespace after maxval
    if magic == b'P5':
        dtype = np.uint8 if maxval < 256 else np.dtype('>u2')
        img = np.frombuffer(data, dtype=dtype, count=w * h, offset=pos)
    elif magic == b'P2':
        img = np.array(data[pos:].split()[:w * h], dtype=np.int32)
    else:
        raise ValueError(f'{path}: unsupported PGM type {magic!r}')
    return (img.reshape(h, w).astype(np.float64) / maxval * 255.0).round().astype(np.uint8)


@dataclass
class GridMap:
    occ: np.ndarray          # int8 (h, w): FREE / OCCUPIED / UNKNOWN, occ[iy, ix]
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def width(self) -> int:
        return self.occ.shape[1]

    @property
    def height(self) -> int:
        return self.occ.shape[0]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'GridMap':
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        img_path = meta['image']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(yaml_path), img_path)
        img = read_pgm(img_path)[::-1, :]  # bottom row first
        # map_server "trinary" mode
        p = img / 255.0 if meta.get('negate', 0) else (255 - img) / 255.0
        occ = np.full(img.shape, UNKNOWN, dtype=np.int8)
        occ[p > meta['occupied_thresh']] = OCCUPIED
        occ[p < meta['free_thresh']] = FREE
        ox, oy = meta['origin'][0], meta['origin'][1]
        return cls(occ, float(meta['resolution']), float(ox), float(oy))

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor((x - self.origin_x) / self.resolution)),
                int(math.floor((y - self.origin_y) / self.resolution)))

    def cell_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        """Center of the cell."""
        return (self.origin_x + (ix + 0.5) * self.resolution,
                self.origin_y + (iy + 0.5) * self.resolution)

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height

    def value_at(self, x: float, y: float) -> int:
        ix, iy = self.world_to_cell(x, y)
        return int(self.occ[iy, ix]) if self.in_bounds(ix, iy) else UNKNOWN
