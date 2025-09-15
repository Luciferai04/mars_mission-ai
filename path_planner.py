from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import heapq
import numpy as np
from rasterio.transform import rowcol


def _neighbors_8(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    res = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                res.append((nr, nc))
    return res


@dataclass
class PathPlanner:
    """A* path planner over slope-derived cost maps.

    - Hazards: cells with slope > slope_max_deg are impassable
    - Cost: base 1.0 + penalty factor for slope approaching limit
    - Input coordinates are latitude/longitude (planetocentric, 0-360 lon)
    """

    slope_max_deg: float = 15.0

    def cost_map(self, slope_deg: np.ndarray) -> np.ndarray:
        hazard = slope_deg > self.slope_max_deg
        norm = np.clip(slope_deg / max(self.slope_max_deg, 1e-3), 0.0, 1.0)
        cost = 1.0 + 4.0 * norm  # up to 5x cost near threshold
        cost[hazard] = np.inf
        return cost

    def latlon_to_rc(self, transform, lat: float, lon: float) -> Tuple[int, int]:
        # Raster assumed in geographic CRS suitable for row/col transform
        r, c = rowcol(transform, lon, lat)
        return int(r), int(c)

    def astar(
        self, cost: np.ndarray, start_rc: Tuple[int, int], goal_rc: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        h, w = cost.shape
        sr, sc = start_rc
        gr, gc = goal_rc
        if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
            return None
        if not np.isfinite(cost[sr, sc]) or not np.isfinite(cost[gr, gc]):
            return None
        openq: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0
        heapq.heappush(openq, (0.0, counter, (sr, sc)))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        gscore = {(sr, sc): 0.0}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        closed = set()
        while openq:
            _, _, current = heapq.heappop(openq)
            if current in closed:
                continue
            if current == (gr, gc):
                # Reconstruct
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            closed.add(current)
            cr, cc = current
            for nr, nc in _neighbors_8(cr, cc, h, w):
                if not np.isfinite(cost[nr, nc]):
                    continue
                tentative = gscore[current] + 0.5 * (cost[cr, cc] + cost[nr, nc])
                if tentative < gscore.get((nr, nc), float("inf")):
                    came_from[(nr, nc)] = current
                    gscore[(nr, nc)] = tentative
                    counter += 1
                    fscore = tentative + heuristic((nr, nc), (gr, gc))
                    heapq.heappush(openq, (fscore, counter, (nr, nc)))
        return None

    def plan_route_from_slope(
        self,
        slope_deg: np.ndarray,
        raster_transform,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
    ) -> Dict[str, Any]:
        cost = self.cost_map(slope_deg)
        sr, sc = self.latlon_to_rc(raster_transform, start_lat, start_lon)
        gr, gc = self.latlon_to_rc(raster_transform, goal_lat, goal_lon)
        path = self.astar(cost, (sr, sc), (gr, gc))
        return {
            "start_rc": [sr, sc],
            "goal_rc": [gr, gc],
            "path_rc": path if path is not None else [],
            "success": path is not None,
        }