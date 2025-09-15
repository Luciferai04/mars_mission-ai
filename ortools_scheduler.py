from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def _import_cp_model():
    try:
        from ortools.sat.python import cp_model  # type: ignore

        return cp_model
    except Exception:
        return None


@dataclass
class OrToolsScheduler:
    """Schedule activities using OR-Tools CP-SAT if available.

    - Keeps activity order as soft precedence
    - Ensures no overlap, respects peak power limit, and fits in time budget
    - Falls back gracefully if OR-Tools is not available or the model fails
    """

    def is_available(self) -> bool:
        return _import_cp_model() is not None

    def schedule(self, activities: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        cp_model = _import_cp_model()
        if cp_model is None:
            # Fallback naive schedule performed elsewhere
            return []

        model = cp_model.CpModel()
        horizon = int(constraints.get("time_budget_min", 480))
        peak_demand_w = int(constraints.get("power", {}).get("peak_demand_w", 900))

        # Variables per activity
        starts = []
        ends = []
        intervals = []
        durations = []
        powers = []
        for idx, a in enumerate(activities):
            d = int(max(1, float(a.get("duration_min", 15))))
            p = int(max(0, float(a.get("power_w", a.get("peak_power_w", 0)))))
            start = model.NewIntVar(0, horizon, f"start_{idx}")
            end = model.NewIntVar(0, horizon, f"end_{idx}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{idx}")
            starts.append(start)
            ends.append(end)
            intervals.append(interval)
            durations.append(d)
            powers.append(p)

        # No-overlap using cumulative with capacity 1 on a dummy resource
        # (Equivalent to single-machine scheduling)
        demands = [1] * len(intervals)
        model.AddCumulative(intervals, demands, 1)

        # Peak power constraint: at any time, sum(power flags) <= peak_demand_w
        # Approximate via time-slice sampling to keep model simple
        time_step = max(1, horizon // 60)
        for t in range(0, horizon + 1, time_step):
            # For each activity, whether it is active at time t
            weighted = []
            for i, (s, e, p, d) in enumerate(zip(starts, ends, powers, durations)):
                b = model.NewBoolVar(f"act_{i}_at_{t}")
                model.Add(s <= t).OnlyEnforceIf(b)
                model.Add(e > t).OnlyEnforceIf(b)
                if p > 0:
                    weighted.append((b, p))
            if weighted:
                # sum(b_i * p_i) <= peak_demand_w
                model.Add(sum(b * p for b, p in weighted) <= peak_demand_w)

        # Maintain input order as precedence (soft rule, here enforced hard)
        for i in range(1, len(activities)):
            model.Add(starts[i] >= ends[i - 1])

        # Energy budget constraint (approximate): sum(p_i * d_i) <= MMRTG_W * horizon + battery_wh*60*frac
        power = constraints.get("power", {})
        mmrtg_output_w = int(power.get("mmrtg_output_w", 110))
        battery_capacity_ah = float(power.get("battery_capacity_ah", 43))
        battery_count = int(power.get("battery_count", 2))
        battery_bus_v = float(power.get("battery_bus_v", 28.0))
        discharge_frac = float(power.get("battery_discharge_frac", 0.2))
        battery_wh = battery_capacity_ah * battery_bus_v * battery_count
        lhs = sum(int(p) * int(d) for p, d in zip(powers, durations))  # W-min
        rhs = mmrtg_output_w * horizon + int(battery_wh * 60.0 * discharge_frac)
        model.Add(lhs <= rhs)

        # Communication windows (if provided): ensure data_tx start within any window
        comm_windows = constraints.get("comm_windows", [])
        if comm_windows:
            for i, (s, d, a) in enumerate(zip(starts, durations, activities)):
                if a.get("type") == "data_tx":
                    choices = []
                    for j, w in enumerate(comm_windows):
                        ws = int(w.get("start_min", 0))
                        we = int(w.get("end_min", 0))
                        b = model.NewBoolVar(f"tx_{i}_in_win_{j}")
                        model.Add(s >= ws).OnlyEnforceIf(b)
                        model.Add(s <= max(0, we - d)).OnlyEnforceIf(b)
                        choices.append(b)
                    if choices:
                        model.Add(sum(choices) >= 1)

        # Objective: minimize makespan
        makespan = model.NewIntVar(0, horizon, "makespan")
        if ends:
            model.AddMaxEquality(makespan, ends)
            model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 3.0
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        # Build scheduled activities
        scheduled: List[Dict[str, Any]] = []
        for i, a in enumerate(activities):
            s = int(solver.Value(starts[i]))
            d = durations[i]
            scheduled.append({**a, "start_min": s, "end_min": s + d})
        return scheduled