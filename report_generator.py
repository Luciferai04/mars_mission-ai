from __future__ import annotations

from typing import Any, Dict, List


def _fmt_minutes(m: float | int) -> str:
    return f"{int(m)} min"


def make_report(plan: Dict[str, Any]) -> str:
    goals = plan.get("goals", [])
    power = plan.get("power", {})
    route = plan.get("route")
    activities: List[Dict[str, Any]] = plan.get("activities", [])

    lines: List[str] = []
    lines.append("Mars Mission Plan Report")
    lines.append("========================\n")
    if goals:
        lines.append("Objectives:")
        for g in goals:
            lines.append(f"- {g}")
        lines.append("")

    if route:
        if route.get("success"):
            steps = len(route.get("path_rc", []))
            lines.append(f"Route: success with {steps} steps (row/col path elided)")
        else:
            lines.append("Route: no feasible path found")
        lines.append("")

    if power:
        lines.append("Power Budget:")
        lines.append(f"- Budget: {power.get('budget_wh', 'n/a'):.1f} Wh")
        lines.append(f"- Consumed: {power.get('consumed_wh', 'n/a'):.1f} Wh")
        lines.append("")

    lines.append("Activity Timeline:")
    for a in activities:
        start = a.get("start_min", 0)
        end = a.get("end_min", 0)
        duration = a.get("duration_min", end - start)
        tag = a.get("type", a.get("id", "activity"))
        status = a.get("status")
        s = f"- {tag}: {start}-{end} ({_fmt_minutes(duration)})"
        if status:
            s += f" [{status}]"
        lines.append(s)

    return "\n".join(lines) + "\n"