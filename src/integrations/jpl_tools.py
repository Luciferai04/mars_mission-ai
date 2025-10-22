#!/usr/bin/env python3
"""
Adapters for JPL planning tools (simplified).

Provides exporters to hypothetical APGEN and PLEXIL-like formats.
"""
from __future__ import annotations

from typing import Dict, Any


def export_apgen(plan: Dict[str, Any]) -> str:
    # Produce a minimal JSON-like string that could be adapted
    return "APGEN_EXPORT\n" + _pretty(plan)


def export_plexil(plan: Dict[str, Any]) -> str:
    # Produce a minimal XML-like structure
    lines = ["<PlexilPlan>"]
    lines.append("  <PlanId>" + str(plan.get("plan_id", "PLAN")) + "</PlanId>")
    lines.append("  <Activities>")
    for i, a in enumerate(plan.get("optimized_actions", []), 1):
        lines.append(f"    <Activity id=\"{i}\" type=\"{a.get('type','act')}\">")
        if a.get("target"):
            t = a["target"]
            lines.append(f'      <Target lat="{t[0]}" lon="{t[1]}"/>')
        lines.append(f"      <Duration>{a.get('duration',0)}</Duration>")
        lines.append("    </Activity>")
    lines.append("  </Activities>")
    lines.append("</PlexilPlan>")
    return "\n".join(lines)


def _pretty(obj: Any, indent: int = 0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        lines = [sp + "{"]
        for k, v in obj.items():
            lines.append(sp + f"  {k}: " + _pretty(v, indent + 1).lstrip())
        lines.append(sp + "}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = [sp + "["]
        for v in obj:
            lines.append(_pretty(v, indent + 1))
        lines.append(sp + "]")
        return "\n".join(lines)
    return sp + repr(obj)
