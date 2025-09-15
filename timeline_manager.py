from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TimelineManager:
    """Manage activity scheduling on a resource timeline.

    Ensures activities are ordered and scheduled without overlaps and
    respecting resource constraints.
    """

    def schedule(self, activities: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Placeholder: return as-is with naive timestamps
        t = 0
        scheduled = []
        for act in activities:
            duration = int(act.get("duration_min", 15))
            scheduled.append({**act, "start_min": t, "end_min": t + duration})
            t += duration
        return scheduled
