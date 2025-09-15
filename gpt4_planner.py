from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import os

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class MissionPlannerLLM:
    """LLM-driven mission planning interface.

    Generates draft activity sequences using structured prompts, to be
    validated and repaired by the constraint/timeline layers.
    """

    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.1
    max_tokens: int = 4096

    def generate_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI LLM to draft plan if available; fallback to stub activities."""
        use_openai = os.getenv("OPENAI_API_KEY") and OpenAI is not None
        if use_openai:
            try:
                client = OpenAI()
                system = "You generate conservative, NASA-style rover activity sequences in JSON."
                objectives = context.get("objectives", [])
                constraints = context.get("constraints", {})
                rover_state = context.get("rover_state", {})
                user = (
                    f"OBJECTIVES: {objectives}\nCONSTRAINTS: {constraints}\nROVER_STATE: {rover_state}\n"
                    "Return JSON with a key 'activities' as a list of activities, each with id, type, and duration_min;"
                    " include a data_tx at the end."
                )
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content or ""
                # Do not parse blindly; attach LLM output for operator review
                return {
                    "objectives": objectives,
                    "constraints": constraints,
                    "rover_state": rover_state,
                    "llm_draft": text,
                    "activities": [
                        {"id": "pre_checks", "type": "system_checks", "duration_min": 10},
                        {"id": "post_tx", "type": "data_tx", "duration_min": 5},
                    ],
                }
            except Exception:  # pragma: no cover
                pass
        # Fallback stub
        return {
            "objectives": context.get("objectives", []),
            "constraints": context.get("constraints", {}),
            "rover_state": context.get("rover_state", {}),
            "activities": [
                {"id": "pre_checks", "type": "system_checks", "duration_min": 10},
                {"id": "drive_segment", "type": "drive", "distance_m": 50},
                {"id": "science_op", "type": "imaging", "instrument": "Mastcam-Z"},
                {"id": "post_tx", "type": "data_tx", "duration_min": 5},
            ],
        }
