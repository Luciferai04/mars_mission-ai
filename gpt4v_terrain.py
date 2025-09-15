from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import base64
import os

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class TerrainAnalyzer:
    """LLM Vision interface for terrain analysis.

    This layer delegates to an external multimodal model (e.g., gpt-4o) to:
    - Identify hazards, traversable paths, and science targets from Mastcam-Z images
    - Summarize DEM-derived risks into structured JSON for planning

    Networking/LLM calls are not performed at import; methods accept preloaded data/paths.
    """

    model: str = "gpt-4o-2024-08-06"

    def _encode_image_b64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def analyze_mastcam_image(self, image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI vision if available; otherwise return a stub structure.

        Does not run at import; requires OPENAI_API_KEY in env to actually call API.
        """
        use_openai = os.getenv("OPENAI_API_KEY") and OpenAI is not None
        if use_openai:
            try:
                client = OpenAI()
                b64 = self._encode_image_b64(image_path)
                prompt = (
                    "Analyze this Mars rover image as a NASA mission planner would. "
                    "Return JSON with keys safety, science, routes, resource_impact."
                )
                # Using responses API (if available) or chat.completions fallback
                # Here we use chat.completions with image content
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a Mars mission planner."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                },
                            ],
                        },
                    ],
                    temperature=0.1,
                )
                text = resp.choices[0].message.content or "{}"
                # We do not eval; return raw text in analysis.notes and leave structure for caller
                return {
                    "image_path": image_path,
                    "context": context,
                    "analysis": {
                        "safety": {"hazards": [], "notes": text},
                        "science": {"targets": [], "priorities": []},
                        "routes": [],
                        "resource_impact": {"power_wh": None, "time_min": None},
                    },
                }
            except Exception as e:  # pragma: no cover
                pass
        # Fallback stub
        return {
            "image_path": image_path,
            "context": context,
            "analysis": {
                "safety": {"hazards": [], "notes": "stub"},
                "science": {"targets": [], "priorities": []},
                "routes": [],
                "resource_impact": {"power_wh": None, "time_min": None},
            },
        }

    def process_elevation_model(self, dem_stats: Dict[str, Any]) -> Dict[str, Any]:
        return {"hazard_map": None, "notes": "stub"}
