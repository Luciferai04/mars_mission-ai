#!/usr/bin/env python3
"""
Production GPT-4o Vision Terrain Analyzer

Real implementation for analyzing Mastcam-Z, Hazcam, and Navcam imagery
for hazard detection, science targets, and traversability assessment.
"""

import os
import base64
import requests
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime


class GPT4oVisionAnalyzer:
    """Production-ready GPT-4o Vision analyzer for Mars rover imagery."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for vision analysis")

        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o"  # Production model
        self.max_tokens = 4096
        self.temperature = 0.1

        self.logger = logging.getLogger(__name__)

    def analyze_terrain_image(
        self,
        image_path: str,
        camera_type: str,
        sol: int,
        coordinates: tuple,
        environmental_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze Mars terrain image for hazards, targets, and traversability.

        Args:
            image_path: Path to rover image
            camera_type: MASTCAM_Z, HAZCAM, or NAVCAM
            sol: Mars sol number
            coordinates: (lat, lon) tuple
            environmental_context: Temperature, dust, wind data

        Returns:
            Comprehensive terrain analysis with NASA-grade assessments
        """
        self.logger.info(f"Analyzing {camera_type} image from Sol {sol}")

        # Encode image
        encoded_image = self._encode_image(image_path)

        # Generate NASA-aligned prompt
        prompt = self._generate_vision_prompt(
            camera_type, sol, coordinates, environmental_context
        )

        # Call GPT-4o Vision API
        response = self._call_vision_api(encoded_image, prompt)

        # Parse and validate response
        analysis = self._parse_vision_response(response, camera_type)

        # Add metadata
        analysis["metadata"] = {
            "analyzed_at": datetime.utcnow().isoformat(),
            "model": self.model,
            "camera_type": camera_type,
            "sol": sol,
            "coordinates": coordinates,
            "image_path": image_path,
        }

        return analysis

    def batch_analyze_images(
        self, images: List[Dict[str, Any]], mission_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze multiple images for comprehensive mission planning.

        Args:
            images: List of image dicts with paths and metadata
            mission_context: Overall mission parameters

        Returns:
            List of analyses with cross-image insights
        """
        analyses = []

        for img_data in images:
            analysis = self.analyze_terrain_image(
                image_path=img_data["path"],
                camera_type=img_data.get("camera_type", "MASTCAM_Z"),
                sol=img_data.get("sol", mission_context.get("sol", 0)),
                coordinates=img_data.get("coordinates", (0, 0)),
                environmental_context=mission_context.get("environment", {}),
            )
            analyses.append(analysis)

        # Generate cross-image summary
        summary = self._generate_multi_image_summary(analyses)

        return {
            "individual_analyses": analyses,
            "cross_image_summary": summary,
            "recommendation": self._generate_mission_recommendation(analyses),
        }

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _generate_vision_prompt(
        self,
        camera_type: str,
        sol: int,
        coordinates: tuple,
        env_context: Dict[str, Any],
    ) -> str:
        """Generate NASA-aligned vision analysis prompt."""

        camera_specs = {
            "MASTCAM_Z": {
                "fov": "25.6°",
                "resolution": "1600x1200",
                "purpose": "high-resolution science imaging",
                "focus": "science targets, geological features, sample sites",
            },
            "HAZCAM": {
                "fov": "124°",
                "resolution": "1024x1024",
                "purpose": "hazard detection near rover",
                "focus": "immediate hazards, rocks, slopes, wheel clearance",
            },
            "NAVCAM": {
                "fov": "45°",
                "resolution": "1024x1024",
                "purpose": "navigation and path planning",
                "focus": "traversable paths, distant hazards, waypoint visibility",
            },
        }

        spec = camera_specs.get(camera_type, camera_specs["MASTCAM_Z"])

        return f"""You are a NASA Mars rover mission planner analyzing {camera_type} imagery from Perseverance.

**MISSION CONTEXT:**
- Sol: {sol}
- Location: {coordinates[0]:.6f}°N, {coordinates[1]:.6f}°E (Jezero Crater)
- Temperature: {env_context.get('temperature_c', -60)}°C
- Dust opacity: {env_context.get('dust_opacity', 0.5)}
- Wind: {env_context.get('wind_speed_ms', 5)} m/s

**CAMERA SPECIFICATIONS ({camera_type}):**
- Field of view: {spec['fov']}
- Resolution: {spec['resolution']}
- Purpose: {spec['purpose']}
- Analysis focus: {spec['focus']}

**ANALYSIS REQUIREMENTS:**

Provide a structured NASA-grade terrain analysis with:

1. **HAZARD ASSESSMENT** (Critical for rover safety):
   - Rock sizes and distribution (>50cm are hazards)
   - Slope angles (>30° prohibited, 15-30° caution, <15° safe)
   - Surface roughness and wheel hazards
   - Cliff edges, trenches, or drop-offs
   - Loose/sandy terrain risk
   - Overall safety rating: SAFE | CAUTION | HAZARD

2. **SCIENCE TARGETS** (For sample collection priority):
   - Interesting geological formations
   - Rock types and compositions (visual assessment)
   - Potential sampling sites
   - Scientific priority: HIGH | MEDIUM | LOW
   - Accessibility rating for each target

3. **TRAVERSABILITY ANALYSIS**:
   - Recommended paths through visible terrain
   - Drive distance estimates
   - Energy cost assessment: LOW | MEDIUM | HIGH
   - Waypoint suggestions
   - Confidence in route viability

4. **OPERATIONAL RECOMMENDATIONS**:
   - Suggested rover actions (approach, sample, avoid, investigate)
   - Required instrument deployments
   - Contingency considerations
   - Time estimates for activities

**OUTPUT FORMAT:**
Return analysis as JSON:
{{
  "hazard_assessment": {{
    "overall_safety": "SAFE|CAUTION|HAZARD",
    "hazards": [
      {{"type": "rock", "size_cm": 60, "location": "center-left", "severity": "high"}},
      {{"type": "slope", "angle_deg": 25, "location": "right-side", "severity": "medium"}}
    ],
    "safety_notes": ["specific concerns"],
    "confidence": 0.0-1.0
  }},
  "science_targets": [
    {{
      "target_id": 1,
      "type": "layered_rock",
      "location": "foreground_center",
      "priority": "HIGH",
      "accessibility": "easy",
      "scientific_value": "potential delta deposit",
      "sampling_feasible": true
    }}
  ],
  "traversability": {{
    "overall_difficulty": "easy|moderate|difficult",
    "recommended_paths": ["path description"],
    "drive_distance_estimate_m": 50,
    "energy_cost": "low|medium|high",
    "waypoints": [coordinates],
    "confidence": 0.0-1.0
  }},
  "operational_recommendations": [
    "Approach target 1 from north to minimize slope",
    "Deploy PIXL and SHERLOC for detailed analysis",
    "Maintain 2m safety margin from rocks >50cm"
  ]
}}

**NASA SAFETY STANDARDS:**
- Be conservative: when in doubt, rate as HAZARD
- Never recommend actions that exceed rover capabilities
- Flag any uncertainty or ambiguity
- Prioritize rover preservation over aggressive science

Analyze the image with mission-critical precision."""

    def _call_vision_api(self, encoded_image: str, prompt: str) -> Dict[str, Any]:
        """Call GPT-4o Vision API with image and prompt."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a NASA Mars rover mission operations AI analyzing terrain imagery with expert precision.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                self.base_url, headers=headers, json=payload, timeout=120
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            self.logger.error(f"GPT-4o Vision API call failed: {e}")
            raise

    def _parse_vision_response(
        self, response: Dict[str, Any], camera_type: str
    ) -> Dict[str, Any]:
        """Parse and validate GPT-4o vision response."""

        try:
            content = response["choices"][0]["message"]["content"]

            # Extract JSON
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                analysis = json.loads(content[json_start:json_end])

                # Validate required fields
                required = ["hazard_assessment", "traversability"]
                for field in required:
                    if field not in analysis:
                        self.logger.warning(f"Missing field: {field}")
                        analysis[field] = self._get_default_field(field)

                return analysis
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            self.logger.error(f"Failed to parse vision response: {e}")
            return self._get_fallback_analysis(camera_type)

    def _get_default_field(self, field: str) -> Dict[str, Any]:
        """Get default value for missing field."""
        defaults = {
            "hazard_assessment": {
                "overall_safety": "UNKNOWN",
                "hazards": [],
                "safety_notes": ["Analysis incomplete"],
                "confidence": 0.0,
            },
            "traversability": {
                "overall_difficulty": "unknown",
                "recommended_paths": [],
                "confidence": 0.0,
            },
            "science_targets": [],
        }
        return defaults.get(field, {})

    def _get_fallback_analysis(self, camera_type: str) -> Dict[str, Any]:
        """Fallback analysis when API fails."""
        return {
            "hazard_assessment": {
                "overall_safety": "UNKNOWN",
                "hazards": [],
                "safety_notes": ["Vision analysis failed - manual review required"],
                "confidence": 0.0,
            },
            "science_targets": [],
            "traversability": {
                "overall_difficulty": "unknown",
                "recommended_paths": ["Manual assessment required"],
                "confidence": 0.0,
            },
            "operational_recommendations": [
                "GPT-4o Vision analysis unavailable",
                "Proceed with manual terrain assessment",
                "Do not execute autonomous operations",
            ],
            "error": "Vision API failure",
        }

    def _generate_multi_image_summary(
        self, analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary across multiple images."""

        total_hazards = sum(
            len(a.get("hazard_assessment", {}).get("hazards", [])) for a in analyses
        )

        total_targets = sum(len(a.get("science_targets", [])) for a in analyses)

        # Aggregate safety ratings
        safety_ratings = [
            a.get("hazard_assessment", {}).get("overall_safety", "UNKNOWN")
            for a in analyses
        ]

        most_restrictive = "SAFE"
        if "HAZARD" in safety_ratings:
            most_restrictive = "HAZARD"
        elif "CAUTION" in safety_ratings:
            most_restrictive = "CAUTION"

        return {
            "total_images_analyzed": len(analyses),
            "total_hazards_identified": total_hazards,
            "total_science_targets": total_targets,
            "overall_safety_rating": most_restrictive,
            "recommendation": self._get_aggregate_recommendation(most_restrictive),
        }

    def _generate_mission_recommendation(
        self, analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate overall mission recommendation."""

        summary = self._generate_multi_image_summary(analyses)

        if summary["overall_safety_rating"] == "HAZARD":
            return {
                "proceed": False,
                "reason": "Hazardous terrain identified",
                "alternative_action": "Re-route or conduct detailed analysis from safe position",
            }
        elif summary["overall_safety_rating"] == "CAUTION":
            return {
                "proceed": True,
                "conditions": [
                    "Reduce speed",
                    "Monitor wheel currents",
                    "Ready abort sequence",
                ],
                "reason": "Traversable with elevated monitoring",
            }
        else:
            return {
                "proceed": True,
                "reason": "Safe terrain conditions",
                "optimization": f'Prioritize {summary["total_science_targets"]} science targets',
            }

    def _get_aggregate_recommendation(self, safety_rating: str) -> str:
        """Get recommendation based on safety rating."""
        recommendations = {
            "SAFE": "Proceed with planned operations",
            "CAUTION": "Proceed with enhanced monitoring and reduced speed",
            "HAZARD": "Do not proceed - replan route or conduct detailed remote sensing",
            "UNKNOWN": "Manual review required before proceeding",
        }
        return recommendations.get(safety_rating, "Manual review required")
