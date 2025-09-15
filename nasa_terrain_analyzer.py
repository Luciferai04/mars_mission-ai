#!/usr/bin/env python3
"""
NASA-Standard GPT-4V Terrain Analyzer for Mars Mission Planning

Provides comprehensive AI-powered analysis of Mars rover imagery using GPT-4V:
- NASA operational safety assessment
- Scientific target identification and prioritization
- Traversability analysis with energy estimates
- Route recommendations with operational constraints

Implements NASA JPL mission planning standards and Perseverance operational limits.
"""

import os
import base64
import requests
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime
from PIL import Image
import numpy as np


class NASATerrainAnalyzer:
    """NASA-standard terrain analyzer using GPT-4V for Mars mission planning."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required for NASA terrain analysis")
        
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4-vision-preview"
        self.max_tokens = 4096
        self.temperature = 0.1  # Low temperature for mission-critical analysis
        
        self.logger = logging.getLogger(__name__)
        
        # NASA Perseverance operational constraints (from JPL specifications)
        self.PERSEVERANCE_CONSTRAINTS = {
            "max_slope_degrees": 30.0,          # Absolute maximum safe slope
            "preferred_slope_limit": 15.0,      # Preferred operational limit
            "max_rock_height_cm": 50.0,         # Maximum traversable rock height
            "ground_clearance_cm": 35.0,        # Rover belly clearance
            "wheel_diameter_cm": 52.5,          # Wheel diameter (affects climbing)
            "wheelbase_cm": 300.0,              # Distance between front/rear wheels
            "max_daily_drive_m": 200.0,         # Typical daily drive limit
            "autonomous_drive_limit_m": 100.0,  # Auto-nav limit per drive
            "power_drive_wh_per_m": 15.0,       # Approximate energy per meter
            "sample_arm_reach_cm": 225.0        # Max reach for sampling
        }
    
    def analyze_rover_imagery(self, image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive NASA-standard terrain analysis using GPT-4V.
        
        Args:
            image_path: Path to Mars rover image (Mastcam-Z, Navcam, etc.)
            context: Mission context including sol, position, environmental data
            
        Returns:
            Structured analysis compatible with NASA mission planning tools
        """
        try:
            self.logger.info(f"Starting NASA terrain analysis for Sol {context.get('sol', 'unknown')}")
            
            # Validate inputs
            if not os.path.exists(image_path):
                return self._create_error_result(f"Image not found: {image_path}")
            
            # Prepare image for analysis
            encoded_image = self._encode_image_for_api(image_path)
            
            # Generate NASA operational analysis prompt
            prompt = self._generate_nasa_analysis_prompt(context)
            
            # Perform GPT-4V analysis
            api_response = self._call_gpt4v_api(encoded_image, prompt)
            
            # Parse and validate response
            analysis_result = self._parse_nasa_analysis_response(api_response, context)
            
            # Add metadata and validation
            analysis_result['metadata'] = {
                'image_path': image_path,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'model_version': self.model,
                'nasa_constraints_applied': True,
                'perseverance_specifications': self.PERSEVERANCE_CONSTRAINTS
            }
            
            self.logger.info(f"NASA terrain analysis completed successfully")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"NASA terrain analysis failed: {e}")
            return self._create_error_result(str(e))
    
    def _encode_image_for_api(self, image_path: str) -> str:
        """Encode and optimize image for GPT-4V API transmission."""
        try:
            with Image.open(image_path) as img:
                # Optimize image size for API limits while preserving detail
                original_size = img.size
                
                # Resize if too large (API limit ~20MB, optimal around 2048px)
                if img.size[0] > 2048 or img.size[1] > 2048:
                    img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                    self.logger.info(f"Resized image from {original_size} to {img.size}")
                
                # Ensure RGB format
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Optimize quality vs size
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=90, optimize=True)
                image_bytes = buffer.getvalue()
                
                self.logger.info(f"Image encoded: {len(image_bytes)/1024:.1f} KB")
                return base64.b64encode(image_bytes).decode('utf-8')
                
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {e}")
    
    def _generate_nasa_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Generate comprehensive NASA operational analysis prompt."""
        
        sol = context.get('sol', 0)
        latitude = context.get('latitude', 0.0)
        longitude = context.get('longitude', 0.0)
        mission_phase = context.get('mission_phase', 'operations')
        temperature = context.get('temperature', 'unknown')
        pressure = context.get('pressure', 'unknown')
        wind_speed = context.get('wind_speed', 'unknown')
        
        return f"""You are a senior NASA JPL Mars mission planner analyzing Perseverance rover imagery for Sol {sol} operational planning. You have responsibility for safe operations of a $2.7 billion Mars rover.

MISSION CONTEXT:
- Sol: {sol}
- Location: {latitude:.6f}°N, {longitude:.6f}°E
- Mission Phase: {mission_phase}
- Environmental: {temperature}°C, {pressure} Pa, {wind_speed} m/s

ANALYSIS DIRECTIVE:
Perform comprehensive terrain analysis with NASA operational standards. Your analysis will directly inform rover operations and crew safety decisions.

REQUIRED ANALYSIS COMPONENTS:

1. OPERATIONAL SAFETY ASSESSMENT:
   - Identify all potential hazards using Perseverance safety protocols
   - Assess slope angles against 30° absolute limit and 15° preferred limit
   - Evaluate rock/obstacle sizes against 50cm traversable height and 35cm clearance
   - Identify cliff edges, drop-offs, and unstable terrain
   - Assess surface material: bedrock, sand, loose rocks, dust
   - Rate terrain stability for drilling/sampling operations
   - Overall safety classification: GO, CAUTION, NO-GO

2. SCIENTIFIC TARGET ANALYSIS:
   - Identify geological features of scientific interest
   - Classify rock types, formations, layered deposits, mineral veins
   - Assess sample collection feasibility with 225cm arm reach
   - Evaluate drilling sites for sample collection
   - Prioritize targets by: scientific value, accessibility, operational risk
   - Identify ChemCam/SuperCam laser targets
   - Note any unusual or high-priority geological formations

3. TRAVERSABILITY AND ROUTE PLANNING:
   - Assess overall terrain difficulty for rover mobility
   - Identify optimal paths considering wheel-ground interaction
   - Evaluate energy requirements using 15 Wh/meter baseline
   - Assess autonomous navigation feasibility
   - Identify waypoints and intermediate targets
   - Note areas requiring detailed path planning or manual control
   - Estimate drive complexity: simple, moderate, complex, extreme

4. OPERATIONAL RECOMMENDATIONS:
   - Specific driving directions and approach angles
   - Recommended drive distances and waypoint spacing
   - Contingency plans for blocked routes
   - Instrument deployment recommendations
   - Risk mitigation strategies
   - Timeline estimates for proposed activities

PERSEVERANCE ROVER SPECIFICATIONS (CRITICAL CONSTRAINTS):
- Maximum safe slope: {self.PERSEVERANCE_CONSTRAINTS['max_slope_degrees']}° (absolute limit)
- Preferred slope limit: {self.PERSEVERANCE_CONSTRAINTS['preferred_slope_limit']}° (operational preference)
- Maximum rock height: {self.PERSEVERANCE_CONSTRAINTS['max_rock_height_cm']}cm (wheel climbing limit)
- Ground clearance: {self.PERSEVERANCE_CONSTRAINTS['ground_clearance_cm']}cm (belly clearance)
- Wheel diameter: {self.PERSEVERANCE_CONSTRAINTS['wheel_diameter_cm']}cm
- Wheelbase: {self.PERSEVERANCE_CONSTRAINTS['wheelbase_cm']}cm
- Daily drive limit: {self.PERSEVERANCE_CONSTRAINTS['max_daily_drive_m']}m typical
- Auto-nav limit: {self.PERSEVERANCE_CONSTRAINTS['autonomous_drive_limit_m']}m per segment
- Sample arm reach: {self.PERSEVERANCE_CONSTRAINTS['sample_arm_reach_cm']}cm

RESPONSE FORMAT REQUIREMENT:
Provide analysis in structured JSON format for mission planning systems:

{{
  "operational_safety": {{
    "overall_classification": "GO|CAUTION|NO-GO",
    "hazards_identified": [
      {{
        "type": "slope|rock|cliff|sand|unstable_surface",
        "severity": "low|medium|high|critical",
        "location": "description of location in image",
        "mitigation": "how to avoid or manage this hazard"
      }}
    ],
    "slope_analysis": {{
      "max_visible_slope": "degrees estimate",
      "areas_exceeding_15_degrees": "description",
      "areas_exceeding_30_degrees": "description",
      "overall_slope_assessment": "within_limits|approaching_limits|exceeds_limits"
    }},
    "surface_assessment": {{
      "material_type": "bedrock|sand|gravel|mixed",
      "stability": "stable|moderate|unstable",
      "trafficability": "good|moderate|poor|impassable"
    }}
  }},
  "scientific_analysis": {{
    "targets_identified": [
      {{
        "target_id": "T1, T2, etc.",
        "type": "rock|formation|deposit|layering|vein",
        "scientific_priority": "high|medium|low",
        "accessibility": "easy|moderate|difficult|impossible",
        "sampling_feasibility": "excellent|good|challenging|not_feasible",
        "location_description": "precise location in image",
        "scientific_rationale": "why scientifically interesting",
        "instrument_recommendations": "PIXL|SHERLOC|SUPERCAM|MASTCAM-Z|drill"
      }}
    ],
    "geological_summary": "overall geological context",
    "sampling_recommendations": "priority order for sample collection"
  }},
  "mobility_analysis": {{
    "traversability_score": 0.0-1.0,
    "terrain_difficulty": "easy|moderate|difficult|extreme|impassable",
    "optimal_path": {{
      "description": "detailed path recommendation",
      "waypoints": ["list of navigation waypoints"],
      "total_distance": "estimated meters",
      "energy_estimate": "low|medium|high|very_high",
      "drive_time": "estimated minutes"
    }},
    "alternative_routes": [
      "backup path options if primary blocked"
    ],
    "autonomous_nav_assessment": "full_auto|waypoint_guidance|manual_control"
  }},
  "operational_recommendations": [
    "Specific actionable recommendations for mission operations team",
    "Include approach angles, drive distances, instrument usage",
    "Risk mitigation strategies and contingency plans"
  ],
  "mission_planning_notes": {{
    "confidence_level": 0.0-1.0,
    "analysis_limitations": "what cannot be determined from this image",
    "additional_data_needed": "what other information would improve analysis",
    "weather_considerations": "how environmental conditions affect recommendations"
  }}
}}

ANALYSIS STANDARDS:
- Apply NASA operational safety margins and conservative estimates
- Consider Perseverance's proven operational capabilities and limitations
- Prioritize crew safety and rover preservation over science opportunities
- Use terminology and classifications consistent with JPL mission operations
- Provide actionable recommendations suitable for Sol planning meetings

Analyze this image with the thoroughness, precision, and conservative approach that has enabled successful Mars rover operations for over two decades."""

    def _call_gpt4v_api(self, encoded_image: str, prompt: str) -> Dict[str, Any]:
        """Execute GPT-4V API call with error handling and retry logic."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a NASA JPL Mars mission planner with expertise in rover operations, geology, and planetary science. Provide precise, actionable analysis for mission-critical operations."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                                "detail": "high"  # Maximum detail for terrain analysis
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }
        
        self.logger.info(f"Calling GPT-4V API for NASA terrain analysis")
        
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=180  # Extended timeout for detailed analysis
            )
            response.raise_for_status()
            
            api_response = response.json()
            
            # Log usage statistics
            usage = api_response.get('usage', {})
            self.logger.info(f"API usage: {usage.get('total_tokens', 0)} tokens, "
                           f"${usage.get('total_tokens', 0) * 0.00002:.4f} cost estimate")
            
            return api_response
            
        except requests.RequestException as e:
            self.logger.error(f"GPT-4V API request failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"GPT-4V API call failed: {e}")
            raise
    
    def _parse_nasa_analysis_response(self, api_response: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse GPT-4V response into NASA mission planning format."""
        try:
            content = api_response['choices'][0]['message']['content']
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                analysis_data = json.loads(json_str)
                
                # Convert to mission planning interface format
                result = {
                    # Core analysis components
                    'hazards_detected': self._extract_hazards(analysis_data),
                    'science_targets': analysis_data.get('scientific_analysis', {}).get('targets_identified', []),
                    'traversability_score': analysis_data.get('mobility_analysis', {}).get('traversability_score', 0.5),
                    'recommendations': analysis_data.get('operational_recommendations', []),
                    
                    # NASA-specific classifications
                    'safety_classification': analysis_data.get('operational_safety', {}).get('overall_classification', 'CAUTION'),
                    'terrain_difficulty': analysis_data.get('mobility_analysis', {}).get('terrain_difficulty', 'moderate'),
                    'autonomous_nav_capability': analysis_data.get('mobility_analysis', {}).get('autonomous_nav_assessment', 'waypoint_guidance'),
                    
                    # Operational planning data
                    'optimal_path': analysis_data.get('mobility_analysis', {}).get('optimal_path', {}),
                    'energy_estimate': analysis_data.get('mobility_analysis', {}).get('optimal_path', {}).get('energy_estimate', 'medium'),
                    'drive_time_estimate': analysis_data.get('mobility_analysis', {}).get('optimal_path', {}).get('drive_time', 'unknown'),
                    
                    # Quality metrics
                    'confidence': analysis_data.get('mission_planning_notes', {}).get('confidence_level', 0.7),
                    'analysis_limitations': analysis_data.get('mission_planning_notes', {}).get('analysis_limitations', []),
                    
                    # Complete structured analysis
                    'detailed_nasa_analysis': analysis_data,
                    'status': 'success'
                }
                
                # Validate critical safety information
                self._validate_safety_analysis(result)
                
                return result
                
            else:
                raise json.JSONDecodeError("No valid JSON found in response", content, 0)
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse structured response: {e}")
            # Fallback to text-based parsing
            return self._parse_unstructured_response(content, context)
    
    def _extract_hazards(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Extract hazard list for mission planning compatibility."""
        hazards = []
        safety_data = analysis_data.get('operational_safety', {})
        
        for hazard in safety_data.get('hazards_identified', []):
            if isinstance(hazard, dict):
                hazard_desc = f"{hazard.get('type', 'unknown')} ({hazard.get('severity', 'unknown')} severity)"
                hazards.append(hazard_desc)
            else:
                hazards.append(str(hazard))
        
        return hazards
    
    def _validate_safety_analysis(self, result: Dict[str, Any]) -> None:
        """Validate safety analysis meets NASA standards."""
        # Ensure conservative classification if high-risk hazards detected
        safety_class = result.get('safety_classification', 'CAUTION')
        hazards = result.get('hazards_detected', [])
        
        # Apply conservative safety override
        high_risk_indicators = ['critical', 'exceeds_limits', 'cliff', 'unstable']
        if any(indicator in str(hazards).lower() for indicator in high_risk_indicators):
            if safety_class == 'GO':
                result['safety_classification'] = 'CAUTION'
                result['recommendations'].insert(0, 'SAFETY OVERRIDE: Classification elevated to CAUTION due to identified risks')
        
        # Ensure minimum confidence for safety-critical decisions
        if result.get('confidence', 0) < 0.5 and safety_class == 'GO':
            result['safety_classification'] = 'CAUTION'
            result['recommendations'].insert(0, 'CONFIDENCE OVERRIDE: Low confidence requires CAUTION classification')
    
    def _parse_unstructured_response(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback parsing for unstructured responses."""
        self.logger.warning("Using fallback text parsing for terrain analysis")
        
        content_lower = content.lower()
        
        # Extract safety assessment
        safety_classification = 'CAUTION'  # Conservative default
        if 'no-go' in content_lower or 'prohibited' in content_lower:
            safety_classification = 'NO-GO'
        elif 'go' in content_lower and 'safe' in content_lower:
            safety_classification = 'GO'
        
        # Extract mentioned hazards
        hazards = []
        hazard_keywords = ['slope', 'rock', 'boulder', 'cliff', 'sand', 'unstable', 'rough']
        for keyword in hazard_keywords:
            if keyword in content_lower:
                hazards.append(f"Potential {keyword} hazard identified")
        
        # Basic confidence assessment
        confidence = 0.4  # Low confidence for unstructured parsing
        if 'confident' in content_lower or 'clear' in content_lower:
            confidence = 0.6
        elif 'uncertain' in content_lower or 'unclear' in content_lower:
            confidence = 0.2
        
        return {
            'hazards_detected': hazards,
            'science_targets': [],
            'traversability_score': 0.5,
            'recommendations': ['Manual analysis recommended - structured parsing failed'],
            'safety_classification': safety_classification,
            'terrain_difficulty': 'moderate',
            'confidence': confidence,
            'raw_text_response': content,
            'status': 'partial_success',
            'parsing_method': 'unstructured_fallback'
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error result for mission planning systems."""
        return {
            'hazards_detected': [],
            'science_targets': [],
            'traversability_score': 0.0,
            'recommendations': [f'Terrain analysis failed: {error_msg}', 'Manual terrain assessment required'],
            'safety_classification': 'NO-GO',  # Conservative error handling
            'terrain_difficulty': 'unknown',
            'autonomous_nav_capability': 'manual_control',
            'confidence': 0.0,
            'error': error_msg,
            'status': 'failed',
            'requires_manual_review': True
        }