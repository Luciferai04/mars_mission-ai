#!/usr/bin/env python3
"""
GPT-4V Terrain Analysis for Mars Mission Planning

Provides AI-powered analysis of Mars rover imagery for:
- Hazard detection and safety assessment
- Scientific target identification
- Traversability analysis
- Route recommendation

Based on NASA operational procedures and mission planner workflows.
"""

import os
import base64
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from PIL import Image
import numpy as np


@dataclass
class TerrainAnalysisRequest:
    """Request structure for terrain analysis."""
    image_path: str
    sol: int
    coordinates: Tuple[float, float]  # (latitude, longitude)
    mission_phase: str
    environmental_conditions: Dict[str, Any]
    analysis_type: str = "comprehensive"  # comprehensive, hazard, science, traverse


@dataclass
class TerrainAnalysisResult:
    """Result structure for terrain analysis."""
    safety_assessment: Dict[str, Any]
    science_targets: List[Dict[str, Any]]
    traversability: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float
    analysis_metadata: Dict[str, Any]


class GPT4VTerrainAnalyzer:
    """Advanced terrain analyzer using GPT-4V for Mars mission planning."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required for terrain analysis")
        
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4-vision-preview"
        self.max_tokens = 4096
        self.temperature = 0.1  # Low temperature for consistent analysis
        
        self.logger = logging.getLogger(__name__)
        
        # NASA operational constraints for analysis
        self.operational_limits = {
            "max_slope_degrees": 30.0,
            "preferred_slope_limit": 15.0,
            "max_rock_height_cm": 50.0,
            "wheel_diameter_cm": 52.5,
            "ground_clearance_cm": 35.0
        }
    
    def analyze_terrain(self, request: TerrainAnalysisRequest) -> TerrainAnalysisResult:
        """Perform comprehensive terrain analysis using GPT-4V.
        
        Args:
            request: Terrain analysis request with image and context
            
        Returns:
            Structured terrain analysis result
        """
        self.logger.info(f"Starting terrain analysis for Sol {request.sol}")
        
        try:
            # Encode image for API
            encoded_image = self._encode_image(request.image_path)
            
            # Generate analysis prompt based on request type
            prompt = self._generate_analysis_prompt(request)
            
            # Call GPT-4V API
            response = self._call_gpt4v_api(encoded_image, prompt)
            
            # Parse and structure the response
            analysis_result = self._parse_analysis_response(response, request)
            
            self.logger.info(f"Terrain analysis completed for Sol {request.sol}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Terrain analysis failed: {e}")
            return self._create_error_result(str(e), request)
    
    def analyze_rover_imagery(self, image_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze rover imagery with mission context (legacy interface)."""
        request = TerrainAnalysisRequest(
            image_path=image_path,
            sol=context.get('sol', 0),
            coordinates=(context.get('latitude', 0), context.get('longitude', 0)),
            mission_phase=context.get('mission_phase', 'unknown'),
            environmental_conditions=context.get('environmental', {}),
            analysis_type=context.get('analysis_type', 'comprehensive')
        )
        
        result = self.analyze_terrain(request)
        
        # Convert to legacy format
        return {
            'hazards_detected': result.safety_assessment.get('hazards', []),
            'science_targets': result.science_targets,
            'traversability_score': result.traversability.get('overall_score', 0.5),
            'recommendations': result.recommendations,
            'confidence': result.confidence_score
        }
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API transmission."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load and optionally resize image to manage API limits
            with Image.open(image_path) as img:
                # Resize if too large (API has size limits)
                if img.size[0] > 2048 or img.size[1] > 2048:
                    img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_bytes = buffer.getvalue()
            
            return base64.b64encode(image_bytes).decode('utf-8')
            
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {e}")
    
    def _generate_analysis_prompt(self, request: TerrainAnalysisRequest) -> str:
        """Generate comprehensive analysis prompt based on NASA procedures."""
        
        base_prompt = f"""You are a NASA Mars mission planner analyzing rover imagery for operational planning. 
Analyze this Mars terrain image with the expertise of a JPL mission operations specialist.

MISSION CONTEXT:
- Sol: {request.sol}
- Location: {request.coordinates[0]:.4f}°N, {request.coordinates[1]:.4f}°E  
- Mission Phase: {request.mission_phase}
- Environmental: {self._format_environmental_conditions(request.environmental_conditions)}

ANALYSIS REQUIREMENTS:
Provide a structured analysis following NASA operational standards:"""

        if request.analysis_type in ['comprehensive', 'hazard']:
            base_prompt += """

1. SAFETY ASSESSMENT:
   - Identify potential hazards: rocks, slopes, soft terrain, cliffs
   - Assess slope angles relative to 30° operational limit
   - Evaluate rock sizes relative to 50cm traversable height
   - Identify any prohibited zones for rover operations
   - Rate overall safety level: SAFE, CAUTION, HAZARD

2. TRAVERSABILITY ANALYSIS:
   - Assess terrain difficulty: easy, moderate, difficult
   - Identify preferred paths for rover movement  
   - Note any obstacles requiring navigation around
   - Estimate drive complexity and energy requirements"""

        if request.analysis_type in ['comprehensive', 'science']:
            base_prompt += """

3. SCIENTIFIC TARGET IDENTIFICATION:
   - Identify interesting geological features
   - Assess rock types and formations
   - Note potential sampling targets
   - Evaluate scientific value of visible features
   - Prioritize targets by accessibility and scientific interest"""

        if request.analysis_type in ['comprehensive', 'traverse']:
            base_prompt += """

4. ROUTE RECOMMENDATIONS:
   - Suggest optimal paths through the terrain
   - Identify waypoints for navigation
   - Recommend drive distances and approach angles
   - Note any required path modifications"""

        base_prompt += f"""

OPERATIONAL CONSTRAINTS (Perseverance rover):
- Maximum safe slope: {self.operational_limits['max_slope_degrees']}°
- Preferred slope limit: {self.operational_limits['preferred_slope_limit']}°
- Maximum rock height: {self.operational_limits['max_rock_height_cm']}cm
- Ground clearance: {self.operational_limits['ground_clearance_cm']}cm
- Wheel diameter: {self.operational_limits['wheel_diameter_cm']}cm

OUTPUT FORMAT:
Respond with a structured JSON analysis containing:
{{
  "safety_assessment": {{
    "overall_level": "SAFE|CAUTION|HAZARD",
    "hazards": [list of identified hazards],
    "slope_assessment": "assessment of visible slopes",
    "rock_assessment": "assessment of rocks and obstacles"
  }},
  "science_targets": [
    {{
      "target_id": 1,
      "type": "rock|formation|deposit",
      "location": "description of location in image",
      "priority": "high|medium|low",
      "accessibility": "easy|moderate|difficult",
      "scientific_value": "description"
    }}
  ],
  "traversability": {{
    "overall_score": 0.0-1.0,
    "difficulty": "easy|moderate|difficult",
    "preferred_path": "description of best route",
    "energy_estimate": "low|medium|high"
  }},
  "recommendations": [
    "specific operational recommendations"
  ],
  "confidence": 0.0-1.0
}}

Analyze the image with the precision and caution of a NASA mission operations team responsible for a $2.7B Mars rover."""

        return base_prompt
    
    def _format_environmental_conditions(self, conditions: Dict[str, Any]) -> str:
        """Format environmental conditions for prompt context."""
        if not conditions:
            return "No environmental data available"
        
        formatted = []
        if 'temperature_c' in conditions:
            formatted.append(f"Temperature: {conditions['temperature_c']}°C")
        if 'pressure_pa' in conditions:
            formatted.append(f"Pressure: {conditions['pressure_pa']} Pa")
        if 'wind_speed_ms' in conditions:
            formatted.append(f"Wind: {conditions['wind_speed_ms']} m/s")
        if 'operational_status' in conditions:
            formatted.append(f"Status: {conditions['operational_status']}")
        
        return ", ".join(formatted) if formatted else "Nominal conditions"
    
    def _call_gpt4v_api(self, encoded_image: str, prompt: str) -> Dict[str, Any]:
        """Call GPT-4V API with image and analysis prompt."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        self.logger.info(f"Calling GPT-4V API for terrain analysis")
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            api_response = response.json()
            
            if 'choices' not in api_response or not api_response['choices']:
                raise ValueError("Invalid API response: no choices returned")
            
            return api_response
            
        except requests.RequestException as e:
            self.logger.error(f"GPT-4V API request failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"GPT-4V API call failed: {e}")
            raise
    
    def _parse_analysis_response(self, api_response: Dict[str, Any], 
                                request: TerrainAnalysisRequest) -> TerrainAnalysisResult:
        """Parse GPT-4V response into structured result."""
        try:
            # Extract response content
            content = api_response['choices'][0]['message']['content']
            
            # Try to parse JSON response
            try:
                # Find JSON block in response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    analysis_data = json.loads(json_str)
                else:
                    # Fallback: try to parse entire content as JSON
                    analysis_data = json.loads(content)
                    
            except json.JSONDecodeError:
                # Fallback: parse unstructured response
                self.logger.warning("Failed to parse JSON response, using text parsing")
                analysis_data = self._parse_text_response(content)
            
            # Create structured result
            result = TerrainAnalysisResult(
                safety_assessment=analysis_data.get('safety_assessment', {}),
                science_targets=analysis_data.get('science_targets', []),
                traversability=analysis_data.get('traversability', {}),
                recommendations=analysis_data.get('recommendations', []),
                confidence_score=analysis_data.get('confidence', 0.5),
                analysis_metadata={
                    'sol': request.sol,
                    'coordinates': request.coordinates,
                    'analysis_type': request.analysis_type,
                    'timestamp': datetime.utcnow().isoformat(),
                    'model_used': self.model,
                    'api_response_tokens': api_response.get('usage', {}).get('total_tokens', 0)
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse analysis response: {e}")
            return self._create_error_result(f"Response parsing failed: {e}", request)
    
    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """Parse unstructured text response into analysis data."""
        # Basic text parsing for fallback
        analysis_data = {
            'safety_assessment': {
                'overall_level': 'CAUTION',  # Conservative default
                'hazards': [],
                'slope_assessment': 'Unable to assess from text response',
                'rock_assessment': 'Unable to assess from text response'
            },
            'science_targets': [],
            'traversability': {
                'overall_score': 0.5,
                'difficulty': 'moderate',
                'preferred_path': 'Requires detailed analysis',
                'energy_estimate': 'medium'
            },
            'recommendations': ['Detailed visual analysis recommended'],
            'confidence': 0.3  # Low confidence for text-only parsing
        }
        
        # Extract key terms and assessments
        content_lower = content.lower()
        
        # Safety assessment
        if any(word in content_lower for word in ['safe', 'clear', 'smooth']):
            analysis_data['safety_assessment']['overall_level'] = 'SAFE'
        elif any(word in content_lower for word in ['danger', 'hazard', 'steep', 'cliff']):
            analysis_data['safety_assessment']['overall_level'] = 'HAZARD'
        
        # Extract hazards mentioned
        hazard_keywords = ['rock', 'boulder', 'slope', 'cliff', 'crater', 'sand', 'dust']
        for keyword in hazard_keywords:
            if keyword in content_lower:
                analysis_data['safety_assessment']['hazards'].append(f"Potential {keyword} hazard mentioned")
        
        return analysis_data
    
    def _create_error_result(self, error_msg: str, request: TerrainAnalysisRequest) -> TerrainAnalysisResult:
        """Create error result when analysis fails."""
        return TerrainAnalysisResult(
            safety_assessment={
                'overall_level': 'UNKNOWN',
                'hazards': [],
                'slope_assessment': f'Analysis failed: {error_msg}',
                'rock_assessment': 'Unable to assess due to error'
            },
            science_targets=[],
            traversability={
                'overall_score': 0.0,
                'difficulty': 'unknown',
                'preferred_path': 'Manual analysis required',
                'energy_estimate': 'unknown'
            },
            recommendations=[f'Terrain analysis failed: {error_msg}', 'Manual review recommended'],
            confidence_score=0.0,
            analysis_metadata={
                'sol': request.sol,
                'coordinates': request.coordinates,
                'analysis_type': request.analysis_type,
                'timestamp': datetime.utcnow().isoformat(),
                'error': error_msg
            }
        )
    
    def batch_analyze_images(self, image_paths: List[str], 
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze multiple images in batch for efficiency."""
        results = []
        
        self.logger.info(f"Starting batch analysis of {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths):
            try:
                self.logger.info(f"Analyzing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                # Create individual request
                request = TerrainAnalysisRequest(
                    image_path=image_path,
                    sol=context.get('sol', 0),
                    coordinates=(context.get('latitude', 0), context.get('longitude', 0)),
                    mission_phase=context.get('mission_phase', 'unknown'),
                    environmental_conditions=context.get('environmental', {}),
                    analysis_type=context.get('analysis_type', 'comprehensive')
                )
                
                result = self.analyze_terrain(request)
                
                # Convert to dictionary format
                result_dict = {
                    'image_path': image_path,
                    'safety_level': result.safety_assessment.get('overall_level', 'UNKNOWN'),
                    'hazards': result.safety_assessment.get('hazards', []),
                    'science_targets': result.science_targets,
                    'traversability_score': result.traversability.get('overall_score', 0.5),
                    'recommendations': result.recommendations,
                    'confidence': result.confidence_score
                }
                
                results.append(result_dict)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze image {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'safety_level': 'UNKNOWN',
                    'confidence': 0.0
                })
        
        self.logger.info(f"Batch analysis completed: {len(results)} results")
        return results