#!/usr/bin/env python3
"""
NASA GPT-4 Mission Planning Engine

Implements ASPEN-inspired mission planning using GPT-4 for Mars rover operations:
- Resource-constrained activity scheduling
- NASA operational procedures and constraints
- Power, thermal, and communication timeline management
- Science prioritization with safety considerations
- Command sequence generation compatible with rover systems

Based on NASA JPL operational standards and Perseverance mission parameters.
"""

import os
import requests
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class MissionConstraints:
    """NASA operational constraints for mission planning."""
    # Power constraints (Perseverance MMRTG + batteries)
    mmrtg_output_w: float = 110.0          # Declining ~3% per year
    battery_capacity_wh: float = 43.0 * 2  # Two 43 Wh lithium-ion batteries
    peak_power_demand_w: float = 900.0     # Maximum instantaneous power
    survival_heating_w: float = 150.0      # Overnight heating requirement
    
    # Thermal constraints
    operating_temp_min_c: float = -40.0    # Minimum operating temperature
    operating_temp_max_c: float = 40.0     # Maximum operating temperature
    survival_temp_min_c: float = -73.0     # Absolute survival minimum
    
    # Communication constraints
    earth_mars_delay_min: float = 20.0     # Round-trip communication delay
    comm_windows_per_sol: int = 2          # Average communication opportunities
    data_transmission_limit_mb: float = 250.0  # Daily data transmission limit
    
    # Operational constraints
    sol_length_hours: float = 24.65        # Mars solar day length
    max_activities_per_sol: int = 100      # Planning complexity limit
    safety_margin_factor: float = 1.2     # Safety factor for all calculations


class NASAMissionPlanner:
    """NASA-standard mission planner using GPT-4 for Mars rover operations."""
    
    def __init__(self, api_key: str = None, constraints: MissionConstraints = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required for NASA mission planning")
        
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4-1106-preview"  # Latest GPT-4 for complex reasoning
        self.max_tokens = 4096
        self.temperature = 0.1  # Low temperature for mission-critical planning
        
        self.constraints = constraints or MissionConstraints()
        self.logger = logging.getLogger(__name__)
        
    def generate_mission_plan(self, goals: List[str], constraints: Dict[str, Any], 
                            rover_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive mission plan using NASA operational standards.
        
        Args:
            goals: List of mission objectives
            constraints: Operational constraints (power, time, environmental)
            rover_state: Current rover status and position
            
        Returns:
            Structured mission plan with activity sequences and resource allocation
        """
        try:
            self.logger.info(f"Generating NASA mission plan with {len(goals)} objectives")
            
            # Prepare mission planning context
            planning_context = self._prepare_planning_context(goals, constraints, rover_state)
            
            # Generate NASA mission planning prompt
            prompt = self._generate_mission_planning_prompt(planning_context)
            
            # Call GPT-4 for mission planning
            api_response = self._call_gpt4_api(prompt)
            
            # Parse and validate mission plan
            mission_plan = self._parse_mission_plan(api_response, planning_context)
            
            # Apply NASA safety and resource validation
            validated_plan = self._validate_mission_plan(mission_plan, planning_context)
            
            self.logger.info(f"NASA mission plan generated successfully")
            return validated_plan
            
        except Exception as e:
            self.logger.error(f"Mission planning failed: {e}")
            return self._create_error_plan(str(e), goals)
    
    def _prepare_planning_context(self, goals: List[str], constraints: Dict[str, Any], 
                                rover_state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive context for mission planning."""
        
        # Extract rover position
        location = rover_state.get('location', {})
        latitude = location.get('lat', 0.0)
        longitude = location.get('lon', 0.0)
        
        # Calculate current sol from constraints or estimate
        current_sol = constraints.get('current_sol', self._estimate_current_sol())
        
        # Prepare environmental data
        environmental = constraints.get('environmental', {})
        temperature = environmental.get('temperature_c', -60)  # Typical Mars temp
        pressure = environmental.get('pressure_pa', 800)      # Typical Mars pressure
        wind_speed = environmental.get('wind_speed_ms', 5)    # Moderate wind
        
        # Power budget calculation
        power_budget = constraints.get('power_budget_wh', self._calculate_daily_power_budget())
        time_budget = constraints.get('time_budget_min', 8 * 60)  # 8 hours operational
        
        return {
            'objectives': goals,
            'current_sol': current_sol,
            'rover_location': {'latitude': latitude, 'longitude': longitude},
            'environmental_conditions': {
                'temperature_c': temperature,
                'pressure_pa': pressure,
                'wind_speed_ms': wind_speed,
                'operational_status': self._assess_environmental_status(temperature, wind_speed)
            },
            'resource_constraints': {
                'power_budget_wh': power_budget,
                'time_budget_min': time_budget,
                'data_budget_mb': constraints.get('data_budget_mb', self.constraints.data_transmission_limit_mb)
            },
            'rover_capabilities': {
                'instruments': ['MASTCAM-Z', 'SUPERCAM', 'PIXL', 'SHERLOC', 'MOXIE', 'RIMFAX', 'MEDA'],
                'mobility': 'full',  # Assume full mobility unless specified
                'sampling': 'operational',
                'max_drive_distance_m': constraints.get('max_drive_m', 200)
            },
            'operational_constraints': {
                'max_slope_degrees': 30.0,
                'comm_windows': 2,
                'earth_mars_delay_min': 20.0,
                'autonomous_operations': True
            }
        }
    
    def _generate_mission_planning_prompt(self, context: Dict[str, Any]) -> str:
        """Generate comprehensive NASA mission planning prompt."""
        
        objectives_str = '\n'.join([f"- {obj}" for obj in context['objectives']])
        
        return f"""You are a senior NASA JPL Mars mission operations planner creating a Sol {context['current_sol']} activity plan for the Perseverance rover. Your plan will be executed autonomously with a 20-minute Earth-Mars communication delay.

MISSION CONTEXT:
Sol: {context['current_sol']}
Location: {context['rover_location']['latitude']:.6f}°N, {context['rover_location']['longitude']:.6f}°E
Environmental: {context['environmental_conditions']['temperature_c']}°C, {context['environmental_conditions']['pressure_pa']} Pa, {context['environmental_conditions']['wind_speed_ms']} m/s
Status: {context['environmental_conditions']['operational_status']}

MISSION OBJECTIVES:
{objectives_str}

RESOURCE CONSTRAINTS:
- Power Budget: {context['resource_constraints']['power_budget_wh']:.1f} Wh available
- Time Budget: {context['resource_constraints']['time_budget_min']:.0f} minutes operational
- Data Budget: {context['resource_constraints']['data_budget_mb']:.1f} MB transmission limit
- Max Drive Distance: {context['rover_capabilities']['max_drive_distance_m']} meters

PLANNING REQUIREMENTS:
Create a detailed Sol activity plan following NASA operational procedures:

1. ACTIVITY SEQUENCING:
   - Pre-operational system checks and warmup
   - Science activities prioritized by objectives
   - Mobility operations with safety margins
   - Data collection and transmission
   - Post-operational system safing

2. RESOURCE ALLOCATION:
   - Power budget distribution across activities
   - Timeline with activity start/stop times
   - Data volume estimates for each activity
   - Contingency reserves (20% power, 15% time)

3. RISK MANAGEMENT:
   - Identify single points of failure
   - Autonomous operation safeguards  
   - Environmental condition monitoring
   - Backup plans for critical activities

4. OPERATIONAL PROCEDURES:
   - Pre-activity health checks
   - Instrument calibration sequences
   - Drive execution with waypoints
   - Post-activity data validation

PERSEVERANCE OPERATIONAL PARAMETERS:
- MMRTG Power: {self.constraints.mmrtg_output_w}W continuous
- Battery Capacity: {self.constraints.battery_capacity_wh}Wh total
- Peak Power Limit: {self.constraints.peak_power_demand_w}W
- Operating Temperature: {self.constraints.operating_temp_min_c}°C to {self.constraints.operating_temp_max_c}°C
- Communication Windows: {self.constraints.comm_windows_per_sol} per sol
- Sol Length: {self.constraints.sol_length_hours} hours

REQUIRED OUTPUT FORMAT:
Generate structured mission plan as JSON:

{{
  "mission_plan": {{
    "sol": {context['current_sol']},
    "plan_id": "P{context['current_sol']:04d}",
    "planning_timestamp": "ISO timestamp",
    "objectives_addressed": ["list of objectives addressed"],
    "risk_level": "LOW|MEDIUM|HIGH",
    "execution_strategy": "AUTONOMOUS|SUPERVISED|MANUAL"
  }},
  "activity_sequence": [
    {{
      "activity_id": "A001",
      "type": "SYSTEM_CHECK|SCIENCE|MOBILITY|COMMUNICATION|MAINTENANCE",
      "name": "Descriptive activity name",
      "start_time_min": 0,
      "duration_min": 30,
      "priority": "CRITICAL|HIGH|MEDIUM|LOW",
      "power_requirement_wh": 25.5,
      "data_volume_mb": 10.2,
      "description": "Detailed activity description",
      "parameters": {{
        "instrument": "MASTCAM-Z|SUPERCAM|PIXL|etc",
        "target": "specific target name/location",
        "configuration": "instrument-specific settings"
      }},
      "success_criteria": "how to validate completion",
      "contingency": "backup plan if activity fails",
      "dependencies": ["prerequisite activity IDs"]
    }}
  ],
  "resource_allocation": {{
    "total_power_required_wh": 150.0,
    "power_margin_wh": 30.0,
    "total_time_required_min": 420,
    "time_margin_min": 60,
    "total_data_volume_mb": 180.0,
    "data_margin_mb": 20.0
  }},
  "risk_assessment": {{
    "primary_risks": [
      {{
        "risk": "description of risk",
        "probability": "LOW|MEDIUM|HIGH", 
        "impact": "LOW|MEDIUM|HIGH|CRITICAL",
        "mitigation": "how risk is mitigated"
      }}
    ],
    "contingency_plans": [
      "backup plans for critical failures"
    ],
    "abort_criteria": [
      "conditions that would halt execution"
    ]
  }},
  "communication_plan": {{
    "uplink_windows": [
      {{
        "window_id": 1,
        "start_time_min": 60,
        "duration_min": 30,
        "data_volume_mb": 5.0,
        "content": "telemetry, images, science data"
      }}
    ],
    "downlink_requirements": {{
      "critical_data_mb": 20.0,
      "science_data_mb": 150.0,
      "housekeeping_data_mb": 10.0
    }}
  }},
  "validation_metrics": {{
    "power_efficiency": 0.0-1.0,
    "timeline_feasibility": 0.0-1.0, 
    "science_value": 0.0-1.0,
    "operational_risk": 0.0-1.0,
    "overall_plan_quality": 0.0-1.0
  }},
  "operational_notes": [
    "Important notes for mission operations team",
    "Special considerations and constraints",
    "Lessons learned from previous sols"
  ]
}}

PLANNING STANDARDS:
- Apply NASA operational safety margins and conservative resource estimates
- Prioritize rover preservation over aggressive science objectives
- Plan for autonomous execution with minimal Earth intervention
- Include detailed contingency procedures for all critical activities
- Ensure plan can be modified during execution based on real-time conditions
- Follow JPL mission operations terminology and procedures

Create a mission plan that balances ambitious science goals with the operational reality of Mars rover operations, ensuring mission success and rover longevity."""

    def _call_gpt4_api(self, prompt: str) -> Dict[str, Any]:
        """Execute GPT-4 API call for mission planning."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a NASA JPL Mars mission operations planner with expertise in rover systems, resource management, and autonomous operations. Generate precise, executable mission plans."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }
        
        self.logger.info("Calling GPT-4 API for NASA mission planning")
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=180  # Extended timeout for complex planning
            )
            response.raise_for_status()
            
            api_response = response.json()
            
            # Log usage statistics
            usage = api_response.get('usage', {})
            self.logger.info(f"Mission planning API usage: {usage.get('total_tokens', 0)} tokens")
            
            return api_response
            
        except requests.RequestException as e:
            self.logger.error(f"GPT-4 mission planning API request failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"GPT-4 mission planning API call failed: {e}")
            raise
    
    def _parse_mission_plan(self, api_response: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse GPT-4 response into structured mission plan."""
        try:
            content = api_response['choices'][0]['message']['content']
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                mission_plan = json.loads(json_str)
                
                # Add parsing metadata
                mission_plan['parsing_metadata'] = {
                    'parsed_at': datetime.utcnow().isoformat(),
                    'model_version': self.model,
                    'api_tokens': api_response.get('usage', {}).get('total_tokens', 0),
                    'planning_context': context
                }
                
                return mission_plan
                
            else:
                raise json.JSONDecodeError("No valid JSON found in mission planning response", content, 0)
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse structured mission plan: {e}")
            # Create fallback plan from text content
            return self._create_fallback_plan(content, context)
    
    def _validate_mission_plan(self, mission_plan: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mission plan against NASA operational constraints."""
        
        validation_results = {
            'validation_passed': True,
            'validation_warnings': [],
            'validation_errors': [],
            'modifications_applied': []
        }
        
        # Validate resource constraints
        resource_validation = self._validate_resources(mission_plan, context)
        validation_results.update(resource_validation)
        
        # Validate timeline feasibility
        timeline_validation = self._validate_timeline(mission_plan)
        validation_results['validation_warnings'].extend(timeline_validation.get('warnings', []))
        validation_results['validation_errors'].extend(timeline_validation.get('errors', []))
        
        # Apply safety margins
        safety_validation = self._apply_safety_margins(mission_plan)
        validation_results['modifications_applied'].extend(safety_validation.get('modifications', []))
        
        # Add validation metadata
        mission_plan['validation_metadata'] = {
            'validated_at': datetime.utcnow().isoformat(),
            'validator_version': 'NASA_MPV_1.0',
            'constraints_applied': self.constraints.__dict__,
            'validation_summary': validation_results
        }
        
        if validation_results['validation_errors']:
            mission_plan['plan_status'] = 'VALIDATION_FAILED'
            mission_plan['execution_approval'] = 'DENIED'
        elif validation_results['validation_warnings']:
            mission_plan['plan_status'] = 'CONDITIONAL_APPROVAL'
            mission_plan['execution_approval'] = 'REVIEW_REQUIRED'
        else:
            mission_plan['plan_status'] = 'VALIDATED'
            mission_plan['execution_approval'] = 'APPROVED'
        
        return mission_plan
    
    def _validate_resources(self, mission_plan: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resource allocation against constraints."""
        validation = {'validation_warnings': [], 'validation_errors': []}
        
        # Power validation
        total_power = mission_plan.get('resource_allocation', {}).get('total_power_required_wh', 0)
        available_power = context['resource_constraints']['power_budget_wh']
        
        if total_power > available_power:
            validation['validation_errors'].append(
                f"Power requirement ({total_power:.1f} Wh) exceeds budget ({available_power:.1f} Wh)"
            )
        elif total_power > available_power * 0.9:
            validation['validation_warnings'].append(
                f"Power usage ({total_power:.1f} Wh) approaching limit ({available_power:.1f} Wh)"
            )
        
        # Time validation
        total_time = mission_plan.get('resource_allocation', {}).get('total_time_required_min', 0)
        available_time = context['resource_constraints']['time_budget_min']
        
        if total_time > available_time:
            validation['validation_errors'].append(
                f"Time requirement ({total_time:.0f} min) exceeds budget ({available_time:.0f} min)"
            )
        
        # Data validation
        total_data = mission_plan.get('resource_allocation', {}).get('total_data_volume_mb', 0)
        available_data = context['resource_constraints']['data_budget_mb']
        
        if total_data > available_data:
            validation['validation_warnings'].append(
                f"Data volume ({total_data:.1f} MB) exceeds transmission capacity ({available_data:.1f} MB)"
            )
        
        return validation
    
    def _validate_timeline(self, mission_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate activity timeline for conflicts and feasibility."""
        validation = {'warnings': [], 'errors': []}
        
        activities = mission_plan.get('activity_sequence', [])
        if not activities:
            validation['errors'].append("No activities defined in mission plan")
            return validation
        
        # Check for timeline conflicts
        sorted_activities = sorted(activities, key=lambda x: x.get('start_time_min', 0))
        
        for i in range(len(sorted_activities) - 1):
            current = sorted_activities[i]
            next_activity = sorted_activities[i + 1]
            
            current_end = current.get('start_time_min', 0) + current.get('duration_min', 0)
            next_start = next_activity.get('start_time_min', 0)
            
            if current_end > next_start:
                validation['errors'].append(
                    f"Timeline conflict: Activity {current.get('activity_id')} overlaps with {next_activity.get('activity_id')}"
                )
        
        # Check for reasonable activity durations
        for activity in activities:
            duration = activity.get('duration_min', 0)
            if duration > 240:  # 4 hours
                validation['warnings'].append(
                    f"Activity {activity.get('activity_id')} duration ({duration} min) unusually long"
                )
            elif duration < 5:
                validation['warnings'].append(
                    f"Activity {activity.get('activity_id')} duration ({duration} min) unusually short"
                )
        
        return validation
    
    def _apply_safety_margins(self, mission_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply NASA safety margins to mission plan."""
        modifications = []
        
        # Apply power safety margin
        resource_alloc = mission_plan.get('resource_allocation', {})
        if 'total_power_required_wh' in resource_alloc:
            original_power = resource_alloc['total_power_required_wh']
            safety_power = original_power * self.constraints.safety_margin_factor
            resource_alloc['total_power_required_wh'] = safety_power
            modifications.append(f"Applied {self.constraints.safety_margin_factor}x power safety margin")
        
        # Ensure minimum margins are preserved
        if 'power_margin_wh' in resource_alloc:
            if resource_alloc['power_margin_wh'] < 20:
                resource_alloc['power_margin_wh'] = 20
                modifications.append("Increased power margin to 20 Wh minimum")
        
        return {'modifications': modifications}
    
    def _calculate_daily_power_budget(self) -> float:
        """Calculate available power budget for Sol planning."""
        # MMRTG continuous power over 24.65 hours
        daily_mmrtg = self.constraints.mmrtg_output_w * self.constraints.sol_length_hours
        
        # Subtract survival heating and system overhead
        survival_power = self.constraints.survival_heating_w * 12  # 12 hours heating
        system_overhead = daily_mmrtg * 0.2  # 20% system overhead
        
        available_power = daily_mmrtg - survival_power - system_overhead
        return max(0, available_power)
    
    def _assess_environmental_status(self, temperature: float, wind_speed: float) -> str:
        """Assess environmental conditions for operations."""
        if temperature < self.constraints.operating_temp_min_c:
            return 'COLD_LIMIT'
        elif temperature > self.constraints.operating_temp_max_c:
            return 'HOT_LIMIT'
        elif wind_speed > 20:
            return 'HIGH_WIND'
        else:
            return 'NOMINAL'
    
    def _estimate_current_sol(self) -> int:
        """Estimate current sol based on mission elapsed time."""
        # Mars 2020 landing: February 18, 2021
        from datetime import date
        landing_date = date(2021, 2, 18)
        current_date = date.today()
        elapsed_days = (current_date - landing_date).days
        # Convert to Mars sols (1 sol = 1.027 Earth days)
        current_sol = int(elapsed_days / 1.027)
        return max(0, current_sol)
    
    def _create_fallback_plan(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback mission plan when structured parsing fails."""
        self.logger.warning("Creating fallback mission plan due to parsing failure")
        
        return {
            "mission_plan": {
                "sol": context['current_sol'],
                "plan_id": f"FALLBACK_{context['current_sol']:04d}",
                "planning_timestamp": datetime.utcnow().isoformat(),
                "objectives_addressed": context['objectives'][:3],  # Limit to first 3
                "risk_level": "HIGH",  # Conservative due to parsing failure
                "execution_strategy": "MANUAL"
            },
            "activity_sequence": [
                {
                    "activity_id": "A001",
                    "type": "SYSTEM_CHECK",
                    "name": "System health verification",
                    "start_time_min": 0,
                    "duration_min": 30,
                    "priority": "CRITICAL",
                    "power_requirement_wh": 15.0,
                    "data_volume_mb": 2.0,
                    "description": "Verify all systems operational before proceeding",
                    "parameters": {},
                    "success_criteria": "All subsystems report nominal status",
                    "contingency": "Contact Earth if anomalies detected",
                    "dependencies": []
                }
            ],
            "resource_allocation": {
                "total_power_required_wh": 50.0,
                "power_margin_wh": 50.0,
                "total_time_required_min": 120,
                "time_margin_min": 60,
                "total_data_volume_mb": 10.0,
                "data_margin_mb": 40.0
            },
            "risk_assessment": {
                "primary_risks": [
                    {
                        "risk": "Plan parsing failure - manual review required",
                        "probability": "HIGH",
                        "impact": "MEDIUM",
                        "mitigation": "Manual mission planning oversight"
                    }
                ],
                "contingency_plans": [
                    "Revert to conservative automated operations",
                    "Request Earth-based mission planning support"
                ],
                "abort_criteria": [
                    "Any system anomaly",
                    "Environmental conditions outside limits"
                ]
            },
            "operational_notes": [
                "FALLBACK PLAN: Structured planning failed - manual review required",
                "Conservative operations until full plan validation",
                "Limited activities to minimize risk"
            ],
            "plan_status": "FALLBACK_MODE",
            "execution_approval": "REVIEW_REQUIRED",
            "fallback_reason": "GPT-4 response parsing failure",
            "raw_planning_response": content
        }
    
    def _create_error_plan(self, error_msg: str, goals: List[str]) -> Dict[str, Any]:
        """Create error response when mission planning completely fails."""
        return {
            "mission_plan": {
                "sol": 0,
                "plan_id": "ERROR_PLAN",
                "planning_timestamp": datetime.utcnow().isoformat(),
                "objectives_addressed": [],
                "risk_level": "CRITICAL",
                "execution_strategy": "ABORT"
            },
            "activity_sequence": [],
            "resource_allocation": {
                "total_power_required_wh": 0.0,
                "power_margin_wh": 0.0,
                "total_time_required_min": 0,
                "time_margin_min": 0,
                "total_data_volume_mb": 0.0,
                "data_margin_mb": 0.0
            },
            "error": error_msg,
            "plan_status": "PLANNING_FAILED",
            "execution_approval": "DENIED",
            "operational_notes": [
                f"Mission planning failed: {error_msg}",
                "Manual intervention required",
                "Do not execute autonomous operations"
            ],
            "requires_manual_planning": True
        }