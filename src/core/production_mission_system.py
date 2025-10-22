#!/usr/bin/env python3
"""
Production NASA Mars Mission Planning System

Integrates all components into a unified production-ready system that
matches the full problem statement requirements.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from pathlib import Path

from .gpt4o_vision_analyzer import GPT4oVisionAnalyzer
from .multi_sol_planner import MultiSolPlanner, MultiSolMissionPlan
from .nl_query_interface import NaturalLanguageQueryInterface, ConversationalInterface
from .msr_sample_caching import MSRSampleCachingSystem, MSRMission, Sample
from .audit_logging import get_audit_logger, EventType, Severity
from .dem_auto_downloader import DEMAutoDownloader

from ..data_pipeline.nasa_api_client import NASAAPIClient
from ..data_pipeline.dem_processor import DEMProcessor


class ProductionMissionSystem:
    """
    Production NASA Mars Mission Planning System
    
    Implements full problem statement:
    - GPT-4o vision analysis of rover imagery
    - Multi-sol mission planning (2-3+ sols)
    - Natural language query interface
    - MSR sample caching and retrieval
    - NASA-grade audit logging
    - Real-time DEM downloading
    - Resource optimization (MMRTG, battery, thermal)
    - Contingency planning
    """
    
    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 nasa_api_key: Optional[str] = None,
                 cache_directory: str = "data/cache",
                 log_directory: str = "logs/audit"):
        """
        Initialize production mission system.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4o
            nasa_api_key: NASA API key for data access
            cache_directory: Cache directory for DEMs and data
            log_directory: Directory for audit logs
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Production NASA Mars Mission Planning System")
        
        # Initialize core components
        self.vision_analyzer = GPT4oVisionAnalyzer(api_key=openai_api_key)
        self.multi_sol_planner = MultiSolPlanner()
        self.nl_interface = NaturalLanguageQueryInterface(api_key=openai_api_key)
        self.conversational = ConversationalInterface(self.nl_interface)
        self.msr_system = MSRSampleCachingSystem()
        self.audit_logger = get_audit_logger(log_directory)
        self.dem_downloader = DEMAutoDownloader(f"{cache_directory}/dem")
        
        # Initialize data pipeline
        self.nasa_client = NASAAPIClient(api_key=nasa_api_key)
        self.dem_processor = DEMProcessor()
        
        # System state
        self.current_mission: Optional[MSRMission] = None
        
        self.logger.info("Production system initialized successfully")
    
    def process_natural_language_request(self,
                                        query: str,
                                        execute: bool = False) -> Dict[str, Any]:
        """
        Process natural language mission request.
        
        Examples:
        - "Plan a 2-sol traverse focusing on delta sample logistics"
        - "Analyze these Mastcam-Z images for hazards"
        - "Update route for dust storm forecast"
        
        Args:
            query: Natural language mission request
            execute: Whether to automatically execute the plan
            
        Returns:
            Structured plan and execution results
        """
        self.logger.info(f"Processing NL request: {query[:100]}...")
        
        # Log query
        event = self.audit_logger.log_event(
            event_type=EventType.PLAN_CREATED,
            action="Process natural language query",
            resource="mission_plan",
            input_data={'query': query}
        )
        
        # Get rover state context
        context = self._get_mission_context()
        
        # Process query
        plan = self.nl_interface.process_query(query, context)
        
        result = {
            'query': query,
            'plan': plan,
            'context': context,
            'executed': False
        }
        
        if execute:
            # Execute plan
            execution_results = self.nl_interface.execute_plan(plan, self)
            result['execution_results'] = execution_results
            result['executed'] = True
            
            # Log execution
            self.audit_logger.log_event(
                event_type=EventType.PLAN_EXECUTED,
                action="Executed natural language plan",
                resource="mission_plan",
                resource_id=plan.get('plan_id'),
                parent_event_id=event.event_id,
                output_data={'success': execution_results.get('success')}
            )
        
        return result
    
    def plan_multi_sol_mission(self,
                               start_location: tuple,
                               science_targets: List[Dict[str, Any]],
                               num_sols: int,
                               mission_objectives: Dict[str, Any],
                               mission_name: Optional[str] = None) -> MultiSolMissionPlan:
        """
        Plan comprehensive multi-sol mission.
        
        Args:
            start_location: Starting (lat, lon)
            science_targets: List of science targets with priorities
            num_sols: Number of sols to plan (typically 2-3)
            mission_objectives: Science objectives
            mission_name: Optional mission name
            
        Returns:
            Complete multi-sol mission plan
        """
        self.logger.info(f"Planning {num_sols}-sol mission from {start_location}")
        
        # Download DEMs for mission area
        self._ensure_dems_for_mission(start_location, science_targets)
        
        # Get environmental forecast
        env_forecast = self._get_environmental_forecast(num_sols)
        
        # Create mission plan
        mission_plan = self.multi_sol_planner.plan_multi_sol_mission(
            start_location=start_location,
            target_locations=science_targets,
            num_sols=num_sols,
            mission_objectives=mission_objectives,
            environmental_forecast=env_forecast,
            constraints={'mission_name': mission_name or f"Mission_{datetime.utcnow().strftime('%Y%m%d')}", 'start_sol': 0}
        )
        
        # Validate plan
        validation = self._validate_mission_plan(mission_plan)
        
        # Log plan creation
        self.audit_logger.log_event(
            event_type=EventType.PLAN_CREATED,
            action=f"Created {num_sols}-sol mission plan",
            resource="mission_plan",
            resource_id=mission_plan.mission_name,
            output_data={
                'num_sols': num_sols,
                'total_distance_m': mission_plan.total_distance_m,
                'samples_planned': mission_plan.total_samples_collected,
                'validation_passed': validation['passed']
            },
            approval_required=True
        )
        
        return mission_plan
    
    def analyze_terrain_imagery(self,
                                image_paths: List[str],
                                camera_type: str,
                                sol: int,
                                location: tuple,
                                environmental_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Mars terrain imagery using GPT-4o Vision.
        
        Args:
            image_paths: Paths to rover images
            camera_type: MASTCAM_Z, HAZCAM, or NAVCAM
            sol: Mars sol number
            location: (lat, lon)
            environmental_context: Temperature, dust, wind data
            
        Returns:
            Comprehensive terrain analysis
        """
        self.logger.info(f"Analyzing {len(image_paths)} {camera_type} images from Sol {sol}")
        
        analyses = []
        
        for image_path in image_paths:
            analysis = self.vision_analyzer.analyze_terrain_image(
                image_path=image_path,
                camera_type=camera_type,
                sol=sol,
                coordinates=location,
                environmental_context=environmental_context
            )
            
            analyses.append(analysis)
            
            # Log hazards if detected
            if analysis.get('hazard_assessment', {}).get('overall_safety') == 'HAZARD':
                self.audit_logger.log_event(
                    event_type=EventType.HAZARD_DETECTED,
                    action=f"Hazard detected in {camera_type} imagery",
                    resource="terrain_analysis",
                    sol=sol,
                    location=location,
                    severity=Severity.WARNING,
                    output_data=analysis['hazard_assessment']
                )
        
        # Generate summary
        if len(analyses) > 1:
            batch_result = self.vision_analyzer.batch_analyze_images(
                [{'path': p, 'camera_type': camera_type, 'sol': sol, 'coordinates': location} 
                 for p in image_paths],
                {'sol': sol, 'environment': environmental_context}
            )
            
            return batch_result
        else:
            return analyses[0] if analyses else {}
    
    def plan_sample_caching_strategy(self,
                                    mission_plan: MultiSolMissionPlan,
                                    science_targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Plan MSR sample caching strategy.
        
        Args:
            mission_plan: Multi-sol mission plan
            science_targets: Science targets for sampling
            
        Returns:
            Complete sample caching strategy
        """
        self.logger.info("Planning MSR sample caching strategy")
        
        # Initialize or get current mission
        if not self.current_mission:
            self.current_mission = MSRMission(
                mission_id=mission_plan.mission_name
            )
        
        # Plan caching strategy
        caching_strategy = self.msr_system.plan_sample_caching_strategy(
            mission=self.current_mission,
            traverse_plan={'sol_plans': [vars(sp) for sp in mission_plan.sol_plans]},
            science_targets=science_targets
        )
        
        # Log caching plan
        self.audit_logger.log_event(
            event_type=EventType.PLAN_CREATED,
            action="Created sample caching strategy",
            resource="sample_cache_plan",
            resource_id=mission_plan.mission_name,
            output_data={
                'total_caches': caching_strategy['total_caches_planned'],
                'total_samples': caching_strategy['total_samples_planned']
            }
        )
        
        return caching_strategy
    
    def execute_contingency_plan(self,
                                hazard_type: str,
                                current_plan: MultiSolMissionPlan,
                                current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute contingency plan for emergent hazard.
        
        Args:
            hazard_type: DUST_STORM, LOW_POWER, EQUIPMENT_FAILURE
            current_plan: Active mission plan
            current_state: Current rover state
            
        Returns:
            Updated contingency plan
        """
        self.logger.warning(f"Executing contingency plan for {hazard_type}")
        
        # Log contingency trigger
        event = self.audit_logger.log_event(
            event_type=EventType.CONTINGENCY_TRIGGERED,
            action=f"Contingency triggered: {hazard_type}",
            resource="contingency_plan",
            resource_id=current_plan.mission_name,
            severity=Severity.WARNING,
            input_data={'hazard_type': hazard_type, 'current_state': current_state}
        )
        
        # Find matching contingency
        matching = [c for c in current_plan.contingency_plans 
                   if c.get('trigger') == hazard_type]
        
        if not matching:
            # Generate new contingency
            contingency = self._generate_emergency_contingency(
                hazard_type, current_plan, current_state
            )
        else:
            contingency = matching[0]
        
        # Execute contingency actions
        result = {
            'hazard_type': hazard_type,
            'contingency': contingency,
            'actions_taken': [],
            'rover_safe': True
        }
        
        # Simulate contingency execution
        action = contingency.get('action', 'Safe rover and wait')
        result['actions_taken'].append(action)
        
        # Log contingency completion
        self.audit_logger.log_event(
            event_type=EventType.EMERGENCY_ACTION,
            action=f"Contingency executed: {action}",
            resource="contingency_plan",
            resource_id=current_plan.mission_name,
            parent_event_id=event.event_id,
            severity=Severity.WARNING,
            output_data=result
        )
        
        return result
    
    def generate_mission_report(self,
                               mission_plan: MultiSolMissionPlan,
                               include_traceability: bool = True) -> str:
        """
        Generate NASA-grade mission report.
        
        Args:
            mission_plan: Mission plan to report on
            include_traceability: Include audit trail
            
        Returns:
            Human-readable mission report
        """
        report = []
        
        report.append("=" * 80)
        report.append(f"NASA MARS MISSION PLANNING REPORT")
        report.append(f"Mission: {mission_plan.mission_name}")
        report.append(f"Generated: {datetime.utcnow().isoformat()}Z")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(f"Mission Duration: Sol {mission_plan.start_sol} to {mission_plan.end_sol} ({mission_plan.num_sols} sols)")
        report.append(f"Total Traverse Distance: {mission_plan.total_distance_m:.1f} m")
        report.append(f"Science Targets: {mission_plan.high_priority_targets_visited} high-priority")
        report.append(f"Samples Planned: {mission_plan.total_samples_collected}")
        report.append(f"Power Budget: {mission_plan.total_power_consumed_wh:.1f} Wh")
        report.append(f"Overall Risk: {mission_plan.overall_risk_level}")
        report.append("")
        
        # Per-Sol Breakdown
        report.append("SOL-BY-SOL ACTIVITIES")
        report.append("-" * 80)
        
        for sol_plan in mission_plan.sol_plans:
            report.append(f"\nSOL {sol_plan.sol_number}")
            report.append(f"  Location: {sol_plan.start_location} → {sol_plan.end_location}")
            report.append(f"  Drive: {sol_plan.drive_distance_m:.1f} m ({sol_plan.drive_time_hours:.1f} hrs)")
            report.append(f"  Power: {sol_plan.power_consumed_wh:.1f} Wh (margin: {sol_plan.power_margin_wh:.1f} Wh)")
            report.append(f"  Activities: {len(sol_plan.activities)}")
            report.append(f"  Samples: {len(sol_plan.sample_targets)}")
            report.append(f"  Images: {len(sol_plan.imaging_sessions)}")
            
            if sol_plan.contingency_triggers:
                report.append(f"    Contingencies: {', '.join(sol_plan.contingency_triggers)}")
        
        report.append("")
        
        # Contingency Plans
        if mission_plan.contingency_plans:
            report.append("CONTINGENCY PLANS")
            report.append("-" * 80)
            for cont in mission_plan.contingency_plans:
                report.append(f"  Trigger: {cont['trigger']}")
                report.append(f"    Condition: {cont['condition']}")
                report.append(f"    Action: {cont['action']}")
            report.append("")
        
        # Traceability
        if include_traceability:
            report.append("TRACEABILITY & VALIDATION")
            report.append("-" * 80)
            
            trace = self.audit_logger.generate_traceability_report(
                mission_plan.mission_name,
                "mission_plan"
            )
            
            report.append(f"  Events Logged: {trace['lifecycle']['total_events']}")
            report.append(f"  Validated: {trace['validation']['validated']}")
            report.append(f"  Safety Validated: {trace['validation'].get('nasa_requirements', {}).get('safety_validated', False)}")
            report.append(f"  Approval Required: {trace['approval']['approval_required']}")
            report.append(f"  Approved: {trace['approval']['approved']}")
        
        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _get_mission_context(self) -> Dict[str, Any]:
        """Get current mission context from NASA APIs."""
        
        try:
            # Get rover telemetry
            telemetry = self.nasa_client.get_rover_telemetry(sol=None)
            
            # Get environmental data
            environment = self.nasa_client.get_environmental_data(sol=None)
            
            # Get traverse data
            traverse = self.nasa_client.get_traverse_data()
            
            context = {
                'telemetry': telemetry,
                'environment': environment,
                'traverse': traverse,
                'current_mission': vars(self.current_mission) if self.current_mission else None
            }
            
            return context
            
        except Exception as e:
            self.logger.warning(f"Failed to get mission context: {e}")
            return {}
    
    def _ensure_dems_for_mission(self,
                                 start_location: tuple,
                                 targets: List[Dict[str, Any]]):
        """Ensure DEMs are available for mission area."""
        
        # Calculate bounding box
        all_lats = [start_location[0]] + [t.get('lat', 0) for t in targets]
        all_lons = [start_location[1]] + [t.get('lon', 0) for t in targets]
        
        lat_min = min(all_lats) - 0.1
        lat_max = max(all_lats) + 0.1
        lon_min = min(all_lons) - 0.1
        lon_max = max(all_lons) + 0.1
        
        # Download DEM
        self.dem_downloader.download_dem_for_region(
            region_name="mission_area",
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
    
    def _get_environmental_forecast(self, num_sols: int) -> Dict[int, Dict[str, Any]]:
        """Get environmental forecast for planning."""
        
        forecast = {}
        
        try:
            # Get MEDA data
            for sol in range(num_sols):
                env = self.nasa_client.get_environmental_data(sol=sol)
                
                forecast[sol] = {
                    'temperature_c': env.get('temperature_c', -60),
                    'dust_opacity': env.get('dust_opacity', 0.5),
                    'wind_speed_ms': env.get('wind_speed_ms', 5)
                }
        except Exception as e:
            self.logger.warning(f"Failed to get environmental forecast: {e}")
            
            # Use defaults
            for sol in range(num_sols):
                forecast[sol] = {
                    'temperature_c': -60,
                    'dust_opacity': 0.5,
                    'wind_speed_ms': 5
                }
        
        return forecast
    
    def _validate_mission_plan(self, mission_plan: MultiSolMissionPlan) -> Dict[str, Any]:
        """Validate mission plan against NASA standards."""
        
        checks = []
        
        # Safety checks
        checks.append({
            'category': 'safety',
            'name': 'Power margin check',
            'passed': all(sp.power_margin_wh >= 200 for sp in mission_plan.sol_plans),
            'confidence': 1.0
        })
        
        checks.append({
            'category': 'safety',
            'name': 'Drive distance check',
            'passed': all(sp.drive_distance_m <= 150 for sp in mission_plan.sol_plans),
            'confidence': 1.0
        })
        
        # Resource checks
        checks.append({
            'category': 'resource',
            'name': 'Power budget check',
            'passed': mission_plan.total_power_consumed_wh <= (900 * mission_plan.num_sols),
            'confidence': 1.0
        })
        
        # Science checks
        checks.append({
            'category': 'science',
            'name': 'Science value check',
            'passed': mission_plan.science_value_score >= 50.0,
            'confidence': 0.8
        })
        
        # Record validation
        validation = self.audit_logger.validate_resource(
            resource_id=mission_plan.mission_name,
            resource_type="mission_plan",
            validator_id="system_validator",
            checks=checks
        )
        
        return vars(validation)
    
    def _generate_emergency_contingency(self,
                                       hazard_type: str,
                                       plan: MultiSolMissionPlan,
                                       state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate emergency contingency plan."""
        
        contingencies = {
            'DUST_STORM': {
                'trigger': 'DUST_STORM',
                'condition': 'Unexpected dust storm detected',
                'action': 'Enter safe mode, minimize power, wait for clearing',
                'fallback_sols': 2,
                'power_reserve_required_wh': 400.0
            },
            'LOW_POWER': {
                'trigger': 'LOW_POWER',
                'condition': 'Power below critical threshold',
                'action': 'Cancel non-essential activities, optimize solar positioning',
                'fallback_sols': 1,
                'power_reserve_required_wh': 200.0
            },
            'EQUIPMENT_FAILURE': {
                'trigger': 'EQUIPMENT_FAILURE',
                'condition': 'Critical equipment malfunction',
                'action': 'Activate backup systems, contact mission control',
                'fallback_sols': 3,
                'power_reserve_required_wh': 500.0
            }
        }
        
        return contingencies.get(hazard_type, contingencies['EQUIPMENT_FAILURE'])


# Convenience function for quick system access
def create_production_system(openai_api_key: Optional[str] = None,
                             nasa_api_key: Optional[str] = None) -> ProductionMissionSystem:
    """Create production mission system with sensible defaults."""
    return ProductionMissionSystem(
        openai_api_key=openai_api_key,
        nasa_api_key=nasa_api_key
    )
