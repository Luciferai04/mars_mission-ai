#!/usr/bin/env python3
"""
Mars Sample Return (MSR) Sample Caching System

Manages sample collection, storage, caching depot operations, and
relay coordination for Mars Sample Return mission architecture.
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json


@dataclass
class Sample:
    """Mars rock/soil sample collected by rover."""
    sample_id: str
    collection_sol: int
    collection_location: Tuple[float, float]  # (lat, lon)
    sample_type: str  # rock, soil, atmospheric, core
    mass_grams: float
    
    # Scientific metadata
    rock_type: Optional[str] = None
    geological_context: Optional[str] = None
    science_priority: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    instruments_used: List[str] = field(default_factory=list)
    
    # Storage metadata
    tube_number: int = 0
    sealed: bool = False
    cached: bool = False
    cache_location: Optional[str] = None
    cached_sol: Optional[int] = None
    
    # Quality and integrity
    quality_score: float = 1.0  # 0-1
    contamination_risk: str = "LOW"  # LOW, MEDIUM, HIGH
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleCache:
    """Sample depot cache location on Mars surface."""
    cache_id: str
    location: Tuple[float, float]
    established_sol: int
    
    # Capacity
    max_capacity: int = 10  # Number of sample tubes
    current_samples: List[Sample] = field(default_factory=list)
    
    # Environmental
    terrain_type: str = "flat"
    accessibility_score: float = 1.0  # 0-1, how easy for retrieval
    landmark_visibility: str = "HIGH"  # HIGH, MEDIUM, LOW
    
    # Operational
    status: str = "ACTIVE"  # ACTIVE, FULL, COMPROMISED, RETRIEVED
    retrieval_priority: int = 1  # Lower is higher priority
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_full(self) -> bool:
        """Check if cache is at capacity."""
        return len(self.current_samples) >= self.max_capacity
    
    def add_sample(self, sample: Sample) -> bool:
        """Add sample to cache."""
        if self.is_full():
            return False
        
        sample.cached = True
        sample.cache_location = self.cache_id
        sample.cached_sol = datetime.utcnow().timestamp()  # Simplified
        
        self.current_samples.append(sample)
        
        if self.is_full():
            self.status = "FULL"
        
        return True


@dataclass
class MSRMission:
    """Mars Sample Return mission state."""
    mission_id: str
    
    # Sample inventory
    collected_samples: List[Sample] = field(default_factory=list)
    onboard_samples: List[Sample] = field(default_factory=list)
    cached_samples: List[Sample] = field(default_factory=list)
    
    # Cache depots
    sample_caches: List[SampleCache] = field(default_factory=list)
    
    # Capacity constraints
    max_onboard_samples: int = 43  # Perseverance capacity
    target_total_samples: int = 30  # Mission target
    
    # Relay coordination
    retrieval_lander_eta_sol: Optional[int] = None
    sample_fetch_rover_active: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class MSRSampleCachingSystem:
    """Manages MSR sample collection, caching, and retrieval operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # NASA MSR requirements
        self.min_cache_spacing_km = 2.0  # Minimum distance between caches
        self.cache_redundancy_factor = 2  # Duplicate caches for safety
        self.samples_per_cache = 5  # Standard cache size
        
        # Retrieval constraints
        self.max_retrieval_distance_km = 50.0  # SFR operational range
        self.cache_visibility_requirement = "HIGH"
    
    def plan_sample_caching_strategy(self,
                                    mission: MSRMission,
                                    traverse_plan: Dict[str, Any],
                                    science_targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate optimal sample caching strategy for mission.
        
        Args:
            mission: Current MSR mission state
            traverse_plan: Multi-sol traverse plan
            science_targets: Prioritized sampling targets
            
        Returns:
            Comprehensive caching strategy with depot locations and operations
        """
        self.logger.info("Planning MSR sample caching strategy")
        
        # Determine optimal cache locations
        cache_sites = self._identify_cache_locations(
            traverse_plan, science_targets
        )
        
        # Allocate samples to caches
        sample_allocation = self._allocate_samples_to_caches(
            mission, cache_sites, science_targets
        )
        
        # Generate caching operations schedule
        caching_schedule = self._generate_caching_schedule(
            mission, sample_allocation, traverse_plan
        )
        
        # Create redundant backup caches
        backup_caches = self._plan_backup_caches(cache_sites, sample_allocation)
        
        # Retrieval coordination
        retrieval_plan = self._plan_sample_retrieval(
            cache_sites, backup_caches, mission
        )
        
        return {
            'cache_sites': cache_sites,
            'backup_caches': backup_caches,
            'sample_allocation': sample_allocation,
            'caching_schedule': caching_schedule,
            'retrieval_plan': retrieval_plan,
            'total_samples_planned': len(science_targets),
            'total_caches_planned': len(cache_sites) + len(backup_caches),
            'metadata': {
                'planned_at': datetime.utcnow().isoformat(),
                'mission_id': mission.mission_id
            }
        }
    
    def execute_sample_collection(self,
                                  sample_target: Dict[str, Any],
                                  sol: int,
                                  mission: MSRMission) -> Sample:
        """Execute sample collection operation.
        
        Args:
            sample_target: Target location and metadata
            sol: Current sol
            mission: Mission state
            
        Returns:
            Collected sample object
        """
        self.logger.info(f"Collecting sample at {sample_target['location']} on Sol {sol}")
        
        # Check capacity
        if len(mission.onboard_samples) >= mission.max_onboard_samples:
            raise ValueError("Onboard sample storage at capacity - cache samples first")
        
        # Create sample
        sample = Sample(
            sample_id=f"SAMPLE_{sol:04d}_{len(mission.collected_samples):03d}",
            collection_sol=sol,
            collection_location=sample_target['location'],
            sample_type=sample_target.get('type', 'rock'),
            mass_grams=sample_target.get('mass_grams', 15.0),
            rock_type=sample_target.get('rock_type'),
            geological_context=sample_target.get('geological_context'),
            science_priority=sample_target.get('priority', 'MEDIUM'),
            instruments_used=sample_target.get('instruments', ['DRILL', 'SEAL']),
            tube_number=len(mission.collected_samples) + 1,
            sealed=True,
            quality_score=sample_target.get('quality_score', 0.95),
            contamination_risk=sample_target.get('contamination_risk', 'LOW'),
            metadata=sample_target.get('metadata', {})
        )
        
        # Add to mission inventory
        mission.collected_samples.append(sample)
        mission.onboard_samples.append(sample)
        
        self.logger.info(f"Sample {sample.sample_id} collected and sealed (Tube {sample.tube_number})")
        
        return sample
    
    def execute_cache_deployment(self,
                                cache_location: Tuple[float, float],
                                samples_to_cache: List[Sample],
                                sol: int,
                                mission: MSRMission) -> SampleCache:
        """Deploy sample cache at specified location.
        
        Args:
            cache_location: (lat, lon) for cache
            samples_to_cache: Samples to deposit
            sol: Current sol
            mission: Mission state
            
        Returns:
            Deployed cache object
        """
        self.logger.info(f"Deploying cache at {cache_location} with {len(samples_to_cache)} samples")
        
        # Validate samples are onboard
        for sample in samples_to_cache:
            if sample not in mission.onboard_samples:
                raise ValueError(f"Sample {sample.sample_id} not onboard rover")
        
        # Create cache
        cache = SampleCache(
            cache_id=f"CACHE_{len(mission.sample_caches):02d}",
            location=cache_location,
            established_sol=sol,
            max_capacity=self.samples_per_cache * 2,  # Allow flexibility
            terrain_type="flat",
            accessibility_score=0.95,
            landmark_visibility="HIGH",
            retrieval_priority=len(mission.sample_caches) + 1
        )
        
        # Transfer samples to cache
        for sample in samples_to_cache:
            success = cache.add_sample(sample)
            if success:
                mission.onboard_samples.remove(sample)
                mission.cached_samples.append(sample)
                self.logger.info(f"Sample {sample.sample_id} cached at {cache.cache_id}")
            else:
                self.logger.error(f"Failed to cache sample {sample.sample_id}")
        
        # Add to mission
        mission.sample_caches.append(cache)
        
        self.logger.info(f"Cache {cache.cache_id} deployed with {len(cache.current_samples)} samples")
        
        return cache
    
    def _identify_cache_locations(self,
                                 traverse_plan: Dict[str, Any],
                                 targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify optimal cache depot locations."""
        
        cache_sites = []
        
        # Strategy: Place caches at intervals along traverse route
        # ensuring accessibility and landmark visibility
        
        if 'sol_plans' in traverse_plan:
            sol_plans = traverse_plan['sol_plans']
            
            # Place cache every ~5 sols of traverse
            for i in range(0, len(sol_plans), 5):
                if i >= len(sol_plans):
                    break
                
                sol_plan = sol_plans[i]
                location = sol_plan.get('end_location', (0, 0))
                
                cache_site = {
                    'location': location,
                    'sol': sol_plan.get('sol_number', 0),
                    'terrain_type': 'flat',
                    'accessibility_score': 0.9,
                    'visibility': 'HIGH',
                    'nearby_landmarks': ['Delta Formation', 'Crater Rim'],
                    'capacity': self.samples_per_cache
                }
                
                cache_sites.append(cache_site)
        
        else:
            # Fallback: distribute caches based on targets
            num_caches = max(1, len(targets) // self.samples_per_cache)
            
            for i in range(num_caches):
                idx = i * len(targets) // num_caches
                if idx < len(targets):
                    target = targets[idx]
                    
                    cache_site = {
                        'location': target.get('location', (0, 0)),
                        'sol': target.get('sol', 0),
                        'terrain_type': 'flat',
                        'accessibility_score': 0.85,
                        'visibility': 'MEDIUM',
                        'capacity': self.samples_per_cache
                    }
                    
                    cache_sites.append(cache_site)
        
        self.logger.info(f"Identified {len(cache_sites)} primary cache sites")
        
        return cache_sites
    
    def _allocate_samples_to_caches(self,
                                   mission: MSRMission,
                                   cache_sites: List[Dict[str, Any]],
                                   targets: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Allocate samples to specific caches."""
        
        allocation = {site['location']: [] for site in cache_sites}
        
        # Simple allocation: nearest cache to target
        for target in targets:
            target_loc = target.get('location', (0, 0))
            
            # Find nearest cache
            nearest_cache = None
            min_distance = float('inf')
            
            for site in cache_sites:
                distance = self._calculate_distance(target_loc, site['location'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_cache = site['location']
            
            if nearest_cache:
                sample_id = f"SAMPLE_{target.get('target_id', 'UNKNOWN')}"
                allocation[nearest_cache].append(sample_id)
        
        return allocation
    
    def _generate_caching_schedule(self,
                                  mission: MSRMission,
                                  allocation: Dict[str, List[str]],
                                  traverse_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate time-sequenced caching operations."""
        
        schedule = []
        
        for cache_loc, sample_ids in allocation.items():
            if not sample_ids:
                continue
            
            schedule.append({
                'operation': 'CACHE_DEPLOYMENT',
                'location': cache_loc,
                'samples': sample_ids,
                'estimated_duration_hours': 2.0,
                'power_required_wh': 100.0,
                'prerequisites': ['samples_collected', 'location_reached'],
                'priority': 'HIGH'
            })
        
        return schedule
    
    def _plan_backup_caches(self,
                           primary_caches: List[Dict[str, Any]],
                           allocation: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Plan redundant backup caches for mission safety."""
        
        backup_caches = []
        
        # Create backup for each primary cache
        for primary in primary_caches:
            # Offset backup cache location slightly (simulate nearby site)
            primary_loc = primary['location']
            backup_loc = (
                primary_loc[0] + 0.001,  # ~60m offset
                primary_loc[1] + 0.001
            )
            
            backup = {
                'location': backup_loc,
                'primary_cache': primary['location'],
                'sol': primary['sol'] + 1,  # Deploy day after primary
                'terrain_type': 'flat',
                'accessibility_score': 0.85,
                'visibility': 'HIGH',
                'capacity': primary['capacity'],
                'purpose': 'REDUNDANCY'
            }
            
            backup_caches.append(backup)
        
        self.logger.info(f"Planned {len(backup_caches)} backup caches")
        
        return backup_caches
    
    def _plan_sample_retrieval(self,
                              primary_caches: List[Dict[str, Any]],
                              backup_caches: List[Dict[str, Any]],
                              mission: MSRMission) -> Dict[str, Any]:
        """Plan Sample Fetch Rover (SFR) retrieval operations."""
        
        all_caches = primary_caches + backup_caches
        
        # Prioritize caches by science value and accessibility
        prioritized = sorted(
            all_caches,
            key=lambda c: c.get('accessibility_score', 0.5),
            reverse=True
        )
        
        retrieval_sequence = []
        for i, cache in enumerate(prioritized):
            retrieval_sequence.append({
                'sequence_number': i + 1,
                'cache_location': cache['location'],
                'estimated_sfr_sol': mission.retrieval_lander_eta_sol + i if mission.retrieval_lander_eta_sol else None,
                'priority': 'HIGH' if i < len(primary_caches) else 'BACKUP',
                'retrieval_method': 'SFR_AUTONOMOUS',
                'contingency': 'Manual retrieval if SFR fails'
            })
        
        return {
            'total_caches_to_retrieve': len(all_caches),
            'retrieval_sequence': retrieval_sequence,
            'estimated_duration_sols': len(all_caches) * 2,  # 2 sols per cache
            'backup_caches_available': len(backup_caches),
            'retrieval_feasibility': 'HIGH' if len(all_caches) < 20 else 'MEDIUM'
        }
    
    def _calculate_distance(self,
                          pos1: Tuple[float, float],
                          pos2: Tuple[float, float]) -> float:
        """Calculate distance between two positions."""
        lat_diff = abs(pos1[0] - pos2[0])
        lon_diff = abs(pos1[1] - pos2[1])
        return ((lat_diff**2 + lon_diff**2)**0.5) * 59.0  # km on Mars
    
    def generate_cache_manifest(self, mission: MSRMission) -> Dict[str, Any]:
        """Generate comprehensive manifest of all cached samples."""
        
        manifest = {
            'mission_id': mission.mission_id,
            'generated_at': datetime.utcnow().isoformat(),
            'total_samples_collected': len(mission.collected_samples),
            'total_samples_cached': len(mission.cached_samples),
            'total_samples_onboard': len(mission.onboard_samples),
            'total_caches': len(mission.sample_caches),
            'caches': []
        }
        
        for cache in mission.sample_caches:
            cache_entry = {
                'cache_id': cache.cache_id,
                'location': cache.location,
                'established_sol': cache.established_sol,
                'status': cache.status,
                'sample_count': len(cache.current_samples),
                'capacity': cache.max_capacity,
                'samples': [
                    {
                        'sample_id': s.sample_id,
                        'tube_number': s.tube_number,
                        'type': s.sample_type,
                        'priority': s.science_priority,
                        'collection_sol': s.collection_sol,
                        'quality': s.quality_score
                    }
                    for s in cache.current_samples
                ]
            }
            manifest['caches'].append(cache_entry)
        
        return manifest
    
    def export_manifest_for_retrieval(self, mission: MSRMission, filepath: str):
        """Export cache manifest to file for SFR mission planning."""
        
        manifest = self.generate_cache_manifest(mission)
        
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Cache manifest exported to {filepath}")
        
        return manifest
