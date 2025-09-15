TERRAIN_ANALYSIS_PROMPT = """
Analyze this Mars rover image as a NASA mission planner would:

1. SAFETY ASSESSMENT: Identify potential hazards (rocks, slopes, soft terrain)
2. SCIENCE TARGETS: Locate interesting geological features for sampling
3. TRAVERSABILITY: Assess routes considering rover limitations
4. RESOURCE IMPACT: Estimate power/time requirements for proposed activities

Context: Sol {sol_number}, Location: {coordinates}, Mission Phase: {phase}
Environmental: Temperature {temp}°C, Dust opacity {dust}, Wind {wind}m/s

Respond with structured analysis suitable for automated mission planning.
"""

MISSION_PLANNING_PROMPT = """
Generate a mission plan as a NASA mission planner for Perseverance rover:

OBJECTIVES: {science_goals}
CONSTRAINTS: Power {power_budget}Wh, Time {time_budget}min, Weather {conditions}
ROVER STATE: Location {current_location}, Battery {battery_level}%, Systems {status}

Create activity sequence following NASA operational procedures:
1. Pre-activity system checks
2. Movement commands with waypoints
3. Science operations with instrument sequences
4. Post-activity data transmission
5. Resource management and contingencies

Output format compatible with NASA command sequencing tools.
"""
