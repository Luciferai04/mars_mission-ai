#!/usr/bin/env python3
"""
Natural Language Query Interface

Translates natural language mission planning requests into actionable
API calls and planning workflows.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import requests
from datetime import datetime


class NaturalLanguageQueryInterface:
    """Translates natural language to mission planning API calls."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for NL interface")

        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o"
        self.logger = logging.getLogger(__name__)

        # Available API endpoints and capabilities
        self.capabilities = {
            "plan_mission": {
                "description": "Plan multi-sol rover mission",
                "parameters": ["start_location", "targets", "num_sols", "objectives"],
            },
            "analyze_terrain": {
                "description": "Analyze terrain imagery for hazards",
                "parameters": ["image_path", "camera_type", "sol"],
            },
            "optimize_route": {
                "description": "Optimize path between waypoints",
                "parameters": ["start", "end", "waypoints", "constraints"],
            },
            "check_resources": {
                "description": "Verify power/thermal constraints",
                "parameters": ["activities", "environment"],
            },
            "identify_samples": {
                "description": "Find high-priority sample targets",
                "parameters": ["region", "science_goals"],
            },
            "generate_contingency": {
                "description": "Create contingency plans",
                "parameters": ["mission_plan", "hazard_type"],
            },
        }

    def process_query(
        self, natural_language_query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process natural language query into mission planning actions.

        Args:
            natural_language_query: User's mission request in plain English
            context: Optional context (current rover state, recent missions, etc.)

        Returns:
            Structured plan with API calls, parameters, and execution sequence
        """
        self.logger.info(f"Processing NL query: {natural_language_query[:100]}...")

        context = context or {}

        # Generate structured plan using GPT-4o
        plan = self._generate_plan_from_nl(natural_language_query, context)

        # Validate and enrich plan
        validated_plan = self._validate_plan(plan)

        # Add execution metadata
        validated_plan["metadata"] = {
            "original_query": natural_language_query,
            "processed_at": datetime.utcnow().isoformat(),
            "model": self.model,
            "context": context,
        }

        return validated_plan

    def execute_plan(self, plan: Dict[str, Any], api_client: Any) -> Dict[str, Any]:
        """Execute generated plan using mission planning API.

        Args:
            plan: Structured plan from process_query
            api_client: Mission planning API client instance

        Returns:
            Execution results with outputs from each step
        """
        results = {
            "plan_id": plan.get("plan_id"),
            "steps_executed": 0,
            "steps_failed": 0,
            "outputs": [],
            "errors": [],
        }

        for step in plan.get("steps", []):
            try:
                self.logger.info(f"Executing step: {step['action']}")

                # Call appropriate API method
                output = self._execute_step(step, api_client)

                results["outputs"].append(
                    {
                        "step": step["step_number"],
                        "action": step["action"],
                        "success": True,
                        "output": output,
                    }
                )
                results["steps_executed"] += 1

            except Exception as e:
                self.logger.error(f"Step {step['step_number']} failed: {e}")
                results["errors"].append(
                    {
                        "step": step["step_number"],
                        "action": step["action"],
                        "error": str(e),
                    }
                )
                results["steps_failed"] += 1

                # Check if failure is critical
                if step.get("critical", False):
                    results["aborted"] = True
                    break

        results["success"] = results["steps_failed"] == 0
        return results

    def _generate_plan_from_nl(
        self, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use GPT-4o to convert NL query to structured plan."""

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)

        response = self._call_gpt4o(system_prompt, user_prompt)

        # Parse JSON response
        try:
            plan = self._extract_json_from_response(response)
            return plan
        except Exception as e:
            self.logger.error(f"Failed to parse plan: {e}")
            return self._get_fallback_plan(query)

    def _build_system_prompt(self) -> str:
        """Build system prompt for plan generation."""

        capabilities_str = json.dumps(self.capabilities, indent=2)

        return f"""You are a NASA Mars rover mission planning AI assistant.

Your role is to translate natural language mission requests into structured, executable plans using the Mars Mission Planning API.

**AVAILABLE CAPABILITIES:**
{capabilities_str}

**YOUR TASK:**
1. Understand the user's mission request
2. Break it down into a sequence of API calls
3. Extract or infer required parameters
4. Generate a structured execution plan

**OUTPUT FORMAT:**
Return a JSON plan with this exact structure:
{{
  "plan_id": "unique_identifier",
  "mission_type": "traverse|sample|observation|emergency",
  "summary": "Brief description of what will be done",
  "steps": [
    {{
      "step_number": 1,
      "action": "API_endpoint_name",
      "description": "What this step does",
      "parameters": {{
        "param_name": "param_value"
      }},
      "critical": true/false,
      "depends_on": [previous_step_numbers]
    }}
  ],
  "expected_outcomes": [
    "Outcome 1",
    "Outcome 2"
  ],
  "safety_considerations": [
    "Safety note 1"
  ]
}}

**PARAMETER INFERENCE RULES:**
- If locations aren't specified, use "current_position" or ask for clarification
- Default to 2-3 sols for multi-sol missions unless specified
- Prioritize safety: always include resource checks and hazard analysis
- For sample collection, include terrain analysis before approach
- Add contingency planning for multi-sol missions

**EXAMPLES:**

Query: "Plan a 2-sol traverse to the delta focusing on sample collection"
→ Steps: analyze_terrain, identify_samples, plan_mission, check_resources, generate_contingency

Query: "Check if we can safely reach waypoint Alpha"
→ Steps: optimize_route, analyze_terrain, check_resources

Query: "Find high-priority sampling targets in Jezero crater"
→ Steps: identify_samples, analyze_terrain

Be precise, conservative, and NASA-grade in your planning."""

    def _build_user_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build user prompt with query and context."""

        context_str = ""
        if context:
            context_str = f"\n**CURRENT CONTEXT:**\n{json.dumps(context, indent=2)}\n"

        return f"""**MISSION REQUEST:**
{query}
{context_str}
Generate a structured execution plan for this request."""

    def _call_gpt4o(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call GPT-4o API."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        response = requests.post(
            self.base_url, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        return response.json()

    def _extract_json_from_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract JSON plan from GPT-4o response."""

        content = response["choices"][0]["message"]["content"]

        # Find JSON block
        json_start = content.find("{")
        json_end = content.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")

    def _validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enrich generated plan."""

        # Ensure required fields
        if "plan_id" not in plan:
            plan["plan_id"] = f"plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        if "steps" not in plan or not plan["steps"]:
            raise ValueError("Plan must contain at least one step")

        # Validate each step
        for step in plan["steps"]:
            if "action" not in step:
                raise ValueError(f"Step {step.get('step_number')} missing action")

            # Check action is valid
            action = step["action"]
            if action not in self.capabilities:
                self.logger.warning(f"Unknown action: {action}")

            # Ensure parameters dict exists
            if "parameters" not in step:
                step["parameters"] = {}

        return plan

    def _get_fallback_plan(self, query: str) -> Dict[str, Any]:
        """Generate fallback plan when GPT-4o fails."""

        return {
            "plan_id": f"fallback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "mission_type": "unknown",
            "summary": f"Manual review required for: {query}",
            "steps": [
                {
                    "step_number": 1,
                    "action": "manual_review",
                    "description": "Request human mission planner review",
                    "parameters": {"query": query},
                    "critical": True,
                }
            ],
            "expected_outcomes": ["Human planner will review and create plan manually"],
            "safety_considerations": ["NL parsing failed - human oversight required"],
            "error": "Failed to parse natural language query automatically",
        }

    def _execute_step(self, step: Dict[str, Any], api_client: Any) -> Any:
        """Execute a single plan step."""

        action = step["action"]
        params = step.get("parameters", {})

        # Map actions to API methods
        method_map = {
            "plan_mission": "plan_multi_sol_mission",
            "analyze_terrain": "analyze_terrain_image",
            "optimize_route": "optimize_route",
            "check_resources": "check_resource_constraints",
            "identify_samples": "identify_sample_targets",
            "generate_contingency": "generate_contingency_plan",
        }

        method_name = method_map.get(action)
        if not method_name:
            raise ValueError(f"Unknown action: {action}")

        # Get method from API client
        if not hasattr(api_client, method_name):
            raise AttributeError(f"API client missing method: {method_name}")

        method = getattr(api_client, method_name)

        # Call with parameters
        return method(**params)


class ConversationalInterface:
    """Multi-turn conversational interface for mission planning."""

    def __init__(self, nl_interface: NaturalLanguageQueryInterface):
        self.nl_interface = nl_interface
        self.conversation_history = []
        self.current_plan = None
        self.logger = logging.getLogger(__name__)

    def chat(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process user message in conversational context.

        Args:
            user_message: User's input
            context: Additional context

        Returns:
            Assistant's response
        """
        self.conversation_history.append(
            {
                "role": "user",
                "message": user_message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Add conversation history to context
        enriched_context = context or {}
        enriched_context["conversation_history"] = self.conversation_history[
            -5:
        ]  # Last 5 messages
        enriched_context["current_plan"] = self.current_plan

        # Process query
        try:
            plan = self.nl_interface.process_query(user_message, enriched_context)
            self.current_plan = plan

            response = self._format_plan_as_response(plan)

        except Exception as e:
            self.logger.error(f"Chat processing failed: {e}")
            response = f"I encountered an issue: {e}. Could you rephrase your request?"

        self.conversation_history.append(
            {
                "role": "assistant",
                "message": response,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        return response

    def _format_plan_as_response(self, plan: Dict[str, Any]) -> str:
        """Format plan as natural language response."""

        response = f"**Mission Plan: {plan.get('summary', 'Unnamed Mission')}**\n\n"

        response += f"I've created a {len(plan.get('steps', []))}-step plan:\n\n"

        for step in plan.get("steps", []):
            response += (
                f"{step['step_number']}. {step.get('description', step['action'])}\n"
            )

        response += "\n**Expected Outcomes:**\n"
        for outcome in plan.get("expected_outcomes", []):
            response += f"- {outcome}\n"

        if plan.get("safety_considerations"):
            response += "\n**Safety Considerations:**\n"
            for safety in plan["safety_considerations"]:
                response += f"  {safety}\n"

        response += "\nWould you like me to execute this plan?"

        return response

    def confirm_and_execute(self, api_client: Any) -> Dict[str, Any]:
        """Execute current plan after user confirmation."""

        if not self.current_plan:
            raise ValueError("No plan to execute")

        return self.nl_interface.execute_plan(self.current_plan, api_client)
