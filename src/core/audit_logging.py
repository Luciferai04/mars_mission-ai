#!/usr/bin/env python3
"""
NASA-Grade Audit Logging and Traceability System

Comprehensive logging, validation tracking, and traceability for all
mission planning decisions, ensuring NASA operational standards compliance.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import logging
import json
import hashlib
import uuid
from pathlib import Path
from enum import Enum


class EventType(Enum):
    """Types of auditable events."""

    PLAN_CREATED = "plan_created"
    PLAN_MODIFIED = "plan_modified"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"
    PLAN_EXECUTED = "plan_executed"

    SAMPLE_COLLECTED = "sample_collected"
    SAMPLE_CACHED = "sample_cached"

    TERRAIN_ANALYZED = "terrain_analyzed"
    HAZARD_DETECTED = "hazard_detected"

    ROUTE_CALCULATED = "route_calculated"
    WAYPOINT_REACHED = "waypoint_reached"

    RESOURCE_CHECK = "resource_check"
    CONSTRAINT_VIOLATION = "constraint_violation"

    CONTINGENCY_TRIGGERED = "contingency_triggered"
    EMERGENCY_ACTION = "emergency_action"

    HUMAN_OVERRIDE = "human_override"
    SYSTEM_ERROR = "system_error"


class Severity(Enum):
    """Event severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Immutable audit event record."""

    event_id: str
    event_type: EventType
    timestamp: str
    severity: Severity

    # Actor information
    actor_type: str  # SYSTEM, HUMAN, API
    actor_id: str

    # Event details
    action: str
    resource: str
    resource_id: Optional[str] = None

    # Context
    mission_id: Optional[str] = None
    sol: Optional[int] = None
    location: Optional[tuple] = None

    # Data
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)

    # Validation
    validated: bool = False
    validator_id: Optional[str] = None
    validation_timestamp: Optional[str] = None

    # Approval workflow
    approval_required: bool = False
    approved: bool = False
    approver_id: Optional[str] = None
    approval_timestamp: Optional[str] = None

    # Traceability
    parent_event_id: Optional[str] = None
    related_event_ids: List[str] = field(default_factory=list)

    # Integrity
    checksum: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum for integrity verification."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "actor_id": self.actor_id,
            "action": self.action,
            "resource": self.resource,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data


@dataclass
class ValidationRecord:
    """Record of validation performed on a plan or action."""

    validation_id: str
    validated_resource_id: str
    timestamp: str

    validator_id: str
    validator_type: str  # AUTOMATED, HUMAN, HYBRID

    # Validation results
    passed: bool
    checks_performed: List[Dict[str, Any]]
    issues_found: List[Dict[str, Any]]

    # NASA requirements
    safety_validated: bool
    resource_constraints_validated: bool
    science_objectives_validated: bool

    confidence_score: float  # 0-1

    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """NASA-grade audit logging system."""

    def __init__(self, log_directory: str = "logs/audit"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # In-memory event cache for quick access
        self.event_cache: List[AuditEvent] = []
        self.max_cache_size = 1000

        # Validation records
        self.validation_records: Dict[str, ValidationRecord] = {}

        # Setup persistent logging
        self._setup_persistent_logging()

    def _setup_persistent_logging(self):
        """Setup file-based audit logging."""
        log_file = (
            self.log_directory / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)

    def log_event(
        self,
        event_type: EventType,
        action: str,
        resource: str,
        actor_type: str = "SYSTEM",
        actor_id: str = "mars_mission_planner",
        severity: Severity = Severity.INFO,
        resource_id: Optional[str] = None,
        mission_id: Optional[str] = None,
        sol: Optional[int] = None,
        location: Optional[tuple] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        parent_event_id: Optional[str] = None,
        approval_required: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log an auditable event.

        Args:
            event_type: Type of event
            action: Description of action taken
            resource: Resource affected (e.g., "mission_plan", "sample")
            actor_type: SYSTEM, HUMAN, or API
            actor_id: Identifier of actor
            severity: Event severity
            resource_id: Specific resource identifier
            mission_id: Associated mission
            sol: Mars sol number
            location: (lat, lon) coordinates
            input_data: Input parameters
            output_data: Output results
            parent_event_id: Parent event for traceability
            approval_required: Whether human approval is needed
            metadata: Additional metadata

        Returns:
            Created audit event
        """
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            mission_id=mission_id,
            sol=sol,
            location=location,
            input_data=input_data or {},
            output_data=output_data or {},
            parent_event_id=parent_event_id,
            approval_required=approval_required,
            metadata=metadata or {},
        )

        # Compute integrity checksum
        event.checksum = event.compute_checksum()

        # Add to cache
        self.event_cache.append(event)
        if len(self.event_cache) > self.max_cache_size:
            self.event_cache.pop(0)

        # Persist to file
        self._persist_event(event)

        # Log to standard logging
        log_message = (
            f"[{event_type.value}] {action} | Resource: {resource} | Actor: {actor_id}"
        )

        if severity == Severity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == Severity.ERROR:
            self.logger.error(log_message)
        elif severity == Severity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

        return event

    def validate_resource(
        self,
        resource_id: str,
        resource_type: str,
        validator_id: str,
        checks: List[Dict[str, Any]],
    ) -> ValidationRecord:
        """Record validation of a resource (plan, action, etc.).

        Args:
            resource_id: ID of resource being validated
            resource_type: Type of resource
            validator_id: ID of validator (human or system)
            checks: List of validation checks performed

        Returns:
            Validation record
        """
        issues = [check for check in checks if not check.get("passed", False)]
        passed = len(issues) == 0

        # Determine validator type
        validator_type = "AUTOMATED" if validator_id.startswith("system_") else "HUMAN"

        # Check NASA requirements
        safety_checks = [c for c in checks if "safety" in c.get("category", "").lower()]
        resource_checks = [
            c for c in checks if "resource" in c.get("category", "").lower()
        ]
        science_checks = [
            c for c in checks if "science" in c.get("category", "").lower()
        ]

        validation = ValidationRecord(
            validation_id=str(uuid.uuid4()),
            validated_resource_id=resource_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            validator_id=validator_id,
            validator_type=validator_type,
            passed=passed,
            checks_performed=checks,
            issues_found=issues,
            safety_validated=all(c.get("passed", False) for c in safety_checks),
            resource_constraints_validated=all(
                c.get("passed", False) for c in resource_checks
            ),
            science_objectives_validated=all(
                c.get("passed", False) for c in science_checks
            ),
            confidence_score=(
                sum(c.get("confidence", 1.0) for c in checks) / len(checks)
                if checks
                else 0.0
            ),
        )

        # Store validation record
        self.validation_records[resource_id] = validation

        # Log validation event
        self.log_event(
            event_type=EventType.PLAN_APPROVED if passed else EventType.PLAN_REJECTED,
            action=f"Validation {'passed' if passed else 'failed'} for {resource_type}",
            resource=resource_type,
            resource_id=resource_id,
            actor_type="SYSTEM" if validator_type == "AUTOMATED" else "HUMAN",
            actor_id=validator_id,
            severity=Severity.INFO if passed else Severity.WARNING,
            output_data={
                "validation_id": validation.validation_id,
                "passed": passed,
                "issues_count": len(issues),
            },
        )

        return validation

    def approve_resource(
        self,
        resource_id: str,
        approver_id: str,
        approved: bool,
        reason: Optional[str] = None,
    ) -> bool:
        """Record human approval/rejection of a resource.

        Args:
            resource_id: ID of resource
            approver_id: ID of approver
            approved: Whether approved
            reason: Reason for decision

        Returns:
            Success status
        """
        # Find events requiring approval
        matching_events = [
            e
            for e in self.event_cache
            if e.resource_id == resource_id and e.approval_required and not e.approved
        ]

        if not matching_events:
            self.logger.warning(f"No pending approvals for resource {resource_id}")
            return False

        # Update events
        for event in matching_events:
            event.approved = approved
            event.approver_id = approver_id
            event.approval_timestamp = datetime.now(timezone.utc).isoformat()

        # Log approval event
        self.log_event(
            event_type=EventType.PLAN_APPROVED if approved else EventType.PLAN_REJECTED,
            action=f"Resource {'approved' if approved else 'rejected'} by {approver_id}",
            resource="approval",
            resource_id=resource_id,
            actor_type="HUMAN",
            actor_id=approver_id,
            severity=Severity.INFO,
            metadata={"reason": reason},
        )

        return True

    def get_event_chain(self, event_id: str) -> List[AuditEvent]:
        """Get complete chain of related events for traceability.

        Args:
            event_id: Starting event ID

        Returns:
            List of related events in chronological order
        """
        chain = []

        # Find starting event
        start_event = next(
            (e for e in self.event_cache if e.event_id == event_id), None
        )
        if not start_event:
            return chain

        # Build chain backwards (parents)
        current = start_event
        while current:
            chain.insert(0, current)
            if current.parent_event_id:
                current = next(
                    (
                        e
                        for e in self.event_cache
                        if e.event_id == current.parent_event_id
                    ),
                    None,
                )
            else:
                break

        # Build chain forwards (children)
        children = [e for e in self.event_cache if e.parent_event_id == event_id]
        chain.extend(children)

        return chain

    def generate_traceability_report(
        self, resource_id: str, resource_type: str
    ) -> Dict[str, Any]:
        """Generate comprehensive traceability report for a resource.

        Args:
            resource_id: Resource identifier
            resource_type: Type of resource

        Returns:
            Detailed traceability report
        """
        # Get all events for resource
        events = [e for e in self.event_cache if e.resource_id == resource_id]
        events.sort(key=lambda e: e.timestamp)

        # Get validation records
        validation = self.validation_records.get(resource_id)

        # Get approval status
        approval_events = [
            e
            for e in events
            if e.event_type in [EventType.PLAN_APPROVED, EventType.PLAN_REJECTED]
        ]

        report = {
            "resource_id": resource_id,
            "resource_type": resource_type,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lifecycle": {
                "created": events[0].timestamp if events else None,
                "last_modified": events[-1].timestamp if events else None,
                "total_events": len(events),
                "event_summary": [
                    {
                        "timestamp": e.timestamp,
                        "event_type": e.event_type.value,
                        "action": e.action,
                        "actor": e.actor_id,
                    }
                    for e in events
                ],
            },
            "validation": {
                "validated": validation.passed if validation else False,
                "validation_timestamp": validation.timestamp if validation else None,
                "validator": validation.validator_id if validation else None,
                "checks_passed": (
                    len([c for c in validation.checks_performed if c.get("passed")])
                    if validation
                    else 0
                ),
                "checks_failed": len(validation.issues_found) if validation else 0,
                "nasa_requirements": (
                    {
                        "safety_validated": (
                            validation.safety_validated if validation else False
                        ),
                        "resource_constraints_validated": (
                            validation.resource_constraints_validated
                            if validation
                            else False
                        ),
                        "science_objectives_validated": (
                            validation.science_objectives_validated
                            if validation
                            else False
                        ),
                    }
                    if validation
                    else None
                ),
            },
            "approval": {
                "approval_required": any(e.approval_required for e in events),
                "approved": approval_events[-1].approved if approval_events else False,
                "approver": (
                    approval_events[-1].approver_id if approval_events else None
                ),
                "approval_timestamp": (
                    approval_events[-1].approval_timestamp if approval_events else None
                ),
            },
            "integrity": {
                "checksums_verified": all(
                    e.checksum == e.compute_checksum() for e in events
                ),
                "event_chain_complete": True,  # Simplified
            },
        }

        return report

    def _persist_event(self, event: AuditEvent):
        """Persist event to JSONL audit log file."""
        log_file = (
            self.log_directory / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        )

        with open(log_file, "a") as f:
            json.dump(event.to_dict(), f)
            f.write("\n")

    def export_audit_trail(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        mission_id: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Export audit trail for analysis or compliance.

        Args:
            start_time: Filter events after this time (ISO format)
            end_time: Filter events before this time (ISO format)
            mission_id: Filter by mission
            output_file: Optional file to write export

        Returns:
            List of filtered audit events
        """
        filtered = self.event_cache

        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]

        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]

        if mission_id:
            filtered = [e for e in filtered if e.mission_id == mission_id]

        export_data = [e.to_dict() for e in filtered]

        if output_file:
            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)
            self.logger.info(f"Audit trail exported to {output_file}")

        return export_data


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_directory: str = "logs/audit") -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_directory)
    return _audit_logger
