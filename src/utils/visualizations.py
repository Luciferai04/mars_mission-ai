#!/usr/bin/env python3
"""
Visualization utilities for mission planning and terrain analysis.

Generates plots for DEMs, routes, mission timelines, and hazard maps.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


def plot_dem_with_route(
    elevation: np.ndarray,
    route_path: List[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    output_path: Optional[str] = None,
) -> None:
    """Plot DEM with overlaid route.

    Args:
        elevation: 2D elevation array
        route_path: List of (row, col) coordinates
        start: Start coordinates
        goal: Goal coordinates
        output_path: Optional path to save figure
    """
    plt.figure(figsize=(12, 10))

    # Plot elevation
    im = plt.imshow(elevation, cmap="terrain", origin="lower")
    plt.colorbar(im, label="Elevation (m)")

    # Plot route
    if route_path:
        route_array = np.array(route_path)
        plt.plot(route_array[:, 1], route_array[:, 0], "r-", linewidth=2, label="Route")

    # Plot start and goal
    plt.plot(start[1], start[0], "go", markersize=10, label="Start")
    plt.plot(goal[1], goal[0], "r*", markersize=15, label="Goal")

    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("DEM with Planned Route")
    plt.legend()
    plt.grid(alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_slope_map(
    slope: np.ndarray, hazard_threshold: float = 30.0, output_path: Optional[str] = None
) -> None:
    """Plot slope map with hazard overlay.

    Args:
        slope: 2D slope array in degrees
        hazard_threshold: Slope threshold for hazards
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Slope map
    im1 = axes[0].imshow(slope, cmap="hot", origin="lower")
    plt.colorbar(im1, ax=axes[0], label="Slope (degrees)")
    axes[0].set_title("Slope Map")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")

    # Hazard map
    hazard_mask = slope > hazard_threshold
    axes[1].imshow(slope, cmap="Greys", alpha=0.3, origin="lower")
    axes[1].imshow(hazard_mask, cmap="Reds", alpha=0.6, origin="lower")
    axes[1].set_title(f"Hazard Map (>{hazard_threshold}°)")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_mission_timeline(
    activities: List[Dict[str, Any]], output_path: Optional[str] = None
) -> None:
    """Plot mission timeline as Gantt chart.

    Args:
        activities: List of activity dictionaries
        output_path: Optional path to save figure
    """
    if not activities:
        print("No activities to plot")
        return

    fig, ax = plt.subplots(figsize=(14, max(8, len(activities) * 0.5)))

    # Colors by activity type
    colors = {
        "system_checks": "skyblue",
        "drive": "orange",
        "imaging": "green",
        "analysis": "purple",
        "data_tx": "red",
        "sampling": "gold",
    }

    # Plot each activity
    for i, activity in enumerate(activities):
        start = activity.get("start_min", 0)
        duration = activity.get("duration_min", 0)
        activity_type = activity.get("type", "unknown")
        color = colors.get(activity_type, "gray")

        # Draw bar
        ax.barh(
            i,
            duration,
            left=start,
            height=0.8,
            color=color,
            alpha=0.7,
            edgecolor="black",
        )

        # Add label
        label = activity.get("id", f"Activity {i}")
        ax.text(start + duration / 2, i, label, ha="center", va="center", fontsize=8)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Activity")
    ax.set_yticks(range(len(activities)))
    ax.set_yticklabels([a.get("id", f"Act {i}") for i, a in enumerate(activities)])
    ax.set_title("Mission Timeline")
    ax.grid(axis="x", alpha=0.3)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7)
        for type, color in colors.items()
    ]
    ax.legend(legend_elements, colors.keys(), loc="upper right")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_power_profile(
    activities: List[Dict[str, Any]],
    mmrtg_power: float = 110.0,
    output_path: Optional[str] = None,
) -> None:
    """Plot power consumption profile over time.

    Args:
        activities: List of activity dictionaries
        mmrtg_power: MMRTG power output in watts
        output_path: Optional path to save figure
    """
    # Build power profile
    max_time = max(
        [a.get("start_min", 0) + a.get("duration_min", 0) for a in activities] + [0]
    )

    time_points = []
    power_points = []

    for t in range(0, int(max_time) + 1):
        total_power = 0
        for activity in activities:
            start = activity.get("start_min", 0)
            duration = activity.get("duration_min", 0)
            if start <= t < start + duration:
                total_power += activity.get("power_w", 0)

        time_points.append(t)
        power_points.append(total_power)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time_points, power_points, "b-", linewidth=2, label="Power Consumption")
    ax.axhline(
        y=mmrtg_power, color="r", linestyle="--", label=f"MMRTG Output ({mmrtg_power}W)"
    )
    ax.fill_between(time_points, 0, power_points, alpha=0.3)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Mission Power Profile")
    ax.grid(alpha=0.3)
    ax.legend()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_hazard_statistics(
    hazards: Dict[str, float], output_path: Optional[str] = None
) -> None:
    """Plot hazard statistics as pie chart.

    Args:
        hazards: Dictionary with safe/caution/hazard percentages
        output_path: Optional path to save figure
    """
    labels = ["Safe", "Caution", "Hazard"]
    sizes = [
        hazards.get("safe_percentage", 0),
        hazards.get("caution_percentage", 0),
        hazards.get("hazard_percentage", 0),
    ]
    colors = ["green", "yellow", "red"]
    explode = (0, 0.1, 0.2)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax.axis("equal")
    ax.set_title("Terrain Hazard Distribution")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_resource_summary(
    power_summary: Dict[str, float],
    time_budget: int,
    total_time: int,
    output_path: Optional[str] = None,
) -> None:
    """Plot resource utilization summary.

    Args:
        power_summary: Power budget and consumption
        time_budget: Time budget in minutes
        total_time: Total time used in minutes
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Power utilization
    power_budget = power_summary.get("budget_wh", 1000)
    power_used = power_summary.get("consumed_wh", 0)
    power_remaining = power_budget - power_used

    axes[0].bar(
        ["Used", "Remaining"], [power_used, power_remaining], color=["orange", "green"]
    )
    axes[0].set_ylabel("Energy (Wh)")
    axes[0].set_title("Power Budget Utilization")
    axes[0].axhline(y=power_budget, color="r", linestyle="--", alpha=0.5)

    # Time utilization
    time_remaining = time_budget - total_time

    axes[1].bar(
        ["Used", "Remaining"], [total_time, time_remaining], color=["blue", "lightblue"]
    )
    axes[1].set_ylabel("Time (minutes)")
    axes[1].set_title("Time Budget Utilization")
    axes[1].axhline(y=time_budget, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


# Export visualization function for API
def generate_mission_visualizations(
    plan: Dict[str, Any], output_dir: str = "./data/visualizations"
) -> List[str]:
    """Generate all mission visualizations.

    Args:
        plan: Mission plan dictionary
        output_dir: Directory to save visualizations

    Returns:
        List of generated file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    # Timeline
    activities = plan.get("activities", [])
    if activities:
        timeline_path = output_path / "timeline.png"
        plot_mission_timeline(activities, str(timeline_path))
        generated_files.append(str(timeline_path))

        # Power profile
        power_path = output_path / "power_profile.png"
        plot_power_profile(activities, output_path=str(power_path))
        generated_files.append(str(power_path))

    # Resource summary
    power_summary = plan.get("power", {})
    if power_summary:
        total_time = sum(a.get("duration_min", 0) for a in activities)
        resource_path = output_path / "resources.png"
        plot_resource_summary(power_summary, 480, total_time, str(resource_path))
        generated_files.append(str(resource_path))

    return generated_files
