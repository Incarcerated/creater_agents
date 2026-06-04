"""
memory.py - File-based memory system for storing agent outputs.

Each time the pipeline runs, the results are saved as JSON files
inside a `memory/` directory. This lets you:

- Review past results without re-running the pipeline
- Feed previous outputs into future runs
- Track how your content ideas evolve over time

Memory directory structure:
    memory/
        research/   → outputs from the research agent
        script/    → outputs from the script agent
        caption/   → outputs from the caption agent
"""

import json
import os
from typing import Union, Optional
from datetime import datetime

# Base directory for all memory files (relative to project root)
MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory")


def _ensure_dir(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def save(agent_name: str, data: Union[dict, list], topic: str) -> str:
    """
    Save an agent's output to a JSON file.

    The file is named with a timestamp so you can tell runs apart:
        memory/research/2025-04-29_15-30-00_ai_trends.json

    Args:
        agent_name: Which agent produced this data (research / script / caption).
        data:       The structured JSON data to save.
        topic:      The topic string (used in the filename for readability).

    Returns:
        The path to the saved file.
    """

    # Build the agent-specific directory
    agent_dir = os.path.join(MEMORY_DIR, agent_name)
    _ensure_dir(agent_dir)

    # Create a readable filename with timestamp and sanitized topic
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_topic = topic.replace(" ", "_").lower()[:30]  # truncate long topics
    filename = f"{timestamp}_{safe_topic}.json"
    filepath = os.path.join(agent_dir, filename)

    # Write the data with nice formatting
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def load_latest(agent_name: str) -> Optional[Union[dict, list]]:
    """
    Load the most recent output for a given agent.

    This is useful when you want to re-run only part of the pipeline
    (e.g., re-generate captions without re-doing research).

    Args:
        agent_name: Which agent's output to load.

    Returns:
        The parsed JSON data, or None if no files exist yet.
    """

    agent_dir = os.path.join(MEMORY_DIR, agent_name)
    if not os.path.isdir(agent_dir):
        return None

    # List all JSON files and pick the most recent one
    files = sorted(
        [f for f in os.listdir(agent_dir) if f.endswith(".json")],
        reverse=True,  # newest first
    )

    if not files:
        return None

    latest_path = os.path.join(agent_dir, files[0])
    with open(latest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_all(agent_name: str) -> list[str]:
    """
    List all saved memory files for a given agent.

    Args:
        agent_name: Which agent's history to list.

    Returns:
        A list of filenames (newest first).
    """

    agent_dir = os.path.join(MEMORY_DIR, agent_name)
    if not os.path.isdir(agent_dir):
        return []

    files = sorted(
        [f for f in os.listdir(agent_dir) if f.endswith(".json")],
        reverse=True,
    )
    return files
