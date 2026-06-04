"""
main.py - CLI entry point for the content agent system.

Supports two modes:
    pipeline (default):  Fixed 3-stage pipeline (research → script → caption)
    agent:               AI-driven planner that decides which agent to run next

Usage:
    python main.py "your topic here"
    python main.py "your topic here" --mode agent
    python main.py "your topic here" --niche "fitness motivation"
    python main.py "your topic here" --model gpt-4o
    python main.py --history research
"""

import argparse
import json
import sys
import os
from typing import Optional

# Add the project root to Python's import path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import research_agent, script_agent, caption_agent, planner_agent
from config import get_config, validate_config
from utils.ai_client import ai_call
from utils.memory import load_latest, list_all
from audio import generate_audio_for_scripts


def run_pipeline(topic: str, config: dict, model: str = "gpt-4o-mini", providers: Optional[list] = None) -> dict:
    """
    Run the full content creation pipeline (deterministic mode).

    Pipeline flow:
        1. Research Agent  →  5 niche-relevant ideas
        2. Script Agent    →  scripts with scenes for each idea
        3. Audio Pipeline  →  voice + ambient sound (if enabled)
        4. Caption Agent   →  platform-specific captions + hashtags

    Args:
        topic:     The content topic to process.
        config:    The active config dict (niche, tone, audience, etc.).
        model:     The model to use for all agents.
        providers: Ordered list of AI providers to try.

    Returns:
        A dict combining all agent outputs.
    """

    print("=" * 60)
    print(f"🚀 Content Agent System — Pipeline Mode")
    print(f"📌 Topic: {topic}")
    print(f"🎯 Niche: {config['niche']}")
    print(f"🤖 Model: {model}")
    print(f"🔗 Providers: {providers or config.get('ai_providers', ['openai'])}")
    print("=" * 60)

    # --- Stage 1: Research ------------------------------------------------------
    research_data = research_agent.run(topic, config=config, model=model, providers=providers)
    ideas = research_data["ideas"]

    # --- Stage 2: Script Writing -----------------------------------------------
    script_data = script_agent.run(ideas, config=config, model=model, providers=providers)
    scripts = script_data["scripts"]

    # --- Stage 3: Audio Generation (if enabled) ---------------------------------
    audio_files = []
    audio_config = config.get("audio", {})
    if audio_config.get("enabled", False):
        print(f"\n🎵 Audio Pipeline: Enabled (sound_profile: {audio_config.get('sound_profile', 'rain_lullaby')}, quality_mode: {audio_config.get('quality_mode', 'quality')})")
        output_dir = os.path.join(os.path.dirname(__file__), "output", "audio")
        audio_files = generate_audio_for_scripts(
            scripts,
            output_dir,
            sound_profile=audio_config.get("sound_profile", "rain_lullaby"),
            voice_volume=audio_config.get("voice_volume", 1.0),
            ambient_volume=audio_config.get("ambient_volume", 0.15),
            pad_volume=audio_config.get("pad_volume", 0.1),
            quality_mode=audio_config.get("quality_mode", "quality"),
            voice_preset=audio_config.get("voice_preset", "v2/en_speaker_6"),
            use_cache=audio_config.get("use_cache", True)
        )
    else:
        print(f"\n🎵 Audio Pipeline: Disabled (enable in config.py)")

    # --- Stage 4: Caption + Hashtag Generation ----------------------------------
    caption_data = caption_agent.run(scripts, config=config, model=model, providers=providers)

    # --- Combine everything into one final output -------------------------------
    final_output = {
        "topic": topic,
        "niche": config["niche"],
        "ideas": ideas,
        "scripts": scripts,
        "audio_files": audio_files,
        "captions": caption_data["captions"],
    }

    print("\n" + "=" * 60)
    print("🎉 Pipeline Complete! Here's your final output:")
    print("=" * 60)
    print(json.dumps(final_output, indent=2, ensure_ascii=False))

    return final_output


def run_agent(topic: str, config: dict, model: str = "gpt-4o-mini", providers: Optional[list] = None) -> dict:
    """
    Run the planner agent mode (AI decides which agents to call).

    The planner uses a loop (max 6 steps) where the AI decides the
    next action dynamically. It calls existing agents as tools.

    Args:
        topic:  The content topic to process.
        config: The active config dict.
        model:  The OpenAI model to use.

    Returns:
        A dict with whatever the planner produced.
    """

    print("=" * 60)
    print(f"🤖 Content Agent System — Agent Mode")
    print(f"📌 Topic: {topic}")
    print(f"🎯 Niche: {config['niche']}")
    print(f"🤖 Model: {model}")
    print(f"🔗 Providers: {providers or config.get('ai_providers', ['openai'])}")
    print("=" * 60)

    final_output = planner_agent.run(topic, config=config, model=model, providers=providers)

    print("\n" + "=" * 60)
    print("🎉 Agent Mode Complete! Here's your final output:")
    print("=" * 60)
    print(json.dumps(final_output, indent=2, ensure_ascii=False))

    return final_output


def show_history(agent_name: str) -> None:
    """
    Display the saved history for a given agent.

    Args:
        agent_name: Which agent's history to show (research / script / caption / planner).
    """

    files = list_all(agent_name)
    if not files:
        print(f"No history found for '{agent_name}' agent.")
        return

    print(f"\n📂 History for '{agent_name}' agent ({len(files)} files):")
    for f in files:
        print(f"  - {f}")

    # Also show the latest output
    latest = load_latest(agent_name)
    if latest:
        print(f"\n📄 Latest output:")
        print(json.dumps(latest, indent=2, ensure_ascii=False)[:500] + "...")


def main():
    """Parse CLI arguments and run the pipeline, agent, or show history."""

    parser = argparse.ArgumentParser(
        description="Content Agent System — Semi-automated niche-focused content creation"
    )

    # Positional argument: the topic
    parser.add_argument(
        "topic",
        nargs="?",
        help='The content topic (e.g., "bedtime lullabies")',
    )

    # Optional: choose execution mode
    parser.add_argument(
        "--mode",
        choices=["pipeline", "agent"],
        default="pipeline",
        help="Execution mode: 'pipeline' (fixed 3-stage) or 'agent' (AI planner). Default: pipeline",
    )

    # Optional: override the niche from config.py
    parser.add_argument(
        "--niche",
        default=None,
        help='Override the default niche (e.g., "fitness motivation")',
    )

    # Optional: choose a different model
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini, mainly affects OpenAI)",
    )

    # Optional: override provider priority order
    parser.add_argument(
        "--providers",
        default=None,
        help='Comma-separated provider priority (e.g., "gemini,deepseek,openai")',
    )

    # Optional: view history instead of running the pipeline
    parser.add_argument(
        "--history",
        metavar="AGENT",
        choices=["research", "script", "caption", "planner"],
        help="View saved history for an agent (research / script / caption / planner)",
    )

    args = parser.parse_args()

    # --- Handle --history flag --------------------------------------------------
    if args.history:
        show_history(args.history)
        return

    # --- Validate that a topic was provided -------------------------------------
    if not args.topic:
        parser.error("Please provide a topic. Example: python main.py \"bedtime lullabies\"")

    # --- Build the active config (apply --niche override if provided) -----------
    overrides = {}
    if args.niche:
        overrides["niche"] = args.niche
    config = get_config(overrides if overrides else None)

    # --- Parse --providers into a list ------------------------------------------
    providers = None
    if args.providers:
        providers = [p.strip() for p in args.providers.split(",")]
    else:
        providers = config.get("ai_providers", None)

    # --- Validate config --------------------------------------------------------
    warnings = validate_config(config)
    for w in warnings:
        print(f"⚠️  Config warning: {w}")

    # --- Run the selected mode ---------------------------------------------------
    if args.mode == "pipeline":
        run_pipeline(args.topic, config=config, model=args.model, providers=providers)
    elif args.mode == "agent":
        run_agent(args.topic, config=config, model=args.model, providers=providers)


if __name__ == "__main__":
    main()
