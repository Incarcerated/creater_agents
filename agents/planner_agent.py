"""
planner_agent.py - An AI-driven planner that decides which agent to run next.

Unlike the fixed pipeline (research → script → caption), the planner agent
uses a loop where an AI decides the next action dynamically. This enables:

- Skipping steps if not needed
- Re-running steps with different inputs
- Adapting to the topic and niche

The planner maintains internal state and calls existing agents as "tools".
A loop limit prevents infinite execution.

Available actions:
    - generate_ideas:  Calls research_agent
    - generate_script: Calls script_agent
    - generate_caption: Calls caption_agent
    - finish:          Ends the loop and returns results
"""

from typing import Optional
from config import get_config, validate_config
from utils.ai_client import ai_call, parse_json_response
from utils.memory import save
from prompts.prompts import planner_prompt

from agents import research_agent, script_agent, caption_agent


# Maximum number of planner steps to prevent infinite loops
MAX_STEPS = 6

# Valid actions the planner can choose from
VALID_ACTIONS = {"generate_ideas", "generate_script", "generate_caption", "finish"}


def _build_state_summary(state: dict, config: dict) -> str:
    """
    Create a human-readable summary of the current planner state.

    This is fed into the planner prompt so the AI knows what has
    already been produced.

    Args:
        state:  The planner's internal state dict.
        config: The active config dict.

    Returns:
        A formatted string summarizing the current state.
    """

    lines = [
        f"Niche: {config['niche']}",
        f"Topic: {state.get('topic', 'not set')}",
        f"Ideas generated: {len(state.get('ideas', []))}",
        f"Scripts generated: {len(state.get('scripts', []))}",
        f"Captions generated: {len(state.get('captions', []))}",
    ]

    # Add details about what's been produced
    if state.get("ideas"):
        lines.append("Ideas:")
        for idea in state["ideas"]:
            lines.append(f'  #{idea["id"]}: {idea["title"]}')

    if state.get("scripts"):
        lines.append("Scripts:")
        for script in state["scripts"]:
            lines.append(f'  #{script["idea_id"]}: {script["title"]}')

    if state.get("captions"):
        lines.append("Captions:")
        for caption in state["captions"]:
            lines.append(f'  #{caption["idea_id"]}: {caption["title"]}')

    return "\n".join(lines)


def run(topic: str, config: Optional[dict] = None, model: str = "gpt-4o-mini", providers: Optional[list] = None) -> dict:
    """
    Run the planner agent loop.

    The planner repeatedly:
        1. Summarizes current state
        2. Asks the AI what to do next
        3. Executes the chosen action
        4. Updates internal state
    Until the AI says "finish" or we hit MAX_STEPS.

    Args:
        topic:  The content topic.
        config: Optional config dict. If None, uses default from config.py.
        model:  The OpenAI model to use.

    Returns:
        A dict with the final combined output from all agents that ran.
    """

    # --- Resolve and validate config ---
    if config is None:
        config = get_config()

    warnings = validate_config(config)
    for w in warnings:
        print(f"⚠️  Config warning: {w}")

    print("\n" + "=" * 60)
    print(f"🤖 Planner Agent: Starting (niche: {config['niche']})")
    print(f"📌 Topic: {topic}")
    print(f"🔄 Max steps: {MAX_STEPS}")
    print("=" * 60)

    # --- Internal state: tracks what each agent has produced ---
    state = {
        "topic": topic,
        "ideas": [],
        "scripts": [],
        "captions": [],
    }

    # --- Main planner loop ---
    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Planner Step {step}/{MAX_STEPS} ---")

        # 1. Build state summary for the planner prompt
        state_summary = _build_state_summary(state, config)

        # 2. Ask the AI what to do next
        prompt = planner_prompt(state_summary, step, MAX_STEPS)
        raw_response = ai_call(prompt, task_type="planner", providers=providers, model=model, temperature=0.3)

        # 3. Parse the planner's decision
        try:
            decision = parse_json_response(raw_response)
        except ValueError as e:
            print(f"⚠️  Planner returned invalid JSON, retrying... ({e})")
            continue

        action = decision.get("action", "").strip().lower()
        action_input = decision.get("input", "")
        reason = decision.get("reason", "no reason given")

        print(f"🎯 Action: {action}")
        print(f"   Input: {action_input}")
        print(f"   Reason: {reason}")

        # 4. Validate the action
        if action not in VALID_ACTIONS:
            print(f"⚠️  Invalid action '{action}'. Valid: {VALID_ACTIONS}")
            continue

        # 5. Execute the action
        if action == "finish":
            print("✅ Planner decided to finish!")
            break

        elif action == "generate_ideas":
            # Use the topic (or the planner's input) to generate ideas
            try:
                research_data = research_agent.run(topic, config=config, model=model, providers=providers)
                state["ideas"] = research_data.get("ideas", state["ideas"])
            except Exception as e:
                print(f"❌ Research agent failed: {e}")

        elif action == "generate_script":
            # Need ideas before we can write scripts
            if not state["ideas"]:
                print("⚠️  No ideas yet — run 'generate_ideas' first")
                continue
            try:
                script_data = script_agent.run(state["ideas"], config=config, model=model, providers=providers)
                state["scripts"] = script_data.get("scripts", state["scripts"])
            except Exception as e:
                print(f"❌ Script agent failed: {e}")

        elif action == "generate_caption":
            # Need scripts before we can write captions
            if not state["scripts"]:
                print("⚠️  No scripts yet — run 'generate_script' first")
                continue
            try:
                caption_data = caption_agent.run(state["scripts"], config=config, model=model, providers=providers)
                state["captions"] = caption_data.get("captions", state["captions"])
            except Exception as e:
                print(f"❌ Caption agent failed: {e}")

    else:
        # This runs if the loop completes without a "finish" action
        print(f"\n⚠️  Reached max steps ({MAX_STEPS}) without finishing. Returning current state.")

    # --- Build the final output ---
    final_output = {
        "topic": topic,
        "niche": config["niche"],
        "ideas": state["ideas"],
        "scripts": state["scripts"],
        "captions": state["captions"],
    }

    # Save the planner's final output
    filepath = save("planner", final_output, topic)
    print(f"\n💾 Saved planner output → {filepath}")

    return final_output
