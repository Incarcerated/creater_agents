"""
research_agent.py - The first agent in the pipeline.

Job:  Take a topic + config → generate 5 niche-relevant content ideas.

Input:  A topic string + config dict
Output: A dict with structured JSON like:
    {
        "topic": "...",
        "niche": "...",
        "ideas": [
            {
                "id": 1,
                "title": "...",
                "hook": "...",
                "theme": "...",
                "format": "..."
            },
            ...
        ]
    }
"""

from typing import Optional
from config import get_config, validate_config
from utils.ai_client import ai_call, parse_json_response
from utils.memory import save
from prompts.prompts import research_prompt


def run(topic: str, config: Optional[dict] = None, model: str = "gpt-4o-mini", providers: Optional[list] = None) -> dict:
    """
    Run the research agent for a given topic.

    Steps:
        1. Resolve config (use provided or fall back to default)
        2. Validate config
        3. Build the niche-aware prompt
        4. Call the AI model
        5. Parse the JSON response
        6. Save the result to memory
        7. Return the structured data

    Args:
        topic:  The content topic to research.
        config: Optional config dict. If None, uses default from config.py.
        model:  The OpenAI model to use.

    Returns:
        A dict containing the topic and 5 niche-relevant ideas.
    """

    # Step 1: Resolve config
    if config is None:
        config = get_config()

    # Step 2: Validate config
    warnings = validate_config(config)
    for w in warnings:
        print(f"⚠️  Config warning: {w}")

    print(f"\n🔍 Research Agent: Generating ideas for '{topic}' (niche: {config['niche']})...")

    # Step 3: Build the niche-aware prompt
    prompt = research_prompt(topic, config)

    # Step 4: Call the AI
    raw_response = ai_call(prompt, task_type="research", providers=providers, model=model, temperature=0.8)

    # Step 5: Parse the JSON
    data = parse_json_response(raw_response)

    # Validate that we got ideas back
    if "ideas" not in data or not isinstance(data["ideas"], list):
        raise ValueError("Research agent did not return expected 'ideas' list.")

    print(f"✅ Research Agent: Found {len(data['ideas'])} ideas!")

    # Step 6: Save to memory
    filepath = save("research", data, topic)
    print(f"💾 Saved research output → {filepath}")

    # Step 7: Return the data
    return data
