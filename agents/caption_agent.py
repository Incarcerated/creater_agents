"""
caption_agent.py - The third and final agent in the pipeline.

Job:  Take scripts + config → generate platform-specific captions + hashtags.

Input:  A list of script dicts from the script agent + config dict.
Output: A dict with structured JSON like:
    {
        "captions": [
            {
                "idea_id": 1,
                "title": "...",
                "platforms": {
                    "instagram": {"caption": "...", "hashtags": [...]},
                    "youtube": {"title": "...", "description": "...", "hashtags": [...]},
                    "facebook": {"caption": "...", "hashtags": [...]}
                },
                "tone": "..."
            },
            ...
        ]
    }
"""

from typing import Optional
from config import get_config, validate_config
from utils.ai_client import ai_call, parse_json_response
from utils.memory import save
from prompts.prompts import caption_prompt


def run(scripts: list[dict], config: Optional[dict] = None, model: str = "gpt-4o-mini", providers: Optional[list] = None) -> dict:
    """
    Run the caption agent on a list of scripts.

    Steps:
        1. Resolve config (use provided or fall back to default)
        2. Validate config
        3. Build the niche-aware prompt
        4. Call the AI model
        5. Parse the JSON response
        6. Validate platform-specific captions
        7. Save the result to memory
        8. Return the structured data

    Args:
        scripts: A list of script dicts (from script_agent.run()).
        config:  Optional config dict. If None, uses default from config.py.
        model:   The OpenAI model to use.

    Returns:
        A dict containing platform-specific captions and hashtags for each script.
    """

    # Step 1: Resolve config
    if config is None:
        config = get_config()

    # Step 2: Validate config
    warnings = validate_config(config)
    for w in warnings:
        print(f"⚠️  Config warning: {w}")

    print(f"\n📝 Caption Agent: Writing captions for {len(scripts)} scripts (niche: {config['niche']})...")

    # Step 3: Build the niche-aware prompt
    prompt = caption_prompt(scripts, config)

    # Step 4: Call the AI
    raw_response = ai_call(prompt, task_type="caption", providers=providers, model=model, temperature=0.7)

    # Step 5: Parse the JSON
    data = parse_json_response(raw_response)

    # Validate that we got captions back
    if "captions" not in data or not isinstance(data["captions"], list):
        raise ValueError("Caption agent did not return expected 'captions' list.")

    # Step 6: Validate platform-specific structure
    required_platforms = ["instagram", "youtube", "facebook"]
    for caption in data["captions"]:
        platforms = caption.get("platforms", {})
        for platform in required_platforms:
            if platform not in platforms:
                print(f"⚠️  Caption #{caption.get('idea_id', '?')} missing '{platform}' platform data")

    print(f"✅ Caption Agent: Wrote {len(data['captions'])} captions!")

    # Step 7: Save to memory
    topic_label = scripts[0].get("title", "unknown") if scripts else "unknown"
    filepath = save("caption", data, topic_label)
    print(f"💾 Saved caption output → {filepath}")

    # Step 8: Return the data
    return data
