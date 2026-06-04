"""
script_agent.py - Multi-pass script generation for 2-minute lullabies.

Job:  Take ideas + config → generate lyrics → generate scenes → merge into scripts

Uses multi-pass generation to avoid truncation:
1. generate_lyrics() - llama3 generates plain text lyrics
2. generate_scenes() - mistral generates scenes chunked by lyrics
3. merge_script() - combines into final script structure

Input:  A list of idea dicts from the research agent + config dict.
Output: A dict with structured JSON like:
    {
        "scripts": [
            {
                "idea_id": 1,
                "title": "...",
                "hook": "...",
                "lyrics": "...",
                "scenes": [
                    {"text": "...", "visual": "...", "duration": 3}
                ],
                "cta": "..."
            },
            ...
        ]
    }
"""

from typing import Optional
from config import get_config, validate_config
from utils.ai_client import ai_call, parse_json_response
from utils.memory import save


def generate_lyrics(idea: dict, config: dict, providers: Optional[list] = None) -> str:
    """
    Generate 2-minute lullaby lyrics (plain text, not JSON).

    Uses llama3 for creative lyric generation.

    Args:
        idea: The idea dict from research agent.
        config: The active config dict.
        providers: List of AI providers to use.

    Returns:
        Plain text lyrics (20-30 lines).
    """
    niche = config["niche"]
    tone = config["tone"]
    audience = config["audience"]

    prompt = f"""You are a lullaby lyricist specializing in the "{niche}" niche.

Generate a 2-minute lullaby for this idea:
- Title: {idea['title']}
- Hook: {idea['hook']}
- Theme: {idea.get('theme', 'N/A')}

Niche: {niche}
Tone: {tone}
Target Audience: {audience}

CRITICAL COPYRIGHT SAFETY RULES:
- NEVER reuse or mimic any existing nursery rhyme or song
- DO NOT use phrases similar to known songs (e.g. "twinkle twinkle", "rock a bye baby")
- Lyrics must be completely original
- Avoid common nursery rhyme patterns
- Use simple but unique wording

DURATION REQUIREMENTS:
- Target duration: ~120 seconds
- Minimum 20-30 lines of lyrics
- Each line should be short and simple (3-8 words)
- Include repetition (important for lullabies)

LOOPABLE STRUCTURE:
- First line and last line must connect naturally for seamless looping
- Repeat key calming phrases every 4-6 lines
- Lullaby should feel continuous when replayed

LLAMA3 OPTIMIZATION:
- Use repetition instead of complexity
- Keep vocabulary extremely simple (toddler-level)
- Avoid long sentences
- Prioritize rhythm over meaning

Return ONLY plain text lyrics. No JSON. No markdown. No formatting.
Just the lyrics, one line per line."""

    print(f"    [Script] Generating lyrics for '{idea['title']}'...")
    lyrics = ai_call(prompt, task_type="script", providers=providers, temperature=0.7)

    # Count lines
    lyrics_lines = len([line for line in lyrics.split("\n") if line.strip()])
    print(f"    [Script] Lyrics generated: {lyrics_lines} lines")

    # Retry if too short
    if lyrics_lines < 20:
        print(f"    [Script] Lyrics too short ({lyrics_lines} lines) — retrying...")
        prompt += "\n\nSTRICTER: You MUST generate at least 20 lines. Do not cut short."
        lyrics = ai_call(prompt, task_type="script", providers=providers, temperature=0.7)
        lyrics_lines = len([line for line in lyrics.split("\n") if line.strip()])
        print(f"    [Script] Retry lyrics: {lyrics_lines} lines")

    return lyrics


def generate_scenes(lyrics: str, config: dict, providers: Optional[list] = None) -> list:
    """
    Generate scenes chunked by lyrics (1-2 lines per chunk).

    Uses mistral for structured JSON output (faster than llama3 for JSON).

    Args:
        lyrics: Plain text lyrics from generate_lyrics().
        config: The active config dict.
        providers: List of AI providers to use.

    Returns:
        List of scene dicts with text, visual, duration.
    """
    # Split lyrics into chunks (1-2 lines per chunk)
    lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
    chunks = []
    for i in range(0, len(lines), 2):
        chunk = lines[i:i+2]
        chunks.append("\n".join(chunk))

    print(f"    [Script] Generating scenes for {len(chunks)} chunks...")

    scenes = []
    for i, chunk in enumerate(chunks):
        print(f"    [Script] Generating scene chunk {i+1}/{len(chunks)}...")

        prompt = f"""Convert these lyrics lines into a scene:

Lyrics:
{chunk}

Scene requirements:
- text: exact lyrics from above
- visual: calm, repetitive, simple animation (moon, stars, clouds, ocean, gentle movements)
- duration: 3-5 seconds

Return ONLY valid JSON:
{{
  "scene": {{
    "text": "exact lyrics",
    "visual": "description of calm animation",
    "duration": 3
  }}
}}

Do not include markdown. Do not include extra text."""

        try:
            # Use mistral for scenes (task_type="caption" routes to mistral)
            response = ai_call(prompt, task_type="caption", providers=providers, temperature=0.5)
            data = parse_json_response(response, allow_continuation=False)

            if "scene" in data:
                scenes.append(data["scene"])
                print(f"    [Script] Scene chunk {i+1} success")
            else:
                print(f"    [Script] Scene chunk {i+1} missing 'scene' field")
                # Fallback: create a basic scene
                scenes.append({
                    "text": chunk,
                    "visual": "calm animation",
                    "duration": 4
                })
        except Exception as e:
            print(f"    [Script] Scene chunk {i+1} failed: {e}")
            # Fallback: create a basic scene
            scenes.append({
                "text": chunk,
                "visual": "calm animation",
                "duration": 4
            })

    print(f"    [Script] Total scenes generated: {len(scenes)}")
    return scenes


def merge_script(idea: dict, lyrics: str, scenes: list) -> dict:
    """
    Merge lyrics and scenes into final script structure.

    Args:
        idea: The idea dict from research agent.
        lyrics: Plain text lyrics.
        scenes: List of scene dicts.

    Returns:
        Complete script dict with idea_id, title, hook, lyrics, scenes, cta.
    """
    # Generate a simple CTA based on the theme
    cta = f"Sleep well, little one. {idea.get('theme', 'Sweet dreams')}."

    return {
        "idea_id": idea["id"],
        "title": idea["title"],
        "hook": idea["hook"],
        "lyrics": lyrics,
        "scenes": scenes,
        "cta": cta
    }


def run(ideas: list[dict], config: Optional[dict] = None, model: str = "gpt-4o-mini", providers: Optional[list] = None) -> dict:
    """
    Run the script agent using multi-pass generation.

    Steps:
        1. Resolve config
        2. Validate config
        3. For each idea:
           a. Generate lyrics (llama3)
           b. Generate scenes (mistral, chunked)
           c. Merge into script
        4. Validate final output
        5. Save to memory
        6. Return scripts

    Args:
        ideas:  A list of idea dicts (from research_agent.run()).
        config: Optional config dict. If None, uses default from config.py.
        model:  The model to use (ignored in multi-pass, uses task_type routing).
        providers: List of AI providers to use.

    Returns:
        A dict containing scripts with scenes for each idea.
    """

    # Step 1: Resolve config
    if config is None:
        config = get_config()

    # Step 2: Validate config
    warnings = validate_config(config)
    for w in warnings:
        print(f"⚠️  Config warning: {w}")

    print(f"\n✍️  Script Agent: Writing scripts for {len(ideas)} ideas (niche: {config['niche']})...")

    # Step 3: Process each idea with multi-pass generation
    all_scripts = []

    for idea in ideas:
        print(f"\n  → Processing idea #{idea['id']}: {idea['title']}")

        # Step 3a: Generate lyrics (llama3)
        lyrics = generate_lyrics(idea, config, providers)

        # Step 3b: Generate scenes (mistral, chunked)
        scenes = generate_scenes(lyrics, config, providers)

        # Step 3c: Merge into final script
        script = merge_script(idea, lyrics, scenes)
        all_scripts.append(script)

        print(f"    ✅ Script for idea #{idea['id']} complete ({len(scenes)} scenes)")

    # Step 4: Validate final output
    print(f"\n  → Validating final output...")
    for script in all_scripts:
        lyrics_lines = len([line for line in script["lyrics"].split("\n") if line.strip()])
        scene_count = len(script["scenes"])

        if lyrics_lines < 20:
            print(f"    ⚠️  Script #{script['idea_id']} has only {lyrics_lines} lyrics lines")
        if scene_count < 15:
            print(f"    ⚠️  Script #{script['idea_id']} has only {scene_count} scenes")

    # Step 5: Save to memory
    final_data = {"scripts": all_scripts}
    print(f"\n✅ Script Agent: Wrote {len(final_data['scripts'])} scripts!")

    topic_label = ideas[0].get("title", "unknown") if ideas else "unknown"
    filepath = save("script", final_data, topic_label)
    print(f"💾 Saved script output → {filepath}")

    # Step 6: Return the data
    return final_data
