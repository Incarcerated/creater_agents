"""
prompts.py - Centralized prompt templates for each agent.

Each function returns a ready-to-send prompt string.
All prompts are niche-aware — they inject config (niche, tone, audience,
content_rules) so every agent stays focused on the same target.

Keeping prompts in one place makes them easy to find, tweak, and version.
"""

from typing import Any


# =============================================================================
# Helper: format content rules as a bullet list
# =============================================================================

def _format_rules(rules: list[str]) -> str:
    """Convert a list of content rules into a bulleted string for prompts."""
    return "\n".join(f"  - {rule}" for rule in rules)


# =============================================================================
# Research Agent Prompt
# =============================================================================

def research_prompt(topic: str, config: dict) -> str:
    """
    Build the prompt for the research agent.

    The research agent takes a topic + config and returns 5 niche-relevant
    content ideas with supporting details.

    Args:
        topic:  The content topic to research.
        config: The active config dict (niche, tone, audience, content_rules).

    Returns:
        A formatted prompt string.
    """

    niche = config["niche"]
    tone = config["tone"]
    audience = config["audience"]
    rules = _format_rules(config["content_rules"])

    return f"""You are a viral content researcher specializing in the "{niche}" niche.

Given the topic below, generate exactly 5 unique, high-potential content ideas
that are specifically relevant to this niche.

Niche: {niche}
Tone: {tone}
Target Audience: {audience}

Content rules you MUST follow:
{rules}

CRITICAL COPYRIGHT SAFETY RULES:
- NEVER reuse or mimic any existing nursery rhyme or song
- DO NOT use phrases similar to known songs (e.g. "twinkle twinkle", "rock a bye baby")
- Ideas must be completely original
- Avoid common nursery rhyme patterns
- Use simple but unique wording
- Do not reference copyrighted characters or stories

For each idea, provide:
- "title": A catchy title for the idea
- "hook": An attention-grabbing opening line (first 3 seconds)
- "theme": The core theme or message of the idea
- "format": Suggested format (e.g., "lullaby", "bedtime story", "soothing animation")

Topic: "{topic}"

Respond ONLY with valid JSON in this exact format:
{{
  "topic": "{topic}",
  "niche": "{niche}",
  "ideas": [
    {{
      "id": 1,
      "title": "...",
      "hook": "...",
      "theme": "...",
      "format": "..."
    }}
  ]
}}"""


# =============================================================================
# Script Agent Prompt
# =============================================================================

def script_prompt(ideas: list[dict[str, Any]], config: dict) -> str:
    """
    Build the prompt for the script agent.

    The script agent takes research ideas + config and writes a structured
    script for each one. The output includes a "scenes" array designed for
    future video automation (AI video generation, subtitle syncing).

    Args:
        ideas:  A list of idea dicts from the research agent.
        config: The active config dict.

    Returns:
        A formatted prompt string.
    """

    niche = config["niche"]
    tone = config["tone"]
    audience = config["audience"]
    rules = _format_rules(config["content_rules"])

    # Format each idea as a compact summary for the AI
    ideas_text = ""
    for idea in ideas:
        ideas_text += (
            f'\n  Idea #{idea["id"]}: "{idea["title"]}"\n'
            f'    Hook: {idea["hook"]}\n'
            f'    Theme: {idea.get("theme", idea.get("angle", "N/A"))}\n'
            f'    Format: {idea["format"]}\n'
        )

    return f"""You are a scriptwriter specializing in the "{niche}" niche.

For each idea below, write a structured script optimized for 2-minute looping
lullaby content. The script must include scene-by-scene breakdowns for
video automation.

Niche: {niche}
Tone: {tone}
Target Audience: {audience}

Content rules you MUST follow:
{rules}

CRITICAL COPYRIGHT SAFETY RULES:
- NEVER reuse or mimic any existing nursery rhyme or song
- DO NOT use phrases similar to known songs (e.g. "twinkle twinkle", "rock a bye baby")
- Lyrics must be completely original
- Avoid common nursery rhyme patterns
- Use simple but unique wording
- Do not reference copyrighted characters or stories

DURATION REQUIREMENTS (2-MINUTE TARGET):
- Target duration: ~120 seconds
- Minimum 20-30 lines of lyrics
- Each line should be short and simple (3-8 words)
- Include repetition (important for lullabies)

LOOPABLE STRUCTURE:
- First line and last line must connect naturally for seamless looping
- Repeat key calming phrases every 4-6 lines
- Lullaby should feel continuous when replayed

SCENE GENERATION RULES:
- Total scenes: 20-30 scenes per script
- Each scene duration: 3-5 seconds
- Visuals must be: calm, repetitive, simple (moon, stars, clouds, waves, gentle animations)
- Each scene includes: text (lyric line), visual description, duration

LLAMA3 OPTIMIZATION (for local model):
- Use repetition instead of complexity
- Keep vocabulary extremely simple (toddler-level)
- Avoid long sentences
- Prioritize rhythm over meaning
- Simple rhyming patterns are okay but not required

Each script should:
- Open with the given hook
- Include original lyrics (NO copyrighted material)
- Be repetitive and calming — optimized for looping videos
- Break down into 20-30 scenes with text, visual description, and duration
- End with a gentle call-to-action

Ideas:{ideas_text}

Respond ONLY with valid JSON in this exact format:
{{
  "scripts": [
    {{
      "idea_id": 1,
      "title": "...",
      "hook": "...",
      "lyrics": "...",
      "scenes": [
        {{
          "text": "on-screen text or lyric line",
          "visual": "description of visual/animation",
          "duration": 3
        }}
      ],
      "cta": "..."
    }}
  ]
}}

IMPORTANT: Return ONLY valid JSON. Ensure output is COMPLETE. Do not truncate.
If output is long, continue until fully complete."""


# =============================================================================
# Caption Agent Prompt
# =============================================================================

def caption_prompt(scripts: list[dict[str, Any]], config: dict) -> str:
    """
    Build the prompt for the caption agent.

    The caption agent takes scripts + config and generates platform-specific
    captions with hashtags for Instagram, YouTube, and Facebook.

    Args:
        scripts: A list of script dicts from the script agent.
        config:  The active config dict.

    Returns:
        A formatted prompt string.
    """

    niche = config["niche"]
    tone = config["tone"]
    audience = config["audience"]
    rules = _format_rules(config["content_rules"])

    scripts_text = ""
    for script in scripts:
        lyrics_preview = script.get("lyrics", script.get("script", ""))[:120]
        scripts_text += (
            f'\n  Script #{script["idea_id"]}: "{script["title"]}"\n'
            f'    CTA: {script["cta"]}\n'
            f'    Lyrics excerpt: {lyrics_preview}...\n'
        )

    return f"""You are a social media caption expert specializing in the "{niche}" niche.

For each script below, write platform-specific captions + hashtags.

Niche: {niche}
Tone: {tone}
Target Audience: {audience}

Content rules you MUST follow:
{rules}

For each script, provide captions for 3 platforms:
- Instagram: emoji-rich, 1-2 sentences, 10-15 hashtags
- YouTube: descriptive, keyword-rich title + description, 5-8 hashtags
- Facebook: warm, community-oriented, 3-5 hashtags

Scripts:{scripts_text}

Respond ONLY with valid JSON in this exact format:
{{
  "captions": [
    {{
      "idea_id": 1,
      "title": "...",
      "platforms": {{
        "instagram": {{
          "caption": "...",
          "hashtags": ["...", "..."]
        }},
        "youtube": {{
          "title": "...",
          "description": "...",
          "hashtags": ["...", "..."]
        }},
        "facebook": {{
          "caption": "...",
          "hashtags": ["...", "..."]
        }}
      }},
      "tone": "..."
    }}
  ]
}}"""


# =============================================================================
# Planner Agent Prompt (for agent mode)
# =============================================================================

def planner_prompt(state_summary: str, step_number: int, max_steps: int) -> str:
    """
    Build the prompt for the planner agent (agent mode).

    The planner decides what action to take next based on the current state.
    It returns a JSON with: action, input, and reason.

    Args:
        state_summary:  A text summary of what has been done so far.
        step_number:    Current step number (for context).
        max_steps:      Maximum allowed steps (for context).

    Returns:
        A formatted prompt string.
    """

    return f"""You are a content creation planner. You decide what action to take next
to produce niche-focused content. You are on step {step_number} of {max_steps}.

Current state:
{state_summary}

Available actions:
- "generate_ideas": Generate content ideas (input: topic or niche)
- "generate_script": Write a script for ideas (input: idea IDs or "all")
- "generate_caption": Write captions for scripts (input: script IDs or "all")
- "finish": Stop and return the final output (input: summary of what was produced)

Decide the next action. Respond ONLY with valid JSON:
{{
  "action": "generate_ideas|generate_script|generate_caption|finish",
  "input": "what to pass to the action",
  "reason": "why this action makes sense now"
}}"""
