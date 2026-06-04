from typing import Optional

"""
config.py - Central configuration for the content agent system.

This config controls what niche the agents target, the tone of output,
the audience, and any content rules that must be followed.

All agents read from this config so their outputs stay consistent.
You can override the niche at runtime via: python main.py "topic" --niche "your niche"
"""

# Default configuration — edit this dict to change the system's focus
CONFIG = {
    "niche": "children lullaby songs",
    "tone": "calm, soothing, repetitive",
    "audience": "toddlers and babies",
    "content_rules": [
        "must be original",
        "no copyrighted lyrics",
        "simple vocabulary",
        "high repetition",
        "sleep-inducing tone",
    ],
    "ai_providers": ["ollama"],
    # Audio generation settings
    "audio": {
        "enabled": True,
        "sound_profile": "rain_lullaby",  # Options: "rain_lullaby", "ocean_lullaby", "wind_lullaby"
        "voice_preset": "v2/en_speaker_6",  # Bark voice preset (soft female for lullabies)
        "quality_mode": "quality",  # Options: "quality" (Bark), "fast" (macOS say)
        "voice_volume": 1.0,  # 0.0 to 1.0
        "ambient_volume": 0.15,  # 0.0 to 1.0
        "pad_volume": 0.1,  # 0.0 to 1.0
        "use_cache": True,  # Cache generated audio to avoid regeneration
    },
}


def get_config(overrides: Optional[dict] = None) -> dict:
    """
    Return the active config, optionally merged with runtime overrides.

    This lets the CLI --niche flag override the default niche without
    mutating the original CONFIG dict.

    Args:
        overrides: A dict of key-value pairs to override (e.g., {"niche": "fitness"}).

    Returns:
        A new config dict with overrides applied.
    """

    # Start with a shallow copy so we don't mutate the original
    active = dict(CONFIG)

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                active[key] = value

    return active


def validate_config(config: dict) -> list[str]:
    """
    Check that a config dict has all required fields.

    Args:
        config: The config dict to validate.

    Returns:
        A list of warning strings (empty if everything is fine).
    """

    required_keys = ["niche", "tone", "audience", "content_rules"]
    warnings = []

    for key in required_keys:
        if key not in config:
            warnings.append(f"Missing config key: '{key}'")
        elif not config[key]:
            warnings.append(f"Empty config key: '{key}'")

    return warnings
