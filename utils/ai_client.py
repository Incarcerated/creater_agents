"""
ai_client.py - Local-only AI router with Ollama and continuation strategy.

This module provides a unified `ai_call()` function that:
- Routes prompts to the right local model based on task_type
- Handles truncated responses with automatic continuation
- No external APIs — works fully offline with Ollama
- Includes robust JSON parsing with continuation triggers

Supported providers:
    - Ollama (local models: mistral, llama3, phi3, etc.)

Model routing:
    - research → mistral (fallback: phi3)
    - script   → llama3 (retry only, no fallback)
    - caption  → mistral (fallback: phi3)
    - planner  → llama3 (retry only, no fallback)
"""

import os
import json
import re
import time
import urllib.request
import urllib.error
from typing import Union, Optional, List

# --- Auto-load .env file if it exists (no python-dotenv needed) ---------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_project_root, "..", ".env")
if os.path.isfile(_env_path):
    with open(_env_path, "r") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())


# =============================================================================
# System prompts for local models
# =============================================================================

SYSTEM_PROMPT_LOCAL = (
    "You are a creative content strategist. "
    "Return ONLY valid JSON. No markdown. No explanation. No code blocks. "
    "No extra text before or after the JSON. Just the raw JSON object."
)

# JSON completion guard — appended to all prompts
JSON_COMPLETION_GUARD = (
    "\n\nIMPORTANT: Ensure the JSON is COMPLETE and valid. "
    "Do not stop mid-output. If output is long, continue until fully complete. "
    "Return ONLY JSON."
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PROVIDERS = ["ollama"]

# Timeout per Ollama request (seconds) — increased for long generations
# Set high to allow models to take time without interrupting
PROVIDER_TIMEOUT = 300

# Max retry attempts per model
MAX_RETRIES_PER_MODEL = 2

# Delay between retries (seconds) — reduces system load
RETRY_DELAY = 2

# Delay between continuation attempts (seconds) — reduces system load
CONTINUATION_DELAY = 1

# Max continuation attempts for truncated responses
MAX_CONTINUATION_ATTEMPTS = 2

# Ollama server URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# =============================================================================
# Task-type → Ollama model mapping
# =============================================================================

TASK_MODEL_MAP = {
    "research": "mistral",
    "script":   "llama3",
    "caption":  "mistral",
    "planner":  "llama3",
}

# Fallback model for non-script tasks
FALLBACK_MODEL = "phi3"


# =============================================================================
# Ollama provider implementation
# =============================================================================

def _call_ollama(prompt: str, model: str, temperature: float) -> str:
    """
    Call a local model via Ollama with increased generation limits.

    Prerequisites:
        - Ollama installed and running (https://ollama.com)
        - Model pulled: e.g., `ollama pull mistral`

    Args:
        prompt:       The user prompt.
        model:        The Ollama model name (e.g., "mistral", "llama3", "phi3").
        temperature:  Sampling temperature.

    Returns:
        The model's response as a plain string.

    Raises:
        ConnectionError: If Ollama is not running.
        ValueError:      If the model is not available.
    """

    url = f"{OLLAMA_BASE_URL}/api/generate"

    # Add JSON completion guard to prompt
    full_prompt = f"{SYSTEM_PROMPT_LOCAL}\n\n{prompt}{JSON_COMPLETION_GUARD}"

    payload = json.dumps({
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 4000,  # Increased for 2-minute lullaby with 20-30 scenes
            "num_ctx": 8192,      # Increased context window
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=PROVIDER_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Is Ollama running? Error: {e}"
        )
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(
                f"Model '{model}' not found in Ollama. "
                f"Run: ollama pull {model}"
            )
        raise

    return body.get("response", "")


# =============================================================================
# Provider registry
# =============================================================================

PROVIDER_FUNCTIONS = {
    "ollama": _call_ollama,
}


# =============================================================================
# JSON response parser with continuation trigger
# =============================================================================

def parse_json_response(raw: str, allow_continuation: bool = True) -> Union[dict, list]:
    """
    Try to extract valid JSON from the AI's response.

    If parsing fails and allow_continuation is True, returns None to trigger
    a continuation retry. Otherwise raises ValueError.

    Args:
        raw: The raw string returned by the AI.
        allow_continuation: If True, returns None on parse failure to trigger continuation.

    Returns:
        Parsed JSON as a Python dict or list, or None if continuation needed.

    Raises:
        ValueError: If no valid JSON can be extracted and continuation not allowed.
    """

    # --- Step 1: Strip markdown code-block wrappers if present -----------------
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        try:
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1:]
            cleaned = cleaned.rsplit("```", maxsplit=1)[0]
        except ValueError:
            pass  # malformed code block — fall through

    cleaned = cleaned.strip()

    # --- Step 2: Try direct json.loads ------------------------------------------
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass  # fall through to regex

    # --- Step 3: Regex fallback — find the first valid JSON object or array ----
    obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass

    arr_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', cleaned, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass

    # --- Step 4: Greedy regex — grab everything between outermost { } or [ ] ---
    greedy_obj = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if greedy_obj:
        try:
            return json.loads(greedy_obj.group(0))
        except json.JSONDecodeError:
            pass

    greedy_arr = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if greedy_arr:
        try:
            return json.loads(greedy_arr.group(0))
        except json.JSONDecodeError:
            pass

    # All parsing attempts failed
    if allow_continuation:
        return None  # Signal that continuation is needed
    else:
        raise ValueError(
            f"Could not parse AI response as JSON after all attempts.\n"
            f"Raw: {raw[:300]}"
        )


# =============================================================================
# Main router function with continuation strategy
# =============================================================================

def ai_call(
    prompt: str,
    task_type: Optional[str] = None,
    providers: Optional[List[str]] = None,
    model: str = None,
    temperature: float = 0.7,
) -> str:
    """
    Send a prompt to the AI with automatic model fallback and continuation.

    Routing logic (local-only):
        1. If task_type is set → use TASK_MODEL_MAP to pick Ollama model
        2. For script/planner: retry same model only (no phi3 fallback)
        3. For research/caption: fallback to phi3 if primary fails
        4. If JSON is truncated → retry with continuation prompt
        5. If all attempts fail → raise RuntimeError

    Args:
        prompt:       The text prompt to send.
        task_type:    The agent type ("research", "script", "caption", "planner").
        providers:    Ordered list of provider names (only "ollama" supported).
        model:        Override the model name (skips task_type routing).
        temperature:  Controls randomness (0 = deterministic, 1 = creative).

    Returns:
        The AI's response as a plain string.

    Raises:
        RuntimeError: If all model attempts fail.
    """

    if providers is None:
        providers = list(DEFAULT_PROVIDERS)

    # --- Determine the Ollama model based on task_type -------------------------
    ollama_model = FALLBACK_MODEL
    if task_type and task_type in TASK_MODEL_MAP:
        ollama_model = TASK_MODEL_MAP[task_type]
        print(f"[AI] Task '{task_type}' → using model: {ollama_model}")
    elif model is not None:
        ollama_model = model
        print(f"[AI] Using explicit model: {ollama_model}")
    elif task_type:
        print(f"[AI] Unknown task_type '{task_type}' → using fallback model: {FALLBACK_MODEL}")

    # --- Determine fallback strategy based on task type -----------------------
    # script and planner tasks should NOT fall back to phi3 (retry only)
    no_fallback_tasks = {"script", "planner"}
    use_fallback = task_type not in no_fallback_tasks

    errors = []

    for provider_name in providers:
        if provider_name not in PROVIDER_FUNCTIONS:
            print(f"[AI] Unknown provider: {provider_name} — skipping")
            errors.append(f"Unknown provider: {provider_name}")
            continue

        call_fn = PROVIDER_FUNCTIONS[provider_name]

        if provider_name == "ollama":
            # Build list of models to try
            models_to_try = [ollama_model]
            if use_fallback and ollama_model != FALLBACK_MODEL:
                models_to_try.append(FALLBACK_MODEL)

            for try_model in models_to_try:
                for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
                    print(f"[AI] {try_model} attempt {attempt}")

                    try:
                        result = call_fn(prompt, model=try_model, temperature=temperature)

                        if not result or not result.strip():
                            raise ValueError("Empty response from provider")

                        # Try to parse JSON
                        parsed = parse_json_response(result, allow_continuation=True)

                        if parsed is not None:
                            # JSON parsed successfully
                            print(f"[AI] {try_model} attempt {attempt} success")
                            print(f"[AI] Response preview: {result[:200]}...")
                            return result
                        else:
                            # JSON incomplete — try continuation
                            print(f"[AI] JSON incomplete → retrying continuation")
                            print(f"[AI] Truncated response ends with: {result[-300:]}")
                            continuation_attempts = 0
                            partial_result = result

                            while continuation_attempts < MAX_CONTINUATION_ATTEMPTS:
                                continuation_attempts += 1
                                print(f"[AI] Continuation attempt {continuation_attempts}")

                                continuation_prompt = (
                                    f"Continue the JSON from where it stopped. "
                                    f"Do not repeat. Complete all fields. "
                                    f"Return ONLY valid JSON.\n\n"
                                    f"Previous output ends with:\n{partial_result[-200:]}"
                                )

                                continuation_result = call_fn(
                                    continuation_prompt,
                                    model=try_model,
                                    temperature=temperature
                                )

                                if not continuation_result or not continuation_result.strip():
                                    raise ValueError("Empty continuation response")

                                # Merge partial + continuation
                                merged = partial_result + continuation_result
                                parsed = parse_json_response(merged, allow_continuation=True)

                                if parsed is not None:
                                    print(f"[AI] Continuation success on attempt {continuation_attempts}")
                                    return merged
                                else:
                                    partial_result = merged
                                    print(f"[AI] Continuation attempt {continuation_attempts} incomplete")
                                    time.sleep(CONTINUATION_DELAY)

                            # All continuation attempts failed
                            print(f"[AI] All continuation attempts failed for {try_model}")
                            raise ValueError("JSON incomplete after continuation attempts")

                    except Exception as e:
                        error_msg = str(e)
                        if "Cannot connect" in error_msg:
                            reason = "Ollama not running"
                        elif "not found" in error_msg:
                            reason = f"model '{try_model}' not pulled"
                        elif "empty" in error_msg.lower():
                            reason = "empty response"
                        elif "incomplete" in error_msg.lower():
                            reason = "JSON incomplete"
                        else:
                            reason = error_msg[:80]

                        print(f"[AI] {try_model} attempt {attempt} failed: {reason}")

                        if attempt < MAX_RETRIES_PER_MODEL:
                            time.sleep(RETRY_DELAY)
                        else:
                            print(f"[AI] All attempts failed for {try_model}")
                            errors.append(f"ollama/{try_model}: {reason}")

    # All providers/models failed
    error_summary = "; ".join(errors)
    raise RuntimeError(
        f"All AI providers failed. Errors: [{error_summary}]\n"
        "Check Ollama status and ensure models are pulled."
    )


# =============================================================================
# Backward-compatible alias
# =============================================================================

def call_ai(prompt: str, model: str = "mistral", temperature: float = 0.7) -> str:
    """
    Backward-compatible wrapper around ai_call().

    Uses Ollama with the specified model (default: mistral).

    Args:
        prompt:       The text prompt to send.
        model:        The Ollama model to use.
        temperature:  Controls randomness.

    Returns:
        The AI's response as a plain string.
    """

    return ai_call(prompt, providers=["ollama"], model=model, temperature=temperature)
