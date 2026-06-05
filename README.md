# Content Agent System

A semi-automated multi-agent pipeline for short-form content creation. Give it a topic, and it produces 5 viral ideas, scripts, and captions with hashtags — all structured as JSON.

## How It Works

```
Topic → Research Agent → Script Agent → Caption Agent → Final Output
          (5 ideas)       (5 scripts)    (5 captions)
```

| Agent | Input | Output |
|---|---|---|
| **Research Agent** | Topic string | 5 viral ideas (title, hook, angle, audience, format) |
| **Script Agent** | 5 ideas | 5 short-form scripts (30–60s each, with visual cues & CTA) |
| **Caption Agent** | 5 scripts | 5 captions + hashtags (8–12 per idea) |

## Setup

Complete step-by-step setup for local LLM inference using Ollama:

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama server
#    - Launch Ollama from Applications folder, or run:
open -a Ollama

# 3. Install Python client library
pip install ollama

# 4. Download models
ollama pull mistral
ollama pull llama3
ollama pull phi3

# 5. Verify models are available
ollama list

# 6. Test a model (example with phi3)
ollama run phi3 "Return JSON: { \"hello\": \"world\" }"

# 7. (Optional) Test with curl
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "Return JSON: { \"hello\": \"world\" }',
  "stream": false
}'
```

### Environment Variables

- No API key is required when using local models.
- If you wish to use OpenAI as fallback, set `OPENAI_API_KEY` in your environment.

## Usage

```bash
# Run the full pipeline
python main.py "AI tools for creators"

# Use a different model (example: phi3)
python main.py "AI tools for creators" --model phi3

# View saved history for an agent
python main.py --history research
python main.py --history script
python main.py --history caption
```

## Project Structure

```
content_agent_system/
├── main.py                  # CLI entry point — runs the full pipeline
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── agents/
│   ├── __init__.py
│   ├── research_agent.py    # Stage 1: generates 5 viral ideas
│   ├── script_agent.py      # Stage 2: writes scripts for each idea
│   └── caption_agent.py    # Stage 3: writes captions + hashtags
├── utils/
│   ├── __init__.py
│   ├── ai_client.py        # Ollama client wrapper + JSON parser
│   └── memory.py           # File-based JSON memory system
├── prompts/
│   ├── __init__.py
│   └── prompts.py           # All prompt templates in one place
└── memory/                  # Auto-created: stores agent outputs as JSON
    ├── research/
    ├── script/
    └── caption/
```

## Memory System

Every pipeline run automatically saves outputs to the `memory/` directory as timestamped JSON files. This lets you:

- **Review past results** without re-running the pipeline
- **Resume from a specific stage** by loading previous outputs
- **Track how ideas evolve** over multiple runs

## Output Format

The final combined output looks like this:

```json
{
  "topic": "AI tools for creators",
  "ideas": [
    {
      "id": 1,
      "title": "...",
      "hook": "...",
      "angle": "...",
      "audience": "...",
      "format": "..."
    }
  ],
  "scripts": [
    {
      "idea_id": 1,
      "title": "...",
      "script": "...",
      "duration_seconds": 45,
      "visual_cues": ["..."],
      "cta": "..."
    }
  ],
  "captions": [
    {
      "idea_id": 1,
      "title": "...",
      "caption": "...",
      "hashtags": ["...", "..."],
      "tone": "..."
    }
  ]
}
```

## Customization

- **Prompts**: Edit `prompts/prompts.py` to change how each agent thinks
- **Model**: Pass `--model phi3` for speed, `--model mistral` for balanced performance, or `--model llama3` for highest quality
- **AI Provider**: Swap `utils/ai_client.py` to use any OpenAI-compatible API or other providers

## License

MIT
