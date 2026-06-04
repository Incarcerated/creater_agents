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

```bash
# 1. Clone / navigate to the project
cd content_agent_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"
```

## Usage

```bash
# Run the full pipeline
python main.py "AI tools for creators"

# Use a different model
python main.py "AI tools for creators" --model gpt-4o

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
│   ├── ai_client.py        # OpenAI API wrapper + JSON parser
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
- **Model**: Pass `--model gpt-4o` for higher quality, or stick with `gpt-4o-mini` for speed/cost
- **AI Provider**: Swap `utils/ai_client.py` to use any OpenAI-compatible API

## License

MIT
