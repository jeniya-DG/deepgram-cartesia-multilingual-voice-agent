# Deepgram Voice Agent + Cartesia Multilingual TTS

End-to-end demo and validation of multilingual voice agent capabilities using **Deepgram Voice Agent API (v1)** with **Cartesia sonic-multilingual TTS**.

## Stack

| Component | Provider | Model |
|-----------|----------|-------|
| STT | Deepgram | Nova-3 (`language=multi`) |
| LLM | OpenAI (managed by DG) | GPT-4o-mini |
| TTS | Cartesia | sonic-multilingual |

## Key Findings

- **`agent.language=multi` works with Cartesia TTS** — no `INVALID_SETTINGS` error, despite DG docs saying multi is only supported with ElevenLabs or OpenAI TTS.
- **Language mirroring works** — the LLM + Cartesia correctly mirrors the user's language per-turn (English, Spanish, French, Japanese all tested).
- **Strict English-only works** — the LLM prompt can force English-only responses even when the user speaks Spanish.
- **Code-switching works** — Spanish-speaking users can mix in English technical terms (e.g. "cierra el work order") and the agent responds in Spanish while understanding the English terms.
- **Per-turn switching works** — user can alternate between full English and full Spanish turns and the agent matches each time.

## Setup

```bash
# Clone
git clone https://github.com/jeniya-DG/deepgram-cartesia-multilingual-voice-agent.git
cd deepgram-cartesia-multilingual-voice-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your actual keys
```

### Getting a Cartesia Voice ID

```bash
source .env
curl -s -H "X-API-Key: $CARTESIA_API_KEY" \
     -H "Cartesia-Version: 2024-06-10" \
     "https://api.cartesia.ai/voices" | python3 -c "
import json, sys
for v in json.load(sys.stdin)[:20]:
    print(f\"{v['id']}  {v['name']}  (lang: {v.get('language','?')})\")"
```

## Usage

### Client-Facing Demo (`dg_cartesia_agent_e2e.py`)

Interactive demo with 5 pre-built scenarios:

```bash
source .env
python dg_cartesia_agent_e2e.py       # interactive menu
python dg_cartesia_agent_e2e.py 1     # run scenario 1 directly
python dg_cartesia_agent_e2e.py 5     # custom interactive conversation
```

**Scenarios:**

| # | Name | Description |
|---|------|-------------|
| 1 | Field Technician | Spanish-speaking tech mixing English terms ("cierra el work order") |
| 2 | Sales Demo | Per-turn language switching (English ↔ Spanish) |
| 3 | Strict English | Force English-only output regardless of input language |
| 4 | Language Mirror | Agent mirrors whatever language the user speaks |
| 5 | Custom | Type your own messages interactively |

### Internal Test Suite (`test_multilingual_cartesia.py`)

Automated test harness that runs multiple scenarios and generates a structured report:

```bash
source .env
python test_multilingual_cartesia.py          # run all 7 tests
python test_multilingual_cartesia.py T1       # run a specific test
python test_multilingual_cartesia.py T5 T6    # run multiple tests
```

Results are saved as JSON in `test_results/` and audio in `agent_audio_out/`.

## Configuration Notes

- The `agent.language` and `agent.listen.provider.language` fields should be set to `"multi"` for multilingual support.
- The LLM **prompt** is the primary control for output language behavior — Cartesia's TTS "reads whatever language the LLM outputs."
- Cartesia's `sonic-multilingual` model handles Spanish, French, Japanese, and more through a single voice.
- The `KeepAlive` message (`{"type": "KeepAlive"}`) should be sent every ~8 seconds when not streaming audio to prevent connection timeout.

## Files

```
├── dg_cartesia_agent_e2e.py      # Client-facing interactive demo
├── test_multilingual_cartesia.py  # Internal automated test suite
├── requirements.txt               # Python dependencies
├── .env.example                   # Template for API keys
└── README.md
```
