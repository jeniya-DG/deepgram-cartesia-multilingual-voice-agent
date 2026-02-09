# Deepgram Voice Agent + Cartesia Multilingual TTS

End-to-end demo and validation of multilingual voice agent capabilities using **Deepgram Voice Agent API (v1)** with **Cartesia sonic-multilingual TTS**.

## Stack

| Component | Provider | Model |
|-----------|----------|-------|
| STT | Deepgram | Nova-3 (`language=multi`) |
| LLM | OpenAI (managed by DG) | GPT-4o-mini |
| TTS | Cartesia | sonic-multilingual |



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

Automated test harness that runs 7 scenarios and generates a structured report:

```bash
source .env
python test_multilingual_cartesia.py          # run all 7 tests
python test_multilingual_cartesia.py T1       # run a specific test
python test_multilingual_cartesia.py T5 T6    # run multiple tests
```

Results are saved as JSON in `test_results/` and audio in `agent_audio_out/`.

## Test Scenarios & Results

### Core Validation Tests (T1–T4)

| Test | Scenario | Config | What It Validates | Result |
|------|----------|--------|-------------------|--------|
| **T1** | `agent.language=multi` + Cartesia `sonic-multilingual` | `agent.language=multi`, `listen.language=multi` | Does DG accept `language=multi` with Cartesia TTS without throwing `INVALID_SETTINGS`? | **PASS** — Settings accepted. Agent correctly switched EN → ES → EN |
| **T2** | `agent.language=en` + language-mirroring prompt (fallback) | `agent.language=en`, `cartesia.language=en` | If `multi` were blocked, can the LLM prompt still drive Spanish output through Cartesia? | **PASS** — Mirrored Spanish despite `language=en` config. Minor slow-speak warning |
| **T3** | Strict English-only output | `agent.language=multi`, `listen.language=multi` | Can an LLM prompt force English-only responses even when user speaks Spanish? | **PASS** — Responded in English to Spanish input. Prompt override works |
| **T4** | Conditional mixed language | `agent.language=multi`, `listen.language=multi` | Can the agent mix EN/ES only when the user initiates Spanish, and revert to EN-only otherwise? | **PASS** — Mixed only when user spoke Spanish; reverted to English cleanly |

### Samsara-Specific Tests (T5–T6)

| Test | Scenario | Use Case | What It Validates | Result |
|------|----------|----------|-------------------|--------|
| **T5** | Spanish-speaking tech mixing English terms | Field technician says "Cierra el work order", "check engine light está encendido" | Does the agent understand mixed ES/EN input and always respond in Spanish? | **PASS** — Understood English technical terms in Spanish sentences, responded entirely in Spanish |
| **T6** | Sales demo per-turn language switching | User alternates full turns: EN → ES → EN → ES | Does the agent cleanly switch output language to match each turn? | **PASS** — Matched language on every turn with clean switching |

### Edge Case Test (T7)

| Test | Scenario | What It Validates | Result |
|------|----------|-------------------|--------|
| **T7** | Code-switching, French, Japanese, rapid switching | Mid-sentence EN/ES code-switch, French input, Japanese input, rapid back-to-English | **PASS** — Handled all cases: responded in ES for code-switched input, French for French, Japanese for Japanese (はい、日本語でお話ししても大丈夫です), and English on demand |

