"""
Systematic test harness for Deepgram Voice Agent + Cartesia multilingual TTS.

Tests whether agent.language=multi works with Cartesia sonic-multilingual,
and how different LLM prompts control the language of TTS output.

Includes Samsara-specific scenarios (Spanish-speaking technicians mixing
English terms, sales demo per-turn language switching).

Env vars required:
  export DEEPGRAM_API_KEY="..."
  export CARTESIA_API_KEY="..."
  export CARTESIA_VOICE_ID="..."

Run all tests:
  python test_multilingual_cartesia.py

Run specific test(s):
  python test_multilingual_cartesia.py T1
  python test_multilingual_cartesia.py T5 T6
"""

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime

import websocket

# ── Config ────────────────────────────────────────────────────────────
DG_WS_URL = "wss://agent.deepgram.com/v1/agent/converse"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "").strip()

for name, val in [("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
                  ("CARTESIA_API_KEY", CARTESIA_API_KEY),
                  ("CARTESIA_VOICE_ID", CARTESIA_VOICE_ID)]:
    if not val:
        raise SystemExit(f"Missing {name} env var.")

OUT_DIR = os.path.abspath("./test_results")
os.makedirs(OUT_DIR, exist_ok=True)

# KeepAlive interval (docs say every 8 seconds)
KEEPALIVE_INTERVAL = 7


# ═══════════════════════════════════════════════════════════════════════
# TEST SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

# Standard 3-turn conversation: English -> Spanish -> English
STANDARD_TURNS = [
    ("English",  "Hello! What can you do for me today?"),
    ("Spanish",  "¿Puedes ayudarme con mi factura? Necesito entender los cargos."),
    ("English",  "Great, now back to English. Can you summarize what we discussed so far?"),
]

SCENARIOS = [
    # ══════════════════════════════════════════════════════════════════
    # CORE VALIDATION TESTS
    # ══════════════════════════════════════════════════════════════════

    # ── Test 1: The critical test — agent.language=multi + Cartesia ──
    {
        "id": "T1_multi_language_cartesia",
        "name": "Test 1: agent.language=multi + Cartesia sonic-multilingual",
        "description": (
            "Does DG allow agent.language=multi with Cartesia TTS? "
            "Or does it throw INVALID_SETTINGS?"
        ),
        "agent_language": "multi",
        "listen_language": "multi",
        "cartesia_language": None,  # omit to let Cartesia auto-detect
        "prompt": (
            "You are a helpful multilingual assistant.\n"
            "LANGUAGE MIRRORING RULE (STRICT):\n"
            "- Always respond in the same language as the user's MOST RECENT message.\n"
            "- If the user speaks English, respond in English.\n"
            "- If the user speaks Spanish, respond in Spanish.\n"
            "- Do NOT translate unless the user explicitly asks.\n"
        ),
        "turns": STANDARD_TURNS,
    },

    # ── Test 2: language=en but prompt asks for mirroring ──
    {
        "id": "T2_en_language_mirror_prompt",
        "name": "Test 2: agent.language=en + language-mirroring prompt (fallback)",
        "description": (
            "If multi is blocked, does language=en still allow Cartesia "
            "sonic-multilingual to speak Spanish via LLM prompt control?"
        ),
        "agent_language": "en",
        "listen_language": None,
        "cartesia_language": "en",
        "prompt": (
            "You are a helpful multilingual assistant.\n"
            "LANGUAGE MIRRORING RULE (STRICT):\n"
            "- Always respond in the same language as the user's MOST RECENT message.\n"
            "- If the user speaks English, respond in English.\n"
            "- If the user speaks Spanish, respond in Spanish.\n"
            "- Do NOT translate unless the user explicitly asks.\n"
        ),
        "turns": STANDARD_TURNS,
    },

    # ── Test 3: Strict English-only output ──
    {
        "id": "T3_strict_english_only",
        "name": "Test 3: Strict English-only — ignore Spanish input language",
        "description": (
            "Prompt: 'No matter what, even if you get Spanish as input, "
            "always respond in English. Strict requirement.' Does it obey?"
        ),
        "agent_language": "multi",
        "listen_language": "multi",
        "cartesia_language": None,
        "prompt": (
            "You are a helpful assistant.\n"
            "STRICT LANGUAGE RULE:\n"
            "- No matter what language the user speaks, you MUST ALWAYS "
            "respond in English. This is a strict, non-negotiable requirement.\n"
            "- Even if the user speaks Spanish, French, or any other language, "
            "your response must be entirely in English.\n"
            "- You may acknowledge that you understood their non-English input, "
            "but your response text must be 100% English.\n"
        ),
        "turns": STANDARD_TURNS,
    },

    # ── Test 4: Conditional mix ──
    {
        "id": "T4_conditional_mix",
        "name": "Test 4: Conditional mixed language — mix only if user initiates Spanish",
        "description": (
            "Prompt: 'If there is Spanish in the user's utterance, you may "
            "respond in a mix of English and Spanish, but never do this unless "
            "they explicitly say something in Spanish first.'"
        ),
        "agent_language": "multi",
        "listen_language": "multi",
        "cartesia_language": None,
        "prompt": (
            "You are a helpful bilingual assistant (English and Spanish).\n"
            "CONDITIONAL LANGUAGE MIXING RULE:\n"
            "- By default, respond ONLY in English.\n"
            "- If the user includes ANY Spanish in their message, you may "
            "respond in a mix of English and Spanish to be helpful.\n"
            "- NEVER use Spanish unless the user has explicitly spoken Spanish "
            "to you first in that turn.\n"
            "- Once the user switches back to English only, return to "
            "English-only responses.\n"
        ),
        "turns": STANDARD_TURNS,
    },


    # ── Test 5: Field technician — Spanish primary, English terms mixed in ──
    {
        "id": "T5_field_tech_spanish_primary",
        "name": "Test 5: SAMSARA — Spanish-speaking tech mixing English terms",
        "description": (
            "Primary prospect use case: Spanish-speaking drivers/technicians "
            "who mix English technical terms into Spanish sentences. "
            "Agent should understand the mixed input and respond in Spanish."
        ),
        "agent_language": "multi",
        "listen_language": "multi",
        "cartesia_language": None,
        "prompt": (
            "You are a helpful fleet management assistant for a trucking company.\n"
            "You help drivers and technicians manage work orders, vehicle inspections, "
            "and maintenance tasks.\n\n"
            "LANGUAGE RULE:\n"
            "- The user is a Spanish-speaking technician.\n"
            "- They may mix English technical terms (like 'work order', 'dashboard', "
            "'check engine light') into Spanish sentences.\n"
            "- Always respond in Spanish, even if the user mixes in English terms.\n"
            "- You must understand English technical terms in context.\n"
            "- Keep responses concise and action-oriented.\n"
        ),
        "greeting": "¡Hola! Soy tu asistente de gestión de flota. ¿En qué puedo ayudarte?",
        "turns": [
            ("Spanish + English term",
             "Cierra el work order número 4523."),
            ("Spanish + English term",
             "El check engine light está encendido en el camión 78. ¿Qué hago?"),
            ("Pure Spanish",
             "¿Cuáles son los work orders pendientes para hoy?"),
            ("Mixed heavily",
             "Necesito hacer un update al dashboard con el status del delivery."),
        ],
    },

    # ── Test 6: Sales demo — per-turn full language switching ──
    {
        "id": "T6_sales_demo_switching",
        "name": "Test 6: SAMSARA — Sales demo per-turn language switching",
        "description": (
            "Secondary prospect use case (sales demos): User speaks entirely "
            "in English for one turn, then entirely in Spanish the next. "
            "Agent should ideally notice the switch and match the language."
        ),
        "agent_language": "multi",
        "listen_language": "multi",
        "cartesia_language": None,
        "prompt": (
            "You are a helpful fleet management assistant.\n"
            "You help with work orders, vehicle tracking, and maintenance.\n\n"
            "LANGUAGE RULE:\n"
            "- Detect the language of each user message.\n"
            "- If the user speaks English, respond entirely in English.\n"
            "- If the user speaks Spanish, respond entirely in Spanish.\n"
            "- Match the user's language on every turn.\n"
            "- Keep responses concise (1-3 sentences).\n"
        ),
        "greeting": "Hello! I'm your fleet management assistant. How can I help? / ¡Hola! Soy tu asistente. ¿En qué puedo ayudar?",
        "turns": [
            ("English",
             "Show me the open work orders for today."),
            ("Spanish",
             "Ahora dime en español, ¿cuántos camiones están disponibles?"),
            ("English",
             "Switch back to English. What's the status of truck 42?"),
            ("Spanish",
             "Perfecto. Ahora en español: ¿hay algún problema reportado con la flota?"),
        ],
    },
    
    # ── Test 7: Edge cases ──
    {
        "id": "T7_edge_cases",
        "name": "Test 7: Edge cases — code-switching, French, Japanese, rapid switching",
        "description": (
            "Breaking/edge scenarios: mid-sentence code-switching, "
            "non-Spanish foreign languages, and rapid back-and-forth."
        ),
        "agent_language": "multi",
        "listen_language": "multi",
        "cartesia_language": None,
        "prompt": (
            "You are a helpful multilingual assistant.\n"
            "Always respond in the same language as the user's most recent message.\n"
            "Keep responses concise (1-2 sentences).\n"
        ),
        "turns": [
            ("Code-switch mid-sentence",
             "I need help with my account, pero también necesito cambiar mi dirección."),
            ("French",
             "Bonjour! Pouvez-vous m'aider avec mon compte?"),
            ("English immediately",
             "OK, English now. What languages do you support?"),
            ("Japanese",
             "日本語で話してもいいですか？"),
            ("English final",
             "That was interesting. Summarize what languages you just spoke in."),
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════

class ScenarioRunner:
    """Runs a single test scenario and collects results."""

    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.results = {
            "scenario_id": scenario["id"],
            "scenario_name": scenario["name"],
            "description": scenario["description"],
            "config": {
                "agent_language": scenario.get("agent_language"),
                "listen_language": scenario.get("listen_language"),
                "cartesia_language": scenario.get("cartesia_language"),
            },
            "settings_applied": False,
            "errors": [],
            "warnings": [],
            "conversation": [],
            "audio_files": [],
        }
        self.audio_buffer = bytearray()
        self.audio_turn = 0
        self.done_event = threading.Event()
        self._keepalive_stop = threading.Event()
        self._ws_ref = None

    def _ts(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_audio(self, label="agent"):
        if not self.audio_buffer:
            return
        self.audio_turn += 1
        sid = self.scenario["id"]
        fname = f"{sid}_turn{self.audio_turn}_{self._ts()}.pcm"
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(self.audio_buffer)
        size = len(self.audio_buffer)
        self.audio_buffer = bytearray()
        self.results["audio_files"].append({"file": fname, "size_bytes": size, "turn": self.audio_turn})

    def _keepalive_loop(self, ws):
        """Send KeepAlive every 7s to prevent CLIENT_MESSAGE_TIMEOUT."""
        while not self._keepalive_stop.is_set():
            self._keepalive_stop.wait(KEEPALIVE_INTERVAL)
            if self._keepalive_stop.is_set():
                break
            try:
                ws.send(json.dumps({"type": "KeepAlive"}))
            except Exception:
                break

    def build_settings(self) -> dict:
        s = self.scenario
        listen_provider = {
            "type": "deepgram",
            "model": "nova-3",
        }
        if s.get("listen_language"):
            listen_provider["language"] = s["listen_language"]

        speak_provider = {
            "type": "cartesia",
            "model_id": "sonic-multilingual",
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
        }
        if s.get("cartesia_language"):
            speak_provider["language"] = s["cartesia_language"]

        agent = {
            "language": s.get("agent_language", "en"),
            "listen": {"provider": listen_provider},
            "think": {
                "provider": {"type": "open_ai", "model": "gpt-4o-mini"},
                "prompt": s["prompt"],
            },
            "speak": {
                "provider": speak_provider,
                "endpoint": {
                    "url": "https://api.cartesia.ai/tts/bytes",
                    "headers": {
                        "X-API-Key": CARTESIA_API_KEY,
                        "Cartesia-Version": "2024-06-10",
                    },
                },
            },
        }

        # Optional custom greeting
        if s.get("greeting"):
            agent["greeting"] = s["greeting"]

        return {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "linear16", "sample_rate": 16000},
                "output": {"encoding": "linear16", "sample_rate": 24000, "container": "none"},
            },
            "agent": agent,
        }

    def _send_turns(self, ws):
        turns = self.scenario["turns"]
        for i, (label, text) in enumerate(turns):
            # Wait for previous response (shorter for first turn after greeting)
            wait = 3 if i == 0 else 6
            time.sleep(wait)
            print(f"    >> Turn {i+1}/{len(turns)} [{label}]: {text[:70]}...")
            ws.send(json.dumps({"type": "InjectUserMessage", "content": text}))
            self.results["conversation"].append({"role": "user", "label": label, "content": text})

        # Wait for final response
        time.sleep(8)
        self._keepalive_stop.set()
        ws.close()

    def _on_open(self, ws):
        self._ws_ref = ws
        settings = self.build_settings()
        ws.send(json.dumps(settings))
        lang = self.scenario.get("agent_language")
        listen_lang = self.scenario.get("listen_language")
        print(f"    [WS] Settings sent (agent.language={lang}, listen.language={listen_lang})")

    def _on_message(self, ws, message):
        if isinstance(message, bytes):
            return
        try:
            event = json.loads(message)
        except Exception:
            return

        etype = event.get("type", "")

        if etype == "SettingsApplied":
            self.results["settings_applied"] = True
            print("    [OK] SettingsApplied — config accepted!")
            # Start keepalive
            ka = threading.Thread(target=self._keepalive_loop, args=(ws,), daemon=True)
            ka.start()
            # Start conversation turns
            t = threading.Thread(target=self._send_turns, args=(ws,), daemon=True)
            t.start()

        elif etype == "ConversationText":
            role = event.get("role", "")
            content = event.get("content", "")
            self.results["conversation"].append({"role": role, "content": content})
            tag = "AGENT" if role == "assistant" else role.upper()
            print(f"    [{tag}]: {content}")

        elif etype == "AgentAudioDone":
            self._save_audio("agent")

        elif etype == "Error":
            desc = event.get("description", "")
            code = event.get("code", "")
            self.results["errors"].append({"code": code, "description": desc})
            if code != "CLIENT_MESSAGE_TIMEOUT":
                print(f"    [ERROR] {code}: {desc}")
            if code in ("INVALID_SETTINGS", "UNPARSABLE_CLIENT_MESSAGE"):
                self._keepalive_stop.set()
                self.done_event.set()

        elif etype == "Warning":
            desc = event.get("description", "")
            code = event.get("code", "")
            self.results["warnings"].append({"code": code, "description": desc})
            print(f"    [WARN] {code}: {desc}")

    def _on_data(self, ws, data, data_type, continue_flag):
        if data_type == websocket.ABNF.OPCODE_BINARY and len(data) > 0:
            self.audio_buffer.extend(data)

    def _on_error(self, ws, error):
        pass  # Ignore — errors are handled in on_message

    def _on_close(self, ws, code, reason):
        self._keepalive_stop.set()
        self._save_audio("final")
        self.done_event.set()

    def run(self) -> dict:
        ws = websocket.WebSocketApp(
            DG_WS_URL,
            header=[f"Authorization: Token {DEEPGRAM_API_KEY}"],
            on_open=self._on_open,
            on_message=self._on_message,
            on_data=self._on_data,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        ws_thread = threading.Thread(
            target=ws.run_forever,
            kwargs={"ping_interval": 20, "ping_timeout": 10},
            daemon=True,
        )
        ws_thread.start()

        # Generous timeout per scenario
        timeout = 20 + len(self.scenario["turns"]) * 10
        self.done_event.wait(timeout=timeout)
        if ws_thread.is_alive():
            self._keepalive_stop.set()
            try:
                ws.close()
            except Exception:
                pass
            ws_thread.join(timeout=5)

        return self.results


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

def print_report(all_results: list[dict]):
    sep = "=" * 78
    thin = "─" * 78

    print(f"\n\n{sep}")
    print("  DEEPGRAM VOICE AGENT + CARTESIA MULTILINGUAL TTS — TEST REPORT")
    print(f"{sep}\n")
    print(f"  Voice ID  : {CARTESIA_VOICE_ID}")
    print(f"  TTS Model : sonic-multilingual")
    print(f"  STT Model : nova-3")
    print(f"  LLM       : gpt-4o-mini (managed by DG)")
    print(f"  Timestamp : {datetime.now().isoformat()}")
    print()

    for r in all_results:
        print(f"\n{thin}")
        print(f"  {r['scenario_name']}")
        print(f"  Config: agent.language={r['config']['agent_language']}, "
              f"listen.language={r['config']['listen_language']}, "
              f"cartesia.language={r['config']['cartesia_language']}")
        print(f"{thin}")

        status = "ACCEPTED" if r["settings_applied"] else "REJECTED"
        print(f"  Settings: {status}")

        real_errors = [e for e in r["errors"] if e["code"] != "CLIENT_MESSAGE_TIMEOUT"]
        if real_errors:
            print("  Errors:")
            for e in real_errors:
                print(f"    [{e['code']}] {e['description']}")

        if r["warnings"]:
            print("  Warnings:")
            for w in r["warnings"]:
                print(f"    [{w.get('code','')}] {w['description']}")

        print("  Conversation:")
        for msg in r["conversation"]:
            role = msg["role"]
            content = msg["content"]
            label = msg.get("label", "")
            if role == "user":
                print(f"    USER [{label}]: {content}")
            else:
                print(f"    AGENT: {content}")

        if r["audio_files"]:
            total_bytes = sum(af["size_bytes"] for af in r["audio_files"])
            print(f"  Audio: {len(r['audio_files'])} files, {total_bytes:,} bytes total")
        else:
            print("  Audio: NONE — TTS may have failed")

        print()

    


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) > 1:
        test_ids = sys.argv[1:]
        scenarios = [s for s in SCENARIOS if any(tid in s["id"] for tid in test_ids)]
        if not scenarios:
            print(f"No matching scenarios for: {test_ids}")
            print(f"Available: {[s['id'] for s in SCENARIOS]}")
            return
    else:
        scenarios = SCENARIOS

    print(f"{'=' * 78}")
    print(f"  Deepgram Voice Agent + Cartesia Multilingual TTS — Test Suite")
    print(f"  Running {len(scenarios)} scenario(s)")
    print(f"{'=' * 78}\n")

    all_results = []

    for i, scenario in enumerate(scenarios):
        print(f"\n{'━' * 78}")
        print(f"  [{i+1}/{len(scenarios)}] {scenario['name']}")
        print(f"  {scenario['description'][:100]}")
        print(f"{'━' * 78}")

        runner = ScenarioRunner(scenario)
        result = runner.run()
        all_results.append(result)

        if i < len(scenarios) - 1:
            print("\n  ... pausing 3s ...\n")
            time.sleep(3)

    print_report(all_results)


if __name__ == "__main__":
    main()
