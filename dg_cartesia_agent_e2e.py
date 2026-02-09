#!/usr/bin/env python3
"""
Deepgram Voice Agent + Cartesia Multilingual TTS — End-to-End Demo
══════════════════════════════════════════════════════════════════

Demonstrates multilingual voice agent capabilities using:
  - Deepgram Voice Agent API (v1)
  - Deepgram Nova-3 STT with language=multi
  - Cartesia sonic-multilingual TTS
  - GPT-4o-mini LLM (managed by Deepgram)

Scenarios:
  1. Field Technician — Spanish-speaking user mixing English technical terms
  2. Sales Demo       — Per-turn language switching (English ↔ Spanish)
  3. Strict English   — Force English-only output regardless of input language
  4. Language Mirror  — Agent mirrors whatever language the user speaks
  5. Custom           — Enter your own messages interactively

Setup:
  pip install websocket-client

  export DEEPGRAM_API_KEY="your-deepgram-key"
  export CARTESIA_API_KEY="your-cartesia-key"
  export CARTESIA_VOICE_ID="your-cartesia-voice-id"

  # To list available Cartesia voices:
  # curl -H "X-API-Key: $CARTESIA_API_KEY" \\
  #      -H "Cartesia-Version: 2024-06-10" \\
  #      https://api.cartesia.ai/voices

Run:
  python dg_cartesia_agent_e2e.py          # interactive menu
  python dg_cartesia_agent_e2e.py 1        # run scenario 1 directly
  python dg_cartesia_agent_e2e.py 5        # interactive custom conversation
"""

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime

import websocket

# ── Configuration ─────────────────────────────────────────────────────

DG_WS_URL = "wss://agent.deepgram.com/v1/agent/converse"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "").strip()
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "").strip()

for name, val in [("DEEPGRAM_API_KEY", DEEPGRAM_API_KEY),
                  ("CARTESIA_API_KEY", CARTESIA_API_KEY),
                  ("CARTESIA_VOICE_ID", CARTESIA_VOICE_ID)]:
    if not val:
        raise SystemExit(
            f"\n  Missing env var: {name}\n"
            f"  Set it with: export {name}=\"your-value\"\n"
        )

OUT_DIR = os.path.abspath("./agent_audio_out")
os.makedirs(OUT_DIR, exist_ok=True)

KEEPALIVE_INTERVAL = 7  # seconds


# ── Scenario Definitions ─────────────────────────────────────────────

SCENARIOS = {
    "1": {
        "name": "Field Technician",
        "subtitle": "Spanish-speaking tech mixing English terms (e.g. 'cierra el work order')",
        "agent_language": "multi",
        "listen_language": "multi",
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
            ("Spanish + English term",  "Cierra el work order número 4523."),
            ("Spanish + English term",  "El check engine light está encendido en el camión 78. ¿Qué hago?"),
            ("Pure Spanish",            "¿Cuáles son los work orders pendientes para hoy?"),
            ("Heavily mixed",           "Necesito hacer un update al dashboard con el status del delivery."),
        ],
    },
    "2": {
        "name": "Sales Demo",
        "subtitle": "Per-turn language switching — English ↔ Spanish",
        "agent_language": "multi",
        "listen_language": "multi",
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
        "greeting": "Hello! I'm your fleet assistant. How can I help?",
        "turns": [
            ("English",   "Show me the open work orders for today."),
            ("Spanish",   "Ahora dime en español, ¿cuántos camiones están disponibles?"),
            ("English",   "Switch back to English. What's the status of truck 42?"),
            ("Spanish",   "Perfecto. Ahora en español: ¿hay algún problema reportado con la flota?"),
        ],
    },
    "3": {
        "name": "Strict English",
        "subtitle": "Always respond in English, even if user speaks another language",
        "agent_language": "multi",
        "listen_language": "multi",
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
        "greeting": "Hello! How can I help you today?",
        "turns": [
            ("English",   "Hello! What can you do for me?"),
            ("Spanish",   "¿Puedes ayudarme con mi factura? Necesito entender los cargos."),
            ("English",   "Great. Can you summarize what we discussed so far?"),
        ],
    },
    "4": {
        "name": "Language Mirror",
        "subtitle": "Agent mirrors whatever language the user speaks",
        "agent_language": "multi",
        "listen_language": "multi",
        "prompt": (
            "You are a helpful multilingual assistant.\n"
            "LANGUAGE RULE:\n"
            "- Detect the language of each user message.\n"
            "- Always respond in the SAME language as the user's most recent message.\n"
            "- If the user speaks English, respond in English.\n"
            "- If the user speaks Spanish, respond in Spanish.\n"
            "- If the user speaks French, respond in French.\n"
            "- Match the language on every turn. Keep responses concise.\n"
        ),
        "greeting": "Hello! How can I help you today?",
        "turns": [
            ("English",   "Hi! What can you help me with?"),
            ("Spanish",   "¿Puedes ayudarme con mi cuenta?"),
            ("French",    "Pouvez-vous me dire quelles langues vous parlez?"),
            ("English",   "Back to English. Summarize the languages you just used."),
        ],
    },
    "5": {
        "name": "Custom Conversation",
        "subtitle": "Type your own messages — test any scenario interactively",
        "agent_language": "multi",
        "listen_language": "multi",
        "prompt": (
            "You are a helpful multilingual assistant.\n"
            "Always respond in the same language as the user's most recent message.\n"
            "Keep responses concise (1-3 sentences).\n"
        ),
        "greeting": "Hello! I can speak multiple languages. How can I help?",
        "turns": None,  # interactive mode
    },
}


# ── Demo Runner ───────────────────────────────────────────────────────

class DemoRunner:
    """Runs a single demo scenario with clean output."""

    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.audio_buffer = bytearray()
        self.audio_turn = 0
        self.conversation = []
        self.done = threading.Event()
        self._keepalive_stop = threading.Event()
        self._ws = None
        self._settings_applied = threading.Event()
        self._ready_for_input = threading.Event()
        self._errors = []

    def _save_audio(self):
        if not self.audio_buffer:
            return
        self.audio_turn += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"demo_turn{self.audio_turn}_{ts}_{uuid.uuid4().hex[:4]}.pcm"
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(self.audio_buffer)
        size = len(self.audio_buffer)
        self.audio_buffer = bytearray()
        print(f"       audio saved: {fname} ({size:,} bytes)")

    def _keepalive_loop(self):
        while not self._keepalive_stop.is_set():
            self._keepalive_stop.wait(KEEPALIVE_INTERVAL)
            if self._keepalive_stop.is_set():
                break
            try:
                self._ws.send(json.dumps({"type": "KeepAlive"}))
            except Exception:
                break

    def _build_settings(self) -> dict:
        s = self.scenario
        listen = {"type": "deepgram", "model": "nova-3"}
        if s.get("listen_language"):
            listen["language"] = s["listen_language"]

        speak = {
            "type": "cartesia",
            "model_id": "sonic-multilingual",
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
        }

        agent = {
            "language": s.get("agent_language", "multi"),
            "listen": {"provider": listen},
            "think": {
                "provider": {"type": "open_ai", "model": "gpt-4o-mini"},
                "prompt": s["prompt"],
            },
            "speak": {
                "provider": speak,
                "endpoint": {
                    "url": "https://api.cartesia.ai/tts/bytes",
                    "headers": {
                        "X-API-Key": CARTESIA_API_KEY,
                        "Cartesia-Version": "2024-06-10",
                    },
                },
            },
        }
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

    def _send_scripted_turns(self):
        turns = self.scenario["turns"]
        for i, (label, text) in enumerate(turns):
            time.sleep(4 if i == 0 else 6)
            print(f"\n  YOU [{label}]: {text}")
            self._ws.send(json.dumps({"type": "InjectUserMessage", "content": text}))
            self.conversation.append({"role": "user", "label": label, "content": text})

        time.sleep(8)
        self._keepalive_stop.set()
        self._ws.close()

    def _send_interactive_turns(self):
        """Let the user type messages interactively."""
        print("\n  Type messages below. Press Enter to send. Type 'quit' to exit.\n")
        self._ready_for_input.set()

        while not self.done.is_set():
            self._ready_for_input.wait()
            self._ready_for_input.clear()
            try:
                text = input("  YOU: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            self._ws.send(json.dumps({"type": "InjectUserMessage", "content": text}))
            self.conversation.append({"role": "user", "label": "custom", "content": text})
            time.sleep(0.5)  # small delay so response prints before next prompt

        self._keepalive_stop.set()
        self._ws.close()

    def _on_open(self, ws):
        self._ws = ws
        ws.send(json.dumps(self._build_settings()))

    def _on_message(self, ws, message):
        if isinstance(message, bytes):
            return
        try:
            event = json.loads(message)
        except Exception:
            return

        etype = event.get("type", "")

        if etype == "SettingsApplied":
            print("  Connected and configured.\n")
            self._settings_applied.set()
            # Start keepalive
            threading.Thread(target=self._keepalive_loop, daemon=True).start()
            # Start turns
            if self.scenario["turns"] is not None:
                threading.Thread(target=self._send_scripted_turns, daemon=True).start()
            else:
                threading.Thread(target=self._send_interactive_turns, daemon=True).start()

        elif etype == "ConversationText":
            role = event.get("role", "")
            content = event.get("content", "")
            self.conversation.append({"role": role, "content": content})
            if role == "assistant":
                print(f"  AGENT: {content}")

        elif etype == "AgentAudioDone":
            self._save_audio()
            if self.scenario["turns"] is None:
                self._ready_for_input.set()

        elif etype == "Error":
            code = event.get("code", "")
            desc = event.get("description", "")
            if code != "CLIENT_MESSAGE_TIMEOUT":
                self._errors.append(f"[{code}] {desc}")
                print(f"\n  ERROR: {desc}")

    def _on_data(self, ws, data, data_type, continue_flag):
        if data_type == websocket.ABNF.OPCODE_BINARY and len(data) > 0:
            self.audio_buffer.extend(data)

    def _on_error(self, ws, error):
        pass

    def _on_close(self, ws, code, reason):
        self._keepalive_stop.set()
        self._save_audio()
        self.done.set()
        self._ready_for_input.set()  # unblock interactive input

    def run(self):
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

        if self.scenario["turns"] is not None:
            timeout = 30 + len(self.scenario["turns"]) * 10
            self.done.wait(timeout=timeout)
        else:
            # Interactive — wait until user quits
            self.done.wait()

        if ws_thread.is_alive():
            self._keepalive_stop.set()
            try:
                ws.close()
            except Exception:
                pass
            ws_thread.join(timeout=3)

        return self.conversation, self._errors


# ── UI ────────────────────────────────────────────────────────────────

HEADER = """
╔══════════════════════════════════════════════════════════════════════╗
║    Deepgram Voice Agent + Cartesia Multilingual TTS Demo           ║
║    STT: Nova-3 | LLM: GPT-4o-mini | TTS: Cartesia Multilingual    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

def show_menu():
    print(HEADER)
    print("  Select a demo scenario:\n")
    for key in sorted(SCENARIOS.keys()):
        s = SCENARIOS[key]
        print(f"    {key}. {s['name']}")
        print(f"       {s['subtitle']}")
        print()
    print(f"  Audio output: {OUT_DIR}\n")


def print_summary(conversation, errors):
    print(f"\n  {'─' * 60}")
    print("  CONVERSATION SUMMARY")
    print(f"  {'─' * 60}")
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        label = msg.get("label", "")
        if role == "user":
            print(f"    YOU [{label}]: {content}")
        elif role == "assistant":
            print(f"    AGENT: {content}")
    if errors:
        print(f"\n  Errors: {errors}")
    print()


def main():
    # Allow direct scenario selection via CLI arg
    choice = None
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice not in SCENARIOS:
            print(f"  Unknown scenario: {choice}")
            print(f"  Available: {', '.join(sorted(SCENARIOS.keys()))}")
            return

    if choice is None:
        show_menu()
        try:
            choice = input("  Enter scenario number (1-5): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            return

    if choice not in SCENARIOS:
        print(f"  Unknown scenario: {choice}")
        return

    scenario = SCENARIOS[choice]
    print(f"\n  {'━' * 60}")
    print(f"  Scenario {choice}: {scenario['name']}")
    print(f"  {scenario['subtitle']}")
    print(f"  Config: agent.language={scenario['agent_language']}, "
          f"listen.language={scenario.get('listen_language', 'default')}")
    print(f"  {'━' * 60}\n")
    print("  Connecting to Deepgram Voice Agent...")

    runner = DemoRunner(scenario)
    conversation, errors = runner.run()

    print_summary(conversation, errors)
    print("  Done. Audio files saved in:", OUT_DIR)
    print()


if __name__ == "__main__":
    main()
