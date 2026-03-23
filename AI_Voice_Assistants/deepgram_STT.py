import os
import subprocess
import threading
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

ffmpeg = None
connection = None
streaming = False

def start():
    global ffmpeg, connection, streaming

    if streaming:
        print("Already running.")
        return

    print("Starting transcription...")
    streaming = True

    ready = threading.Event()

    def on_message(result):
        event = getattr(result, "event", None)
        turn_index = getattr(result, "turn_index", None)
        eot_confidence = getattr(result, "end_of_turn_confidence", None)
        if event == "StartOfTurn":
            print(f"\n--- StartOfTurn (Turn {turn_index}) ---")
        transcript = getattr(result, "transcript", None)
        if transcript:
            print(transcript)
        if event == "EndOfTurn":
            print(f"--- EndOfTurn (Turn {turn_index}, Confidence: {eot_confidence}) ---")

    def run_connection():
        global connection, ffmpeg, streaming
        with client.listen.v2.connect(
            model="flux-general-en",
            eot_threshold=0.7,
            eot_timeout_ms=5000,
            encoding="linear16",
            sample_rate=16000,
        ) as conn:
            connection = conn
            connection.on(EventType.OPEN, lambda _: ready.set())
            connection.on(EventType.MESSAGE, on_message)

            ffmpeg = subprocess.Popen([
                "ffmpeg", "-loglevel", "quiet",
                "-f", "avfoundation",
                "-i", "none:1",
                "-f", "s16le",
                "-ar", "16000",
                "-ac", "1",
                "-"
            ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            def stream():
                ready.wait()
                while streaming:
                    data = ffmpeg.stdout.read(2560)
                    if not data:
                        break
                    connection.send_media(data)

            threading.Thread(target=stream, daemon=True).start()
            print("Microphone active. Speak now.")
            connection.start_listening()  # blocks until stop() terminates ffmpeg

    threading.Thread(target=run_connection, daemon=True).start()


def stop():
    global ffmpeg, connection, streaming

    if not streaming:
        print("Not running.")
        return

    print("Stopping transcription...")
    streaming = False

    if ffmpeg:
        ffmpeg.terminate()
        ffmpeg = None

    print("Stopped.")


# ── Simple REPL ──────────────────────────────────────────────────────────────
print("Commands:  start | stop | quit")
while True:
    try:
        cmd = input("> ").strip().lower()
        if cmd == "start":
            start()
        elif cmd == "stop":
            stop()
        elif cmd in ("quit", "exit", "q"):
            stop()
            break
        else:
            print("Unknown command. Use: start | stop | quit")
    except KeyboardInterrupt:
        stop()
        break