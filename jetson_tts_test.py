"""Simple TTS test script for Jetson (pyttsx3 + espeak fallback).

Run on the Jetson to verify TTS and Bluetooth/ALSA/PulseAudio output.
Usage:
    python3 scripts/jetson_tts_test.py
"""
import time
import subprocess
import sys

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


def speak_with_pyttsx3(text: str) -> bool:
    if pyttsx3 is None:
        print("pyttsx3 not installed or failed to import")
        return False
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        print("pyttsx3 speak failed:", e)
        return False


def speak_with_espeak(text: str) -> bool:
    # Use espeak as a robust fallback
    try:
        subprocess.run(['espeak', '-s', '140', text], check=True)
        return True
    except Exception as e:
        print("espeak fallback failed:", e)
        return False


def main():
    tests = [
        "This is a text to speech test on the Jetson.",
        "Added Coffee Nescafe.",
        "Added Coke in can.",
        "Subtotal one hundred thirty five pesos."
    ]

    print("Starting Jetson TTS smoke test\n")

    for t in tests:
        print("Announcing:", t)
        ok = speak_with_pyttsx3(t)
        if not ok:
            print("pyttsx3 failed, trying espeak fallback")
            ok = speak_with_espeak(t)
        if not ok:
            print("Both TTS methods failed for text:\n", t)
        time.sleep(1.0)

    print("TTS smoke test completed")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
