import threading
import socket
import sounddevice as sd
import numpy as np
import wave
import queue
import time

import standing 

GO1_IP = '192.168.123.161'   
GO1_PORT = 8000              

DEVICE_INDEX = 13            
CHANNELS = 1                 
SAMPLE_RATE = 48000          
WAV_FILENAME = "mic_recording.wav"

audio_q = queue.Queue()

# --- Thread 1: Connect to the Go1 robot ---
def connect_go1():
    standing.execute_go1_movement_routine()

# --- Thread 2: Record audio, compute dB, save to file ---
def record_mic():
    def callback(indata, frames, time_info, status):
        if status:
            print("[Mic] Status:", status)
        # Convert to mono numpy array
        samples = indata[:, 0]
        # Save raw samples to queue for writing
        audio_q.put(samples.copy())

        # Compute RMS -> dB
        rms = np.sqrt(np.mean(samples**2))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -np.inf
        db_measurements.append(db)
        print(f"[Mic] dB Level: {db:.2f} dB")

    print(f"[Mic] Starting recording from device {DEVICE_INDEX}...")
    with sd.InputStream(device=DEVICE_INDEX,
                        channels=CHANNELS,
                        samplerate=SAMPLE_RATE,
                        callback=callback):
        sd.sleep(1000000)  # Record "forever"


# --- Thread 3: Write queue to .wav file ---
def save_audio():
    wf = wave.open(WAV_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(SAMPLE_RATE)

    while True:
        samples = audio_q.get()
        if samples is None:
            break
        # Convert float32 (-1.0..1.0) to int16
        int_samples = np.int16(samples * 32767)
        wf.writeframes(int_samples.tobytes())

    wf.close()
    print(f"[Mic] Saved recording to {WAV_FILENAME}")


# ------------------------------
# START THREADS
# ------------------------------
if __name__ == "__main__":
    db_measurements = []

    go1_thread = threading.Thread(target=connect_go1, daemon=True)
    mic_thread = threading.Thread(target=record_mic, daemon=True)
    audio_recording_thread = threading.Thread(target=save_audio, daemon=True)
    
    go1_thread.start()
    mic_thread.start()
    audio_recording_thread.start()

    try:
        while True:
            time.sleep(1)
            print(f"dB Level: {db_measurements[-1]:.2f} dB")

    except KeyboardInterrupt:
        audio_q.put(None)
        print("\n[Main] Exiting.")
