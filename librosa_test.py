import librosa
import soundfile as sf
import numpy as np
import os
import pyworld as pw

# Example melody plan: (filename, target_pitch_Hz, duration_in_seconds)
melody = [
    ('00_02.wav', 261.63, 0.3),   # C4
    ('02_04.wav', 293.66, 0.3),   # D4
    ('04_01.wav', 329.63, 0.3),   # E4
    ('01_03.wav', 349.23, 0.3),   # F4
    ('03_00.wav', 392.00, 0.3),   # G4
    ('00_01.wav', 440.00, 0.3),   # A4
    ('01_02.wav', 493.88, 0.3),   # B4
    ('02_00.wav', 523.25, 0.3),   # C5

    # Descending
    ('00_02.wav', 493.88, 0.3),   # B4
    ('02_04.wav', 440.00, 0.3),   # A4
    ('04_01.wav', 392.00, 0.3),   # G4
    ('01_03.wav', 349.23, 0.3),   # F4
    ('03_00.wav', 329.63, 0.3),   # E4
    ('00_01.wav', 293.66, 0.3),   # D4
    ('01_02.wav', 261.63, 0.3)    # C4
]

DIPHONE_DIR = 'diphones'
OUTPUT_FILE = 'phrases/output.wav'

def get_pitch(y, sr):
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return np.nanmean(f0[voiced_flag]) if np.any(voiced_flag) else None

def pitch_shift_with_pyworld(y, sr, target_pitch):
    # Convert to double precision for PyWorld
    y = y.astype(np.float64)
    
    # Step 1: Use pyworld to extract pitch, aperiodicity, and spectrum
    f0, t = pw.dio(y, sr)  # Dio for pitch extraction
    f0 = pw.stonemask(y, f0, t, sr)  # Refine pitch estimation
    sp = pw.cheaptrick(y, f0, t, sr)  # Spectral envelope
    ap = pw.d4c(y, f0, t, sr)  # Aperiodicity
    
    # Step 2: Shift the pitch to target (in Hz)
    mean_f0 = np.mean(f0[f0 > 0])  # Only consider voiced segments
    if mean_f0 <= 0:
        mean_f0 = target_pitch  # Avoid division by zero for silent segments
    
    f0_new = f0 * (target_pitch / mean_f0)  # Scale f0 to the target pitch

    # Step 3: Re-synthesize the audio with the new pitch
    y_new = pw.synthesize(f0_new, sp, ap, sr)

    return y_new.astype(np.float32)  # Convert back to float32 for compatibility

def load_and_process(file, target_pitch, duration, sr=22050):
    path = os.path.join(DIPHONE_DIR, file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Diphone file not found: {path}")
    
    y, _ = librosa.load(path, sr=sr)
    y = pitch_shift_with_pyworld(y, sr, target_pitch)

    # Time-stretch or pad/cut to fit duration
    current_duration = librosa.get_duration(y=y, sr=sr)
    if current_duration > duration:
        y = y[:int(sr * duration)]
    else:
        padding = int(sr * duration) - len(y)
        if padding > 0:
            y = np.pad(y, (0, padding), mode='constant')

    return y

def build_phrase(melody_plan):
    sr = 22050
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    audio_sequence = []
    for file, pitch, dur in melody_plan:
        try:
            processed = load_and_process(file, pitch, dur, sr)
            audio_sequence.append(processed)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    if not audio_sequence:
        raise ValueError("No audio segments were successfully processed")
    
    output = np.concatenate(audio_sequence)
    sf.write(OUTPUT_FILE, output, sr)
    print(f"Saved phrase to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_phrase(melody)