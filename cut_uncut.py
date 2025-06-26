#!/usr/bin/env python3
"""
Generate a balanced “cut / uncut” toy dataset from one long WAV.

Input  :  audio_inputs/1750699829812.wav   (any sample-rate)
Output :  output_cut_uncut/
          ├── uncut_1.wav …  uncut_N.wav   (6-s contiguous chunks)
          └──  cut_1.wav  …   cut_N.wav   (2 × 3-s halves concatenated)

Each file is 16 kHz mono, 24-bit PCM.  
Adjust N, SEG_LEN, or SR below if desired.
"""

import random, math, os
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# --------------------------------------------------------------------------- #
# config — tweak as needed
IN_WAV   = Path("audio_inputs/1750699829812.wav")
OUT_DIR  = Path("output_cut_uncut")
SR       = 16_000            # Hz – must match your model
SEG_LEN  = 6.0               # seconds (uncut length)
N_PAIRS  = 100               # → 100 uncut + 100 cut = 200 files
RNG_SEED = 42
# --------------------------------------------------------------------------- #

random.seed(RNG_SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading {IN_WAV} …")
audio, native_sr = librosa.load(IN_WAV, sr=None, mono=True)
if native_sr != SR:
    audio = librosa.resample(audio, orig_sr=native_sr, target_sr=SR)
total_samples = len(audio)
seg_samples   = int(SEG_LEN * SR)
half_samples  = seg_samples // 2

if seg_samples > total_samples:
    raise ValueError("Source clip is shorter than the requested segment length.")

def random_snip(length):
    """Return a random slice of `audio` with `length` samples."""
    start = random.randrange(0, total_samples - length + 1)
    return audio[start:start + length]

for i in range(1, N_PAIRS + 1):
    # --- uncut --- #
    uncut = random_snip(seg_samples)
    sf.write(OUT_DIR / f"uncut_{i}.wav", uncut, SR, subtype="PCM_24")

    # --- cut (half1 + half2) --- #
    first   = random_snip(half_samples)
    second  = random_snip(half_samples)
    cut     = np.concatenate([first, second])
    sf.write(OUT_DIR / f"cut_{i}.wav", cut, SR, subtype="PCM_24")

    if i % 20 == 0 or i == N_PAIRS:
        print(f"  generated {i}/{N_PAIRS} pairs")

print(f"\nDone. Dataset saved to {OUT_DIR.resolve()}")
