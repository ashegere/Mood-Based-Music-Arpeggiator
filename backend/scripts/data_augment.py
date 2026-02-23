import os
import json
import random
import argparse
import hashlib
import numpy as np
import pretty_midi
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# =========================
# CONFIG DEFAULTS
# =========================

DEFAULT_MOODS = ["melancholic", "dreamy", "energetic", "tense"]

MOOD_CONFIG = {
    "melancholic": {
        "vel_curve": "bell_low",
        "timing_style": "late",
        "register_bias": "low",
        "duration_scale": (1.1, 1.35),
        "intensity": 0.7
    },
    "dreamy": {
        "vel_curve": "down",
        "timing_style": "late",
        "register_bias": "wide",
        "duration_scale": (1.2, 1.5),
        "intensity": 0.6
    },
    "energetic": {
        "vel_curve": "up",
        "timing_style": "early",
        "register_bias": "high",
        "duration_scale": (0.8, 1.0),
        "intensity": 1.0
    },
    "tense": {
        "vel_curve": "random_walk",
        "timing_style": "alternate",
        "register_bias": "mid_high",
        "duration_scale": (0.6, 0.9),
        "intensity": 1.1
    }
}

# =========================
# VELOCITY CURVES
# =========================

def velocity_from_curve(curve, i, total):
    pos = i / max(total - 1, 1)

    if curve == "up":
        return int(70 + pos * 40)

    if curve == "down":
        return int(110 - pos * 40)

    if curve == "bell_low":
        return int(60 + 20 * np.exp(-((pos - 0.5) ** 2) / 0.08))

    if curve == "random_walk":
        return int(np.clip(90 + np.random.randn() * 15, 50, 120))

    return 80

# =========================
# REGISTER LOGIC
# =========================

def register_shift(mode):
    if mode == "low":
        return random.choice([0, -12, -12])
    if mode == "high":
        return random.choice([0, 12, 12])
    if mode == "wide":
        return random.choice([-12, 0, 12])
    if mode == "mid_high":
        return random.choice([0, 0, 12])
    return 0

# =========================
# TIMING ENGINE
# =========================

def timing_offset(style, base_jitter, i):
    if style == "late":
        return abs(base_jitter)

    if style == "early":
        return -abs(base_jitter)

    if style == "alternate":
        return base_jitter if i % 2 == 0 else -base_jitter

    return base_jitter

# =========================
# AUGMENT CORE
# =========================

def augment_midi(base_path, mood, variant, out_dir, intensity=1.0):

    pm = pretty_midi.PrettyMIDI(base_path)
    cfg = MOOD_CONFIG[mood]

    new_pm = pretty_midi.PrettyMIDI()

    for inst in pm.instruments:

        new_inst = pretty_midi.Instrument(program=inst.program)

        notes = sorted(inst.notes, key=lambda n: n.start)

        total = len(notes)

        phrase_jitter = random.uniform(0.005, 0.025) * intensity

        for i, n in enumerate(notes):

            vel = velocity_from_curve(cfg["vel_curve"], i, total)
            vel += random.randint(-5, 5)

            dur_scale = random.uniform(*cfg["duration_scale"]) * intensity

            base_jitter = random.uniform(-phrase_jitter, phrase_jitter)
            t_offset = timing_offset(cfg["timing_style"], base_jitter, i)

            shift = register_shift(cfg["register_bias"])

            start = max(0, n.start + t_offset)
            duration = (n.end - n.start) * dur_scale
            end = start + duration

            pitch = int(np.clip(n.pitch + shift, 0, 127))

            new_note = pretty_midi.Note(
                velocity=int(np.clip(vel, 1, 127)),
                pitch=pitch,
                start=start,
                end=end
            )

            new_inst.notes.append(new_note)

        new_pm.instruments.append(new_inst)

    base_name = os.path.splitext(os.path.basename(base_path))[0]
    out_name = f"{base_name}_{mood}_v{variant}.mid"
    out_path = os.path.join(out_dir, out_name)

    new_pm.write(out_path)

    return {
        "file": out_name,
        "mood": mood,
        "variant": variant,
        "intensity": intensity
    }

# =========================
# WORKER
# =========================

def worker(task):
    return augment_midi(*task)

# =========================
# MAIN
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--moods", nargs="+", default=DEFAULT_MOODS)
    parser.add_argument("--processes", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--intensity", type=float, default=1.0)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    midi_files = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith(".mid") or f.endswith(".midi")
    ]

    tasks = []

    for midi_path in midi_files:
        for mood in args.moods:
            for v in range(args.variants):
                tasks.append(
                    (midi_path, mood, v, args.output, args.intensity)
                )

    print(f"Processing {len(tasks)} augmentations...")

    with Pool(args.processes) as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))

    # Save metadata JSON
    meta_path = os.path.join(args.output, "augmentation_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")

# =========================

if __name__ == "__main__":
    main()
