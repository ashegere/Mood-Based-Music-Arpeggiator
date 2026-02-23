#!/usr/bin/env python3
"""
Training data generator for mood-conditioned music transformer.

This script programmatically generates paired MIDI arpeggios:
1. Neutral arpeggios (baseline, rule-based)
2. Expressive variants (mood-transformed)

The generated data can be used to train the transformer to learn
the mapping: (neutral_arpeggio, mood_embedding) -> expressive_arpeggio

Data Generation Philosophy:
--------------------------

Since we're generating synthetic training data (not using real performances),
we apply rule-based transformations that simulate how different moods
affect musical expression:

- Happy/Energetic: Higher velocity, shorter durations (staccato), slight rush
- Sad/Melancholic: Lower velocity, longer durations (legato), slight drag
- Calm/Peaceful: Medium-low velocity, smooth dynamics, steady timing
- Tense/Dramatic: High velocity variance, accents, irregular timing
- etc.

These transformations target EXPRESSION, not STRUCTURE:
- Pitch sequences remain identical
- Basic rhythm preserved
- Only velocity, duration, and micro-timing change

Output Structure:
----------------

output_dir/
├── metadata.json          # Dataset manifest with all info
├── neutral/               # Neutral MIDI files
│   ├── arp_000000.mid
│   ├── arp_000001.mid
│   └── ...
├── expressive/            # Mood-transformed MIDI files
│   ├── arp_000000_happy.mid
│   ├── arp_000000_sad.mid
│   └── ...
└── pairs.jsonl            # Training pairs (neutral_id, mood, expressive_id)

Usage:
------
    python scripts/generate_training_data.py --output-dir data/training --num-samples 1000

    # With specific configuration
    python scripts/generate_training_data.py \\
        --output-dir data/training \\
        --num-samples 5000 \\
        --moods happy sad calm tense \\
        --seed 42
"""

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.music.arpeggio_generator import (
    Note,
    Arpeggio,
    ArpeggioGenerator,
    ArpeggioPattern,
    GeneratorConfig,
    get_available_keys,
    get_available_scales,
    SCALE_PATTERNS,
)
from app.music.midi_renderer import (
    MIDIRenderer,
    save_midi,
    render_arpeggio_to_bytes,
)
from app.music.tokenization import (
    Tokenizer,
    TokenSequence,
    get_tokenizer,
)


# =============================================================================
# Mood Definitions
# =============================================================================

@dataclass
class MoodProfile:
    """
    Defines how a mood transforms musical expression.

    Each parameter controls a specific aspect of expression:
    - velocity_* : Dynamic changes
    - duration_* : Articulation changes
    - timing_* : Micro-timing/groove changes

    All multipliers are relative to neutral (1.0 = no change).
    """
    name: str
    description: str

    # Velocity transformations
    velocity_mean_mult: float = 1.0      # Multiply average velocity
    velocity_variance: float = 0.0        # Add random variance (0-1 scale)
    velocity_accent_prob: float = 0.0     # Probability of accent on beat
    velocity_accent_mult: float = 1.3     # Accent velocity multiplier

    # Duration transformations
    duration_mult: float = 1.0            # Multiply all durations
    duration_variance: float = 0.0        # Add random variance
    staccato_prob: float = 0.0            # Probability of staccato (short)
    staccato_mult: float = 0.5            # Staccato duration multiplier

    # Timing transformations (micro-timing)
    timing_variance: float = 0.0          # Random timing offset (in beats)
    timing_rush: float = 0.0              # Consistent rush (negative = drag)
    swing_amount: float = 0.0             # Swing feel (0 = straight, 1 = full)

    # Dynamics contour
    dynamics_crescendo: float = 0.0       # Gradual increase over phrase
    dynamics_decrescendo: float = 0.0     # Gradual decrease over phrase

    # Keywords for embedding (used to generate mood text)
    keywords: List[str] = field(default_factory=list)


# Define mood profiles
MOOD_PROFILES: Dict[str, MoodProfile] = {
    # === Positive/High Energy ===
    "happy": MoodProfile(
        name="happy",
        description="Bright, cheerful, uplifting expression",
        velocity_mean_mult=1.15,
        velocity_variance=0.1,
        velocity_accent_prob=0.3,
        velocity_accent_mult=1.2,
        duration_mult=0.9,
        staccato_prob=0.2,
        timing_rush=0.02,
        swing_amount=0.1,
        keywords=["happy", "joyful", "cheerful", "bright", "uplifting"],
    ),

    "energetic": MoodProfile(
        name="energetic",
        description="High energy, driving, powerful",
        velocity_mean_mult=1.25,
        velocity_variance=0.15,
        velocity_accent_prob=0.4,
        velocity_accent_mult=1.35,
        duration_mult=0.85,
        staccato_prob=0.3,
        timing_rush=0.03,
        keywords=["energetic", "powerful", "driving", "intense", "dynamic"],
    ),

    "playful": MoodProfile(
        name="playful",
        description="Light, bouncy, whimsical",
        velocity_mean_mult=1.0,
        velocity_variance=0.2,
        velocity_accent_prob=0.25,
        duration_mult=0.8,
        duration_variance=0.15,
        staccato_prob=0.4,
        staccato_mult=0.4,
        timing_variance=0.03,
        swing_amount=0.2,
        keywords=["playful", "whimsical", "bouncy", "light", "fun"],
    ),

    # === Calm/Peaceful ===
    "calm": MoodProfile(
        name="calm",
        description="Relaxed, peaceful, serene",
        velocity_mean_mult=0.85,
        velocity_variance=0.05,
        duration_mult=1.15,
        timing_variance=0.01,
        dynamics_decrescendo=0.1,
        keywords=["calm", "peaceful", "serene", "tranquil", "relaxed"],
    ),

    "dreamy": MoodProfile(
        name="dreamy",
        description="Ethereal, floating, atmospheric",
        velocity_mean_mult=0.75,
        velocity_variance=0.1,
        duration_mult=1.3,
        duration_variance=0.1,
        timing_variance=0.04,
        timing_rush=-0.02,
        keywords=["dreamy", "ethereal", "floating", "atmospheric", "hazy"],
    ),

    "meditative": MoodProfile(
        name="meditative",
        description="Still, contemplative, centered",
        velocity_mean_mult=0.7,
        velocity_variance=0.02,
        duration_mult=1.2,
        timing_variance=0.005,
        keywords=["meditative", "contemplative", "still", "centered", "mindful"],
    ),

    # === Sad/Melancholic ===
    "sad": MoodProfile(
        name="sad",
        description="Sorrowful, heavy, weighted",
        velocity_mean_mult=0.8,
        velocity_variance=0.08,
        duration_mult=1.1,
        timing_rush=-0.03,
        dynamics_decrescendo=0.15,
        keywords=["sad", "sorrowful", "melancholic", "heavy", "weighted"],
    ),

    "melancholic": MoodProfile(
        name="melancholic",
        description="Wistful, nostalgic, bittersweet",
        velocity_mean_mult=0.75,
        velocity_variance=0.12,
        duration_mult=1.15,
        duration_variance=0.08,
        timing_rush=-0.02,
        timing_variance=0.02,
        dynamics_crescendo=-0.05,
        dynamics_decrescendo=0.1,
        keywords=["melancholic", "wistful", "nostalgic", "bittersweet", "longing"],
    ),

    "somber": MoodProfile(
        name="somber",
        description="Dark, serious, grave",
        velocity_mean_mult=0.7,
        velocity_variance=0.05,
        duration_mult=1.25,
        timing_rush=-0.04,
        keywords=["somber", "dark", "serious", "grave", "solemn"],
    ),

    # === Tense/Dramatic ===
    "tense": MoodProfile(
        name="tense",
        description="Anxious, unsettled, nervous",
        velocity_mean_mult=1.0,
        velocity_variance=0.25,
        velocity_accent_prob=0.35,
        velocity_accent_mult=1.4,
        duration_mult=0.95,
        duration_variance=0.1,
        timing_variance=0.04,
        keywords=["tense", "anxious", "unsettled", "nervous", "uneasy"],
    ),

    "dramatic": MoodProfile(
        name="dramatic",
        description="Epic, powerful, cinematic",
        velocity_mean_mult=1.2,
        velocity_variance=0.2,
        velocity_accent_prob=0.5,
        velocity_accent_mult=1.5,
        duration_mult=1.0,
        duration_variance=0.15,
        timing_variance=0.02,
        dynamics_crescendo=0.2,
        keywords=["dramatic", "epic", "powerful", "cinematic", "grand"],
    ),

    "mysterious": MoodProfile(
        name="mysterious",
        description="Enigmatic, suspenseful, intriguing",
        velocity_mean_mult=0.85,
        velocity_variance=0.18,
        duration_mult=1.1,
        duration_variance=0.12,
        timing_variance=0.03,
        keywords=["mysterious", "enigmatic", "suspenseful", "intriguing", "cryptic"],
    ),

    # === Romantic/Emotional ===
    "romantic": MoodProfile(
        name="romantic",
        description="Tender, passionate, expressive",
        velocity_mean_mult=0.9,
        velocity_variance=0.15,
        duration_mult=1.1,
        duration_variance=0.08,
        timing_rush=-0.01,
        timing_variance=0.02,
        dynamics_crescendo=0.1,
        dynamics_decrescendo=0.1,
        keywords=["romantic", "tender", "passionate", "expressive", "loving"],
    ),

    "hopeful": MoodProfile(
        name="hopeful",
        description="Optimistic, upward, inspiring",
        velocity_mean_mult=1.05,
        velocity_variance=0.1,
        duration_mult=1.0,
        timing_rush=0.01,
        dynamics_crescendo=0.15,
        keywords=["hopeful", "optimistic", "inspiring", "upward", "aspirational"],
    ),

    # === Neutral (minimal transformation) ===
    "neutral": MoodProfile(
        name="neutral",
        description="Baseline, no expressive transformation",
        keywords=["neutral", "plain", "basic", "simple", "standard"],
    ),
}


# =============================================================================
# Expression Transformer
# =============================================================================

class ExpressionTransformer:
    """
    Applies mood-based transformations to arpeggios.

    This class takes a neutral arpeggio and a mood profile, then
    produces an expressive variant by modifying:
    - Velocity (dynamics)
    - Duration (articulation)
    - Timing (micro-timing/groove)

    The transformations are deterministic given a seed, ensuring
    reproducibility for training data generation.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the transformer.

        Args:
            seed: Random seed for reproducible transformations.
        """
        self.rng = random.Random(seed)

    def set_seed(self, seed: int) -> None:
        """Set the random seed."""
        self.rng.seed(seed)

    def transform(
        self,
        arpeggio: Arpeggio,
        mood: MoodProfile,
    ) -> Arpeggio:
        """
        Apply mood transformation to an arpeggio.

        Args:
            arpeggio: Neutral arpeggio to transform.
            mood: Mood profile defining the transformation.

        Returns:
            New arpeggio with expressive modifications.
        """
        if not arpeggio.notes:
            return arpeggio

        transformed_notes = []
        num_notes = len(arpeggio.notes)

        for i, note in enumerate(arpeggio.notes):
            # Calculate position in phrase (0 to 1)
            phrase_position = i / max(1, num_notes - 1)

            # Transform each attribute
            new_velocity = self._transform_velocity(
                note.velocity, i, phrase_position, mood
            )
            new_duration = self._transform_duration(
                note.duration, i, mood
            )
            new_position = self._transform_timing(
                note.position, i, mood
            )

            # Create transformed note
            transformed_notes.append(Note(
                pitch=note.pitch,  # Pitch never changes!
                velocity=new_velocity,
                duration=new_duration,
                position=new_position,
            ))

        return Arpeggio(
            notes=transformed_notes,
            key=arpeggio.key,
            scale=arpeggio.scale,
            tempo=arpeggio.tempo,
            seed=arpeggio.seed,
        )

    def _transform_velocity(
        self,
        velocity: int,
        note_index: int,
        phrase_position: float,
        mood: MoodProfile,
    ) -> int:
        """Transform velocity based on mood profile."""
        # Start with base velocity
        new_vel = float(velocity)

        # Apply mean multiplier
        new_vel *= mood.velocity_mean_mult

        # Apply variance
        if mood.velocity_variance > 0:
            variance = self.rng.gauss(0, mood.velocity_variance * 30)
            new_vel += variance

        # Apply accent on downbeats (every 4th note as proxy)
        if mood.velocity_accent_prob > 0 and note_index % 4 == 0:
            if self.rng.random() < mood.velocity_accent_prob:
                new_vel *= mood.velocity_accent_mult

        # Apply dynamics contour
        if mood.dynamics_crescendo != 0:
            new_vel *= (1.0 + mood.dynamics_crescendo * phrase_position)
        if mood.dynamics_decrescendo != 0:
            new_vel *= (1.0 - mood.dynamics_decrescendo * phrase_position)

        # Clamp to valid MIDI range
        return max(1, min(127, int(round(new_vel))))

    def _transform_duration(
        self,
        duration: float,
        note_index: int,
        mood: MoodProfile,
    ) -> float:
        """Transform duration based on mood profile."""
        new_dur = duration

        # Apply base multiplier
        new_dur *= mood.duration_mult

        # Apply variance
        if mood.duration_variance > 0:
            variance = self.rng.gauss(0, mood.duration_variance * duration)
            new_dur += variance

        # Apply staccato
        if mood.staccato_prob > 0:
            if self.rng.random() < mood.staccato_prob:
                new_dur *= mood.staccato_mult

        # Ensure minimum duration
        return max(0.0625, new_dur)  # Min 64th note

    def _transform_timing(
        self,
        position: float,
        note_index: int,
        mood: MoodProfile,
    ) -> float:
        """Transform timing based on mood profile."""
        new_pos = position

        # Apply consistent rush/drag
        new_pos += mood.timing_rush * note_index

        # Apply random variance
        if mood.timing_variance > 0:
            variance = self.rng.gauss(0, mood.timing_variance)
            new_pos += variance

        # Apply swing (affects off-beats)
        if mood.swing_amount > 0 and note_index % 2 == 1:
            # Delay off-beats for swing feel
            new_pos += mood.swing_amount * 0.1

        # Ensure non-negative
        return max(0.0, new_pos)


# =============================================================================
# Data Generator
# =============================================================================

@dataclass
class GeneratedSample:
    """Represents a generated sample with metadata."""
    sample_id: str
    neutral_path: str
    expressive_path: str
    mood: str
    mood_text: str
    key: str
    scale: str
    tempo: int
    note_count: int
    pattern: str
    seed: int
    neutral_tokens: List[int]
    expressive_tokens: List[int]


class TrainingDataGenerator:
    """
    Generates paired training data for the mood-conditioned transformer.

    Creates:
    1. Neutral arpeggios across various musical parameters
    2. Multiple expressive variants per neutral arpeggio
    3. Metadata and token sequences for training
    """

    def __init__(
        self,
        output_dir: str,
        moods: Optional[List[str]] = None,
        seed: int = 42,
    ):
        """
        Initialize the generator.

        Args:
            output_dir: Directory to save generated data.
            moods: List of mood names to generate. Uses all if None.
            seed: Master random seed.
        """
        self.output_dir = Path(output_dir)
        self.moods = moods or list(MOOD_PROFILES.keys())
        self.master_seed = seed

        # Validate moods
        for mood in self.moods:
            if mood not in MOOD_PROFILES:
                raise ValueError(f"Unknown mood: {mood}")

        # Initialize components
        self.arpeggio_generator = ArpeggioGenerator(default_seed=seed)
        self.expression_transformer = ExpressionTransformer(seed=seed)
        self.midi_renderer = MIDIRenderer(enforce_scale=True)
        self.tokenizer = get_tokenizer()

        # Random number generator
        self.rng = random.Random(seed)

        # Create output directories
        self.neutral_dir = self.output_dir / "neutral"
        self.expressive_dir = self.output_dir / "expressive"
        self.neutral_dir.mkdir(parents=True, exist_ok=True)
        self.expressive_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "neutral_count": 0,
            "expressive_count": 0,
            "total_pairs": 0,
            "moods_generated": {},
            "keys_used": {},
            "scales_used": {},
        }

    def _generate_sample_id(self, index: int) -> str:
        """Generate unique sample ID."""
        return f"arp_{index:06d}"

    def _get_random_config(self) -> GeneratorConfig:
        """Generate random arpeggio configuration."""
        # Select random parameters
        key = self.rng.choice(get_available_keys())
        scale = self.rng.choice([
            "major", "minor", "dorian", "mixolydian",
            "pentatonic_major", "pentatonic_minor"
        ])
        tempo = self.rng.choice([80, 90, 100, 110, 120, 130, 140])
        note_count = self.rng.choice([8, 12, 16, 24, 32])
        pattern = self.rng.choice(list(ArpeggioPattern))
        octave = self.rng.choice([3, 4, 5])

        return GeneratorConfig(
            key=key,
            scale=scale,
            tempo=tempo,
            note_count=note_count,
            pattern=pattern,
            octave=octave,
            seed=self.rng.randint(0, 2**31),
        )

    def _generate_mood_text(self, mood_profile: MoodProfile) -> str:
        """Generate varied mood text for embedding."""
        # Randomly combine keywords for variety
        num_keywords = self.rng.randint(1, 3)
        selected = self.rng.sample(
            mood_profile.keywords,
            min(num_keywords, len(mood_profile.keywords))
        )

        # Different text patterns
        patterns = [
            "{0}",
            "{0} and {1}",
            "{0}, {1}",
            "feeling {0}",
            "{0} mood",
            "{0} atmosphere",
            "a {0} feeling",
        ]

        if len(selected) == 1:
            patterns = [p for p in patterns if "{1}" not in p]

        pattern = self.rng.choice(patterns)

        if len(selected) >= 2:
            return pattern.format(selected[0], selected[1])
        return pattern.format(selected[0])

    def generate_sample(
        self,
        index: int,
        config: Optional[GeneratorConfig] = None,
    ) -> List[GeneratedSample]:
        """
        Generate one neutral arpeggio with all mood variants.

        Args:
            index: Sample index for ID generation.
            config: Optional specific configuration.

        Returns:
            List of GeneratedSample objects (one per mood).
        """
        # Get configuration
        if config is None:
            config = self._get_random_config()

        sample_id = self._generate_sample_id(index)

        # Generate neutral arpeggio
        neutral_arpeggio = self.arpeggio_generator.generate_from_config(config)

        # Save neutral MIDI
        neutral_filename = f"{sample_id}.mid"
        neutral_path = self.neutral_dir / neutral_filename
        neutral_midi = self.midi_renderer.render_arpeggio(neutral_arpeggio)
        save_midi(neutral_midi, str(neutral_path))

        # Tokenize neutral
        neutral_tokens = self.tokenizer.tokenize_arpeggio(neutral_arpeggio)

        # Update stats
        self.stats["neutral_count"] += 1
        self.stats["keys_used"][config.key] = self.stats["keys_used"].get(config.key, 0) + 1
        self.stats["scales_used"][config.scale] = self.stats["scales_used"].get(config.scale, 0) + 1

        # Generate expressive variants for each mood
        samples = []
        for mood_name in self.moods:
            mood_profile = MOOD_PROFILES[mood_name]

            # Set seed for reproducibility
            variant_seed = hash((sample_id, mood_name)) % (2**31)
            self.expression_transformer.set_seed(variant_seed)

            # Transform arpeggio
            expressive_arpeggio = self.expression_transformer.transform(
                neutral_arpeggio, mood_profile
            )

            # Save expressive MIDI
            expressive_filename = f"{sample_id}_{mood_name}.mid"
            expressive_path = self.expressive_dir / expressive_filename
            expressive_midi = self.midi_renderer.render_arpeggio(expressive_arpeggio)
            save_midi(expressive_midi, str(expressive_path))

            # Tokenize expressive
            expressive_tokens = self.tokenizer.tokenize_arpeggio(expressive_arpeggio)

            # Generate mood text
            mood_text = self._generate_mood_text(mood_profile)

            # Create sample record
            sample = GeneratedSample(
                sample_id=sample_id,
                neutral_path=neutral_filename,
                expressive_path=expressive_filename,
                mood=mood_name,
                mood_text=mood_text,
                key=config.key,
                scale=config.scale,
                tempo=config.tempo,
                note_count=config.note_count,
                pattern=config.pattern.value,
                seed=config.seed,
                neutral_tokens=neutral_tokens.to_ids(),
                expressive_tokens=expressive_tokens.to_ids(),
            )
            samples.append(sample)

            # Update stats
            self.stats["expressive_count"] += 1
            self.stats["total_pairs"] += 1
            self.stats["moods_generated"][mood_name] = (
                self.stats["moods_generated"].get(mood_name, 0) + 1
            )

        return samples

    def generate_dataset(
        self,
        num_samples: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[GeneratedSample]:
        """
        Generate complete dataset.

        Args:
            num_samples: Number of neutral arpeggios to generate.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            List of all generated samples.
        """
        all_samples = []

        for i in range(num_samples):
            samples = self.generate_sample(i)
            all_samples.extend(samples)

            if progress_callback:
                progress_callback(i + 1, num_samples)

        return all_samples

    def save_metadata(self, samples: List[GeneratedSample]) -> None:
        """
        Save dataset metadata and training pairs.

        Args:
            samples: List of generated samples.
        """
        # Save pairs as JSONL (one pair per line)
        pairs_path = self.output_dir / "pairs.jsonl"
        with open(pairs_path, "w") as f:
            for sample in samples:
                pair = {
                    "sample_id": sample.sample_id,
                    "neutral": sample.neutral_path,
                    "expressive": sample.expressive_path,
                    "mood": sample.mood,
                    "mood_text": sample.mood_text,
                    "key": sample.key,
                    "scale": sample.scale,
                    "tempo": sample.tempo,
                    "note_count": sample.note_count,
                    "neutral_tokens": sample.neutral_tokens,
                    "expressive_tokens": sample.expressive_tokens,
                }
                f.write(json.dumps(pair) + "\n")

        # Save overall metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "master_seed": self.master_seed,
            "moods": self.moods,
            "mood_profiles": {
                name: {
                    "description": profile.description,
                    "keywords": profile.keywords,
                }
                for name, profile in MOOD_PROFILES.items()
                if name in self.moods
            },
            "statistics": self.stats,
            "paths": {
                "neutral_dir": "neutral/",
                "expressive_dir": "expressive/",
                "pairs_file": "pairs.jsonl",
            },
            "vocabulary_size": self.tokenizer.vocab.size,
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDataset saved to: {self.output_dir}")
        print(f"  - Neutral MIDIs: {self.stats['neutral_count']}")
        print(f"  - Expressive MIDIs: {self.stats['expressive_count']}")
        print(f"  - Training pairs: {self.stats['total_pairs']}")


# =============================================================================
# CLI
# =============================================================================

def print_progress(current: int, total: int) -> None:
    """Print progress bar."""
    bar_length = 40
    progress = current / total
    filled = int(bar_length * progress)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\r[{bar}] {current}/{total} ({progress*100:.1f}%)", end="", flush=True)
    if current == total:
        print()  # Newline at end


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for mood-conditioned music transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 samples with all moods
  python scripts/generate_training_data.py --output-dir data/training --num-samples 1000

  # Generate with specific moods only
  python scripts/generate_training_data.py --output-dir data/training --num-samples 500 \\
      --moods happy sad calm energetic

  # List available moods
  python scripts/generate_training_data.py --list-moods
        """
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/training",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=100,
        help="Number of neutral arpeggios to generate",
    )
    parser.add_argument(
        "--moods", "-m",
        nargs="+",
        type=str,
        default=None,
        help="Moods to generate (default: all)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--list-moods",
        action="store_true",
        help="List available moods and exit",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # List moods if requested
    if args.list_moods:
        print("\nAvailable moods:")
        print("-" * 60)
        for name, profile in sorted(MOOD_PROFILES.items()):
            print(f"  {name:15} - {profile.description}")
            print(f"                  Keywords: {', '.join(profile.keywords[:3])}")
        print()
        return

    # Validate moods
    if args.moods:
        invalid = [m for m in args.moods if m not in MOOD_PROFILES]
        if invalid:
            print(f"Error: Unknown moods: {', '.join(invalid)}")
            print("Use --list-moods to see available options.")
            return 1

    # Create generator
    print(f"\nGenerating training data:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Moods: {args.moods or 'all'}")
    print(f"  Seed: {args.seed}")
    print()

    generator = TrainingDataGenerator(
        output_dir=args.output_dir,
        moods=args.moods,
        seed=args.seed,
    )

    # Generate dataset
    progress_fn = None if args.quiet else print_progress
    samples = generator.generate_dataset(args.num_samples, progress_fn)

    # Save metadata
    generator.save_metadata(samples)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
