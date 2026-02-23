#!/usr/bin/env python3
"""
MIDI Dataset Validator

Validates a folder of MIDI files against a JSON metadata mapping and generates
a comprehensive dataset report.

Validation Checks:
- MIDI loads successfully
- At least 1 instrument track
- Note count > 8
- Duration < 30 seconds
- No invalid pitch values (0-127)
- Mood label exists and is valid

Output:
- dataset_report.json with valid/invalid files, mood distribution, statistics

Usage:
    python scripts/validate_dataset.py --midi-dir data/training/augmented \
        --metadata data/training/arpeggio_mood_data_updated.json \
        --output data/training/dataset_report.json

    # With custom valid moods
    python scripts/validate_dataset.py --midi-dir data/midis \
        --metadata data/metadata.json \
        --valid-moods happy sad calm energetic tense

    # Verbose output
    python scripts/validate_dataset.py --midi-dir data/midis \
        --metadata data/metadata.json --verbose
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import pretty_midi
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

# Default valid mood labels
DEFAULT_VALID_MOODS: Set[str] = {
    # Positive/High Energy
    "happy", "joyful", "euphoric", "excited", "energetic",
    "uplifting", "triumphant", "playful", "cheerful", "bright",
    # Calm/Peaceful
    "calm", "peaceful", "serene", "tranquil", "relaxed",
    "gentle", "soothing", "meditative", "dreamy", "floating",
    # Sad/Melancholic
    "sad", "melancholic", "sorrowful", "wistful", "nostalgic",
    "longing", "bittersweet", "mournful", "pensive", "reflective",
    # Dark/Tense
    "dark", "mysterious", "ominous", "tense", "suspenseful",
    "brooding", "intense", "dramatic", "epic", "powerful",
    # Romantic/Emotional
    "romantic", "passionate", "tender", "intimate", "emotional",
    "heartfelt", "loving", "warm", "hopeful", "inspiring",
    # Neutral/Ambient
    "neutral", "ambient", "atmospheric", "minimal", "sparse",
    "subtle", "understated", "steady", "flowing", "continuous",
}

# Validation thresholds
MIN_NOTE_COUNT = 8
MAX_DURATION_SECONDS = 30.0
MIN_PITCH = 0
MAX_PITCH = 127


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ValidationError:
    """Represents a single validation error."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class FileValidationResult:
    """Result of validating a single MIDI file."""
    filename: str
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    # Metadata (populated if file loads successfully)
    mood: Optional[str] = None
    note_count: Optional[int] = None
    duration: Optional[float] = None
    instrument_count: Optional[int] = None
    pitch_range: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "filename": self.filename,
            "is_valid": self.is_valid,
            "errors": [asdict(e) for e in self.errors],
            "warnings": [asdict(e) for e in self.warnings],
        }

        if self.mood is not None:
            result["mood"] = self.mood
        if self.note_count is not None:
            result["note_count"] = self.note_count
        if self.duration is not None:
            result["duration"] = round(self.duration, 3)
        if self.instrument_count is not None:
            result["instrument_count"] = self.instrument_count
        if self.pitch_range is not None:
            result["pitch_range"] = list(self.pitch_range)

        return result


@dataclass
class DatasetReport:
    """Complete dataset validation report."""
    generated_at: str
    midi_directory: str
    metadata_file: str

    # Counts
    total_files_in_metadata: int
    total_files_found: int
    total_files_missing: int
    valid_files_count: int
    invalid_files_count: int

    # File lists
    valid_files: List[Dict]
    invalid_files: List[Dict]
    missing_files: List[str]

    # Statistics
    mood_distribution: Dict[str, int]
    error_distribution: Dict[str, int]

    # Averages (for valid files only)
    average_note_count: Optional[float]
    average_duration: Optional[float]
    average_instrument_count: Optional[float]

    # Ranges
    note_count_range: Optional[Tuple[int, int]]
    duration_range: Optional[Tuple[float, float]]
    pitch_range: Optional[Tuple[int, int]]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "generated_at": self.generated_at,
            "midi_directory": self.midi_directory,
            "metadata_file": self.metadata_file,
            "summary": {
                "total_files_in_metadata": self.total_files_in_metadata,
                "total_files_found": self.total_files_found,
                "total_files_missing": self.total_files_missing,
                "valid_files_count": self.valid_files_count,
                "invalid_files_count": self.invalid_files_count,
                "validation_rate": round(
                    self.valid_files_count / max(1, self.total_files_found) * 100, 2
                ),
            },
            "statistics": {
                "average_note_count": round(self.average_note_count, 2) if self.average_note_count else None,
                "average_duration": round(self.average_duration, 3) if self.average_duration else None,
                "average_instrument_count": round(self.average_instrument_count, 2) if self.average_instrument_count else None,
                "note_count_range": list(self.note_count_range) if self.note_count_range else None,
                "duration_range": [round(d, 3) for d in self.duration_range] if self.duration_range else None,
                "pitch_range": list(self.pitch_range) if self.pitch_range else None,
            },
            "mood_distribution": dict(sorted(self.mood_distribution.items(), key=lambda x: -x[1])),
            "error_distribution": dict(sorted(self.error_distribution.items(), key=lambda x: -x[1])),
            "valid_files": self.valid_files,
            "invalid_files": self.invalid_files,
            "missing_files": self.missing_files,
        }


# =============================================================================
# Validation Functions
# =============================================================================

def load_metadata(metadata_path: str) -> Dict[str, str]:
    """
    Load metadata JSON file.

    Supports two formats:
    1. Array of objects: [{"Filename": "x.mid", "Mood Description": "happy"}, ...]
    2. Object mapping: {"x.mid": "happy", ...}

    Args:
        metadata_path: Path to JSON metadata file.

    Returns:
        Dictionary mapping filename -> mood label.
    """
    with open(metadata_path, "r") as f:
        data = json.load(f)

    # Handle array format
    if isinstance(data, list):
        metadata = {}
        for entry in data:
            filename = entry.get("Filename") or entry.get("filename")
            mood = entry.get("Mood Description") or entry.get("mood") or entry.get("mood_description")
            if filename and mood:
                metadata[filename] = mood.lower().strip()
        return metadata

    # Handle object format
    if isinstance(data, dict):
        return {k: v.lower().strip() for k, v in data.items()}

    raise ValueError(f"Unsupported metadata format in {metadata_path}")


def validate_midi_file(
    filepath: str,
    mood_label: Optional[str],
    valid_moods: Set[str],
) -> FileValidationResult:
    """
    Validate a single MIDI file.

    Args:
        filepath: Path to the MIDI file.
        mood_label: Expected mood label from metadata.
        valid_moods: Set of valid mood labels.

    Returns:
        FileValidationResult with validation status and details.
    """
    filename = os.path.basename(filepath)
    result = FileValidationResult(filename=filename, is_valid=True)

    # Check 1: MIDI loads successfully
    try:
        midi = pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        result.is_valid = False
        result.errors.append(ValidationError(
            code="MIDI_LOAD_ERROR",
            message=f"Failed to load MIDI file: {str(e)}",
        ))
        return result

    # Check 2: At least 1 instrument track
    instruments = [inst for inst in midi.instruments if len(inst.notes) > 0]
    result.instrument_count = len(instruments)

    if len(instruments) < 1:
        result.is_valid = False
        result.errors.append(ValidationError(
            code="NO_INSTRUMENTS",
            message="MIDI file has no instrument tracks with notes",
        ))
        return result

    # Collect all notes across instruments
    all_notes = []
    for inst in instruments:
        all_notes.extend(inst.notes)

    result.note_count = len(all_notes)

    # Check 3: Note count > 8
    if result.note_count <= MIN_NOTE_COUNT:
        result.is_valid = False
        result.errors.append(ValidationError(
            code="INSUFFICIENT_NOTES",
            message=f"Note count ({result.note_count}) must be > {MIN_NOTE_COUNT}",
            details={"note_count": result.note_count, "minimum": MIN_NOTE_COUNT},
        ))

    # Check 4: Duration < 30 seconds
    result.duration = midi.get_end_time()

    if result.duration > MAX_DURATION_SECONDS:
        result.is_valid = False
        result.errors.append(ValidationError(
            code="DURATION_TOO_LONG",
            message=f"Duration ({result.duration:.2f}s) exceeds {MAX_DURATION_SECONDS}s",
            details={"duration": result.duration, "maximum": MAX_DURATION_SECONDS},
        ))

    # Check 5: No invalid pitch values
    pitches = [note.pitch for note in all_notes]
    invalid_pitches = [p for p in pitches if p < MIN_PITCH or p > MAX_PITCH]

    if pitches:
        result.pitch_range = (min(pitches), max(pitches))

    if invalid_pitches:
        result.is_valid = False
        result.errors.append(ValidationError(
            code="INVALID_PITCH",
            message=f"Found {len(invalid_pitches)} notes with invalid pitch values",
            details={
                "invalid_pitches": list(set(invalid_pitches))[:10],
                "valid_range": [MIN_PITCH, MAX_PITCH],
            },
        ))

    # Check 6: Mood label exists and is valid
    if mood_label is None:
        result.is_valid = False
        result.errors.append(ValidationError(
            code="MISSING_MOOD",
            message="No mood label found in metadata for this file",
        ))
    elif mood_label.lower() not in valid_moods:
        result.is_valid = False
        result.errors.append(ValidationError(
            code="INVALID_MOOD",
            message=f"Mood label '{mood_label}' is not in the valid moods list",
            details={"mood": mood_label},
        ))
    else:
        result.mood = mood_label.lower()

    # Add warnings for edge cases
    if result.note_count and result.note_count < 16:
        result.warnings.append(ValidationError(
            code="LOW_NOTE_COUNT",
            message=f"Note count ({result.note_count}) is low but valid",
        ))

    if result.duration and result.duration < 1.0:
        result.warnings.append(ValidationError(
            code="SHORT_DURATION",
            message=f"Duration ({result.duration:.2f}s) is very short",
        ))

    return result


def validate_dataset(
    midi_dir: str,
    metadata_path: str,
    valid_moods: Optional[Set[str]] = None,
    verbose: bool = False,
) -> DatasetReport:
    """
    Validate entire dataset and generate report.

    Args:
        midi_dir: Directory containing MIDI files.
        metadata_path: Path to JSON metadata file.
        valid_moods: Set of valid mood labels (uses defaults if None).
        verbose: Whether to print detailed progress.

    Returns:
        DatasetReport with complete validation results.
    """
    if valid_moods is None:
        valid_moods = DEFAULT_VALID_MOODS

    # Normalize valid moods to lowercase
    valid_moods = {m.lower() for m in valid_moods}

    # Load metadata
    print(f"Loading metadata from: {metadata_path}")
    metadata = load_metadata(metadata_path)
    print(f"Found {len(metadata)} entries in metadata")

    # Track results
    valid_results: List[FileValidationResult] = []
    invalid_results: List[FileValidationResult] = []
    missing_files: List[str] = []

    error_counter: Counter = Counter()
    mood_counter: Counter = Counter()

    # Statistics accumulators
    note_counts: List[int] = []
    durations: List[float] = []
    instrument_counts: List[int] = []
    all_pitches: List[int] = []

    # Validate each file in metadata
    midi_dir_path = Path(midi_dir)
    filenames = list(metadata.keys())

    print(f"Validating {len(filenames)} files...")

    for filename in tqdm(filenames, desc="Validating", disable=not sys.stdout.isatty()):
        filepath = midi_dir_path / filename

        # Check if file exists
        if not filepath.exists():
            missing_files.append(filename)
            continue

        # Get mood label
        mood_label = metadata.get(filename)

        # Validate file
        result = validate_midi_file(str(filepath), mood_label, valid_moods)

        if result.is_valid:
            valid_results.append(result)

            # Accumulate statistics
            if result.note_count:
                note_counts.append(result.note_count)
            if result.duration:
                durations.append(result.duration)
            if result.instrument_count:
                instrument_counts.append(result.instrument_count)
            if result.pitch_range:
                all_pitches.extend(result.pitch_range)
            if result.mood:
                mood_counter[result.mood] += 1
        else:
            invalid_results.append(result)

            # Count errors
            for error in result.errors:
                error_counter[error.code] += 1

        # Verbose output
        if verbose and not result.is_valid:
            print(f"\n  INVALID: {filename}")
            for error in result.errors:
                print(f"    - {error.code}: {error.message}")

    # Calculate statistics
    avg_note_count = sum(note_counts) / len(note_counts) if note_counts else None
    avg_duration = sum(durations) / len(durations) if durations else None
    avg_instruments = sum(instrument_counts) / len(instrument_counts) if instrument_counts else None

    note_range = (min(note_counts), max(note_counts)) if note_counts else None
    duration_range = (min(durations), max(durations)) if durations else None
    pitch_range = (min(all_pitches), max(all_pitches)) if all_pitches else None

    # Build report
    report = DatasetReport(
        generated_at=datetime.now().isoformat(),
        midi_directory=str(midi_dir_path.absolute()),
        metadata_file=str(Path(metadata_path).absolute()),
        total_files_in_metadata=len(metadata),
        total_files_found=len(valid_results) + len(invalid_results),
        total_files_missing=len(missing_files),
        valid_files_count=len(valid_results),
        invalid_files_count=len(invalid_results),
        valid_files=[r.to_dict() for r in valid_results],
        invalid_files=[r.to_dict() for r in invalid_results],
        missing_files=sorted(missing_files),
        mood_distribution=dict(mood_counter),
        error_distribution=dict(error_counter),
        average_note_count=avg_note_count,
        average_duration=avg_duration,
        average_instrument_count=avg_instruments,
        note_count_range=note_range,
        duration_range=duration_range,
        pitch_range=pitch_range,
    )

    return report


def print_summary(report: DatasetReport) -> None:
    """Print a human-readable summary of the report."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print(f"\nFiles in metadata:  {report.total_files_in_metadata}")
    print(f"Files found:        {report.total_files_found}")
    print(f"Files missing:      {report.total_files_missing}")
    print(f"Valid files:        {report.valid_files_count}")
    print(f"Invalid files:      {report.invalid_files_count}")

    if report.total_files_found > 0:
        rate = report.valid_files_count / report.total_files_found * 100
        print(f"Validation rate:    {rate:.1f}%")

    if report.average_note_count:
        print(f"\nAverage note count:      {report.average_note_count:.1f}")
    if report.average_duration:
        print(f"Average duration:        {report.average_duration:.2f}s")
    if report.average_instrument_count:
        print(f"Average instruments:     {report.average_instrument_count:.1f}")

    if report.note_count_range:
        print(f"Note count range:        {report.note_count_range[0]} - {report.note_count_range[1]}")
    if report.duration_range:
        print(f"Duration range:          {report.duration_range[0]:.2f}s - {report.duration_range[1]:.2f}s")
    if report.pitch_range:
        print(f"Pitch range:             {report.pitch_range[0]} - {report.pitch_range[1]}")

    if report.mood_distribution:
        print("\nMood Distribution:")
        for mood, count in sorted(report.mood_distribution.items(), key=lambda x: -x[1]):
            pct = count / report.valid_files_count * 100 if report.valid_files_count else 0
            bar = "█" * int(pct / 5)
            print(f"  {mood:20} {count:4} ({pct:5.1f}%) {bar}")

    if report.error_distribution:
        print("\nError Distribution:")
        for error, count in sorted(report.error_distribution.items(), key=lambda x: -x[1]):
            print(f"  {error:25} {count:4}")

    print("\n" + "=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate MIDI dataset against metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation
    python scripts/validate_dataset.py --midi-dir data/training/augmented \\
        --metadata data/training/arpeggio_mood_data_updated.json

    # Custom output path
    python scripts/validate_dataset.py --midi-dir data/midis \\
        --metadata data/metadata.json \\
        --output reports/validation.json

    # With custom valid moods
    python scripts/validate_dataset.py --midi-dir data/midis \\
        --metadata data/metadata.json \\
        --valid-moods happy sad calm energetic
        """,
    )

    parser.add_argument(
        "--midi-dir", "-m",
        type=str,
        required=True,
        help="Directory containing MIDI files",
    )

    parser.add_argument(
        "--metadata", "-d",
        type=str,
        required=True,
        help="Path to JSON metadata file",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dataset_report.json",
        help="Output path for report JSON (default: dataset_report.json)",
    )

    parser.add_argument(
        "--valid-moods",
        type=str,
        nargs="+",
        default=None,
        help="List of valid mood labels (uses defaults if not specified)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed validation errors",
    )

    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing summary to console",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.midi_dir).is_dir():
        print(f"Error: MIDI directory not found: {args.midi_dir}")
        sys.exit(1)

    if not Path(args.metadata).is_file():
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Convert valid moods to set if provided
    valid_moods = set(args.valid_moods) if args.valid_moods else None

    # Run validation
    report = validate_dataset(
        midi_dir=args.midi_dir,
        metadata_path=args.metadata,
        valid_moods=valid_moods,
        verbose=args.verbose,
    )

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\nReport saved to: {output_path}")

    # Print summary
    if not args.no_summary:
        print_summary(report)

    # Exit with error code if there are invalid files
    if report.invalid_files_count > 0 or report.total_files_missing > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
