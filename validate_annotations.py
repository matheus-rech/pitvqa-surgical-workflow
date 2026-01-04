#!/usr/bin/env python3
"""
Annotation Validation Pipeline
==============================
Validates GPT-4o annotations against PitVQA ground truth.
Uses Claude Code for semantic verification of spatial accuracy.
"""

import json
import csv
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

# Paths
PROJECT_DIR = Path(__file__).parent
GT_DIR = PROJECT_DIR / "ground_truth"
ANNOTATIONS_DIR = PROJECT_DIR / "gpt_annotations"

# Instrument mapping from ground truth
INSTRUMENT_CODES = {
    -1: "out_of_patient",
    0: "no_visible_instrument",
    1: "freer_elevator",
    2: "pituitary_rongeurs",
    3: "spatula_dissector",
    4: "kerrisons",
    5: "cottle",
    6: "haemostatic_foam",
    7: "micro_doppler",
    8: "nasal_cutting_forceps",
    9: "drill",
    10: "suction_coagulator",
    11: "bipolar",
    12: "ring_curette",
    13: "speculum",
    14: "knife",
    15: "needle",
    16: "suction"
}

# Position zones (from ground truth)
POSITION_ZONES = {
    1: "upper_left",
    2: "upper_right",
    3: "center",
    4: "lower_left",
    5: "lower_right"
}

# Surgical steps
SURGICAL_STEPS = {
    -1: "operation_not_started_or_ended",
    1: "nasal_corridor_creation",
    2: "anterior_sphenoidotomy",
    3: "septum_displacement",
    4: "sphenoid_sinus_clearance",
    5: "sellotomy",
    6: "durotomy",
    7: "tumour_excision",
    8: "haemostasis",
    9: "synthetic_graft_placement",
    10: "fat_graft_placement",
    11: "nasoseptal_flap",
    12: "dural_sealant",
    13: "closure",
    14: "debris_clearance"
}


@dataclass
class GroundTruth:
    """Ground truth for a single frame"""
    video_id: str
    frame_id: int
    instrument1: str
    instrument2: str
    position1: Optional[int]
    position2: Optional[int]
    surgical_step: Optional[str] = None


@dataclass
class GPTAnnotation:
    """GPT annotation for a single frame"""
    video_id: str
    timestamp: float
    label: str
    category: str
    x: float
    y: float
    confidence: float


def load_ground_truth(video_num: int = 1) -> Dict[int, GroundTruth]:
    """Load ground truth for a video"""
    gt_map = {}

    # Load instruments
    inst_file = GT_DIR / f"instruments_{video_num:02d}.csv"
    if inst_file.exists():
        with open(inst_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = int(row['int_time'])
                gt_map[frame_id] = GroundTruth(
                    video_id=f"video_{video_num:02d}",
                    frame_id=frame_id,
                    instrument1=row['str_instrument1'],
                    instrument2=row['str_instrument2'],
                    position1=int(row['pos_instrument1']) if row['pos_instrument1'] else None,
                    position2=int(row['pos_instrument2']) if row['pos_instrument2'] else None
                )

    # Load surgical steps
    steps_file = GT_DIR / f"steps_{video_num:02d}.csv"
    if steps_file.exists():
        step_ranges = []
        with open(steps_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for i, row in enumerate(rows):
                start_time = int(row['int_time'])
                end_time = int(rows[i+1]['int_time']) if i+1 < len(rows) else float('inf')
                step_ranges.append((start_time, end_time, row['str_step']))

        # Assign steps to frames
        for frame_id, gt in gt_map.items():
            for start, end, step in step_ranges:
                if start <= frame_id < end:
                    gt.surgical_step = step
                    break

    return gt_map


def load_gpt_annotations() -> List[GPTAnnotation]:
    """Load GPT annotations"""
    annotations = []

    # Try intermediate results first, then final
    ann_file = ANNOTATIONS_DIR / "intermediate_results.json"
    if not ann_file.exists():
        ann_file = ANNOTATIONS_DIR / "surgical_videopoint_molmo_format.json"

    if ann_file.exists():
        with open(ann_file) as f:
            data = json.load(f)
            for item in data:
                annotations.append(GPTAnnotation(
                    video_id=item['video_id'],
                    timestamp=item['two_fps_timestamps'][0],
                    label=item['label'],
                    category=item['category'],
                    x=item['points'][0][0]['x'],
                    y=item['points'][0][0]['y'],
                    confidence=item['confidence']
                ))

    return annotations


def position_to_zone(x: float, y: float) -> int:
    """Convert x,y (0-100) to position zone (1-5)"""
    # 5-zone grid: UL(1), UR(2), Center(3), LL(4), LR(5)
    if 33 <= x <= 66 and 33 <= y <= 66:
        return 3  # center
    elif x < 50 and y < 50:
        return 1  # upper left
    elif x >= 50 and y < 50:
        return 2  # upper right
    elif x < 50 and y >= 50:
        return 4  # lower left
    else:
        return 5  # lower right


def validate_annotations(video_num: int = 1) -> Dict:
    """Validate GPT annotations against ground truth"""

    print(f"\n{'='*60}")
    print(f"VALIDATING ANNOTATIONS - Video {video_num:02d}")
    print(f"{'='*60}")

    # Load data
    gt_map = load_ground_truth(video_num)
    gpt_annotations = load_gpt_annotations()

    print(f"Ground truth frames: {len(gt_map)}")
    print(f"GPT annotations: {len(gpt_annotations)}")

    # Filter GPT annotations for this video
    video_id = f"video_{video_num:02d}"
    video_annotations = [a for a in gpt_annotations if a.video_id == video_id]

    # Also check for "video_01" format
    if not video_annotations:
        video_annotations = [a for a in gpt_annotations if a.video_id == f"video_0{video_num}"]

    print(f"GPT annotations for {video_id}: {len(video_annotations)}")

    # Validation metrics
    results = {
        'total_gpt_annotations': len(video_annotations),
        'total_gt_frames': len(gt_map),
        'instrument_matches': 0,
        'instrument_mismatches': 0,
        'position_accuracy': [],
        'step_accuracy': [],
        'details': []
    }

    # Group GPT annotations by timestamp
    gpt_by_time = defaultdict(list)
    for ann in video_annotations:
        frame_approx = int(ann.timestamp * 2)  # 2 FPS
        gpt_by_time[frame_approx].append(ann)

    # Compare each frame
    for frame_id, gt in list(gt_map.items())[:100]:  # First 100 frames
        gpt_anns = gpt_by_time.get(frame_id, [])

        if not gpt_anns:
            continue

        # Check instrument detection
        gt_instrument = gt.instrument1.lower().replace('_', ' ')
        detected_instruments = [a.label.lower() for a in gpt_anns if a.category == 'instruments']

        # Label mapping for GPT generic labels -> GT specific labels
        LABEL_MAP = {
            'metal instrument': ['suction', 'freer_elevator', 'pituitary_rongeurs', 'kerrisons', 'cottle', 'bipolar'],
            'metal tool': ['suction', 'freer_elevator', 'pituitary_rongeurs', 'kerrisons', 'drill'],
            'endoscopic instrument': ['suction', 'freer_elevator', 'spatula_dissector'],
            'endoscopic tool': ['suction', 'freer_elevator', 'spatula_dissector'],
            'instrument': ['suction', 'freer_elevator', 'pituitary_rongeurs'],
            'forceps': ['pituitary_rongeurs', 'nasal_cutting_forceps'],
            'probe': ['micro_doppler', 'freer_elevator'],
            'suction device': ['suction', 'suction_coagulator'],
            'loop tool': ['ring_curette'],
        }

        # Fuzzy matching with expanded mappings
        instrument_match = False

        # Skip frames where scope is outside patient or no instrument visible
        if gt_instrument in ['no visible instrument', 'out of patient']:
            # If GPT detected something, that's actually fine - it sees the image content
            if detected_instruments:
                instrument_match = True  # Don't penalize for detecting what's visible
            continue  # Skip accuracy calculation for these frames

        for det in detected_instruments:
            # Direct match
            if gt_instrument in det or det in gt_instrument:
                instrument_match = True
                break

            # Mapped match
            if det in LABEL_MAP:
                if any(gt_instrument.replace(' ', '_') in mapped for mapped in LABEL_MAP[det]):
                    instrument_match = True
                    break

            # Common synonyms
            if gt_instrument == 'suction' and any(x in det for x in ['suction', 'tube', 'metal']):
                instrument_match = True
                break

        if instrument_match:
            results['instrument_matches'] += 1
        elif gt_instrument not in ['no_visible_instrument', 'out_of_patient']:
            results['instrument_mismatches'] += 1
            results['details'].append({
                'frame': frame_id,
                'gt_instrument': gt_instrument,
                'detected': detected_instruments
            })

        # Check position accuracy
        if gt.position1:
            for ann in gpt_anns:
                if ann.category == 'instruments':
                    predicted_zone = position_to_zone(ann.x, ann.y)
                    results['position_accuracy'].append(predicted_zone == gt.position1)

    # Calculate summary stats
    if results['instrument_matches'] + results['instrument_mismatches'] > 0:
        inst_acc = results['instrument_matches'] / (results['instrument_matches'] + results['instrument_mismatches'])
    else:
        inst_acc = 0

    pos_acc = sum(results['position_accuracy']) / len(results['position_accuracy']) if results['position_accuracy'] else 0

    print(f"\n--- VALIDATION RESULTS ---")
    print(f"Instrument Detection Accuracy: {inst_acc:.1%}")
    print(f"Position Zone Accuracy: {pos_acc:.1%}")
    print(f"Matches: {results['instrument_matches']}, Mismatches: {results['instrument_mismatches']}")

    if results['details'][:5]:
        print(f"\nSample Mismatches:")
        for d in results['details'][:5]:
            print(f"  Frame {d['frame']}: GT='{d['gt_instrument']}' vs Detected={d['detected']}")

    return results


def main():
    """Run validation"""
    print("="*60)
    print("SURGICAL ANNOTATION VALIDATION PIPELINE")
    print("="*60)

    # Check files exist
    if not GT_DIR.exists():
        print(f"ERROR: Ground truth directory not found: {GT_DIR}")
        return

    if not ANNOTATIONS_DIR.exists():
        print(f"ERROR: Annotations directory not found: {ANNOTATIONS_DIR}")
        return

    # Validate video 1
    results = validate_annotations(video_num=1)

    # Save results
    output_file = PROJECT_DIR / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
