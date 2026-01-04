#!/usr/bin/env python3
"""
Anatomy Annotation Review Tool

CLI tool for stratified review of Gemini anatomy annotations.
Displays frames with overlaid point annotations for human verification.

Stratified sampling:
- High confidence (>=0.9): 10% sample
- Medium confidence (0.7-0.9): 30% sample
- Low confidence (<0.7): 100% review

Usage:
    python anatomy_review_tool.py                    # Start/resume review
    python anatomy_review_tool.py --status          # Show review progress
    python anatomy_review_tool.py --export          # Export reviewed annotations
    python anatomy_review_tool.py --visualize       # Generate visualization report
"""

import os
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

# Configuration
ANNOTATIONS_FILE = "gemini_annotations/intermediate_results.json"
REVIEW_STATE_FILE = "anatomy_review_state.json"
REVIEWED_OUTPUT_FILE = "anatomy_reviewed_annotations.json"
VISUALIZATION_DIR = "anatomy_review_visualizations"

# Confidence thresholds and sampling rates
HIGH_CONF_THRESHOLD = 0.9
MEDIUM_CONF_THRESHOLD = 0.7
HIGH_CONF_SAMPLE_RATE = 0.10    # 10% of high confidence
MEDIUM_CONF_SAMPLE_RATE = 0.30  # 30% of medium confidence
LOW_CONF_SAMPLE_RATE = 1.0      # 100% of low confidence

# Random seed for reproducibility
RANDOM_SEED = 42


class AnatomyReviewTool:
    """Tool for reviewing and validating anatomy annotations."""

    def __init__(self):
        self.annotations = []
        self.review_queue = []
        self.reviewed = {}
        self.current_idx = 0
        self.dataset = None
        self.frame_cache = {}

    def load_annotations(self) -> bool:
        """Load Gemini anatomy annotations."""
        if not os.path.exists(ANNOTATIONS_FILE):
            print(f"Error: Annotations file not found: {ANNOTATIONS_FILE}")
            return False

        with open(ANNOTATIONS_FILE) as f:
            all_annotations = json.load(f)

        # Filter to anatomy only
        self.annotations = [
            a for a in all_annotations
            if a.get('category') == 'anatomy'
        ]

        print(f"Loaded {len(self.annotations)} anatomy annotations")
        return True

    def load_review_state(self) -> bool:
        """Load existing review state if available."""
        if os.path.exists(REVIEW_STATE_FILE):
            with open(REVIEW_STATE_FILE) as f:
                state = json.load(f)

            self.reviewed = state.get('reviewed', {})
            self.review_queue = state.get('review_queue', [])
            self.current_idx = state.get('current_idx', 0)

            print(f"Resumed review: {len(self.reviewed)} annotations reviewed")
            print(f"Queue position: {self.current_idx}/{len(self.review_queue)}")
            return True
        return False

    def save_review_state(self):
        """Save review state for resuming later."""
        state = {
            'reviewed': self.reviewed,
            'review_queue': self.review_queue,
            'current_idx': self.current_idx,
            'last_updated': datetime.now().isoformat(),
        }

        with open(REVIEW_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

    def create_review_queue(self):
        """Create stratified review queue based on confidence."""
        random.seed(RANDOM_SEED)

        # Categorize by confidence
        high_conf = []
        medium_conf = []
        low_conf = []

        for i, ann in enumerate(self.annotations):
            conf = ann.get('confidence', 0)
            ann_id = self._get_annotation_id(ann)

            # Skip already reviewed
            if ann_id in self.reviewed:
                continue

            if conf >= HIGH_CONF_THRESHOLD:
                high_conf.append((i, ann))
            elif conf >= MEDIUM_CONF_THRESHOLD:
                medium_conf.append((i, ann))
            else:
                low_conf.append((i, ann))

        print(f"\nConfidence distribution (unreviewed):")
        print(f"  High (>={HIGH_CONF_THRESHOLD}): {len(high_conf)}")
        print(f"  Medium ({MEDIUM_CONF_THRESHOLD}-{HIGH_CONF_THRESHOLD}): {len(medium_conf)}")
        print(f"  Low (<{MEDIUM_CONF_THRESHOLD}): {len(low_conf)}")

        # Sample based on rates
        queue = []

        # High confidence: sample 10%
        n_high = max(1, int(len(high_conf) * HIGH_CONF_SAMPLE_RATE))
        sampled_high = random.sample(high_conf, min(n_high, len(high_conf)))
        queue.extend(sampled_high)

        # Medium confidence: sample 30%
        n_med = max(1, int(len(medium_conf) * MEDIUM_CONF_SAMPLE_RATE))
        sampled_med = random.sample(medium_conf, min(n_med, len(medium_conf)))
        queue.extend(sampled_med)

        # Low confidence: all
        queue.extend(low_conf)

        # Shuffle for varied review experience
        random.shuffle(queue)

        self.review_queue = [(idx, ann) for idx, ann in queue]

        print(f"\nReview queue created:")
        print(f"  High conf sampled: {len(sampled_high)}")
        print(f"  Medium conf sampled: {len(sampled_med)}")
        print(f"  Low conf (all): {len(low_conf)}")
        print(f"  Total to review: {len(self.review_queue)}")

        return len(self.review_queue)

    def _get_annotation_id(self, ann: Dict) -> str:
        """Generate unique ID for an annotation."""
        video = ann.get('video_id', '')
        label = ann.get('label', '')
        timestamps = ann.get('two_fps_timestamps', [])
        ts_str = '_'.join(str(t) for t in timestamps[:3])
        return f"{video}_{label}_{ts_str}"

    def load_dataset(self):
        """Load HuggingFace dataset for frame images."""
        try:
            from datasets import load_dataset
            print("\nLoading frame dataset...")
            self.dataset = load_dataset('mmrech/pitvqa-spatial-vlm', split='train')
            print(f"Loaded {len(self.dataset)} frames")

            # Build index by video_id + frame_id
            print("Building frame index...")
            self.frame_index = {}
            for i, sample in enumerate(self.dataset):
                key = (sample.get('video_id', ''), sample.get('frame_id', ''))
                if key not in self.frame_index:
                    self.frame_index[key] = i
            print(f"Indexed {len(self.frame_index)} unique frames")

        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            print("Will proceed without frame visualization")
            self.dataset = None

    def get_frame_image(self, video_id: str, timestamp: float):
        """Get frame image for annotation visualization."""
        if self.dataset is None:
            return None

        # Convert timestamp to frame_id (assuming 2fps -> frame number)
        # PitVQA naming: {frame_num:05d}.png
        frame_num = int(timestamp * 2)  # 2fps
        frame_id = f"{frame_num:05d}.png"

        # Check cache
        cache_key = (video_id, frame_id)
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]

        # Look up in index
        if cache_key in self.frame_index:
            idx = self.frame_index[cache_key]
            image = self.dataset[idx]['image']
            self.frame_cache[cache_key] = image
            return image

        # Try nearby frames
        for offset in range(-2, 3):
            alt_frame = f"{frame_num + offset:05d}.png"
            alt_key = (video_id, alt_frame)
            if alt_key in self.frame_index:
                idx = self.frame_index[alt_key]
                image = self.dataset[idx]['image']
                self.frame_cache[cache_key] = image
                return image

        return None

    def visualize_annotation(self, ann: Dict, save_path: Optional[str] = None):
        """Display or save frame with annotation overlay."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import matplotlib.pyplot as plt
        except ImportError:
            print("Note: PIL/matplotlib not available for visualization")
            return False

        video_id = ann.get('video_id', '')
        timestamps = ann.get('two_fps_timestamps', [0])
        points = ann.get('points', [[]])
        label = ann.get('label', '')
        confidence = ann.get('confidence', 0)

        # Get first timestamp's frame
        ts = timestamps[0] if timestamps else 0
        image = self.get_frame_image(video_id, ts)

        if image is None:
            print(f"  No frame available for {video_id} @ {ts}")
            return False

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)

        # Draw point(s)
        if points and points[0]:
            for pt in points[0]:
                x = pt.get('x', 50) / 100 * image.width
                y = pt.get('y', 50) / 100 * image.height

                # Draw crosshair
                ax.plot(x, y, 'r+', markersize=20, markeredgewidth=3)
                ax.plot(x, y, 'yo', markersize=10, alpha=0.7)

        # Title with info
        ax.set_title(
            f"{label} (conf: {confidence:.2f})\n"
            f"{video_id} @ {ts}s",
            fontsize=12
        )
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            return True
        else:
            plt.tight_layout()
            plt.show()
            return True

    def print_annotation_info(self, ann: Dict, idx: int, total: int):
        """Print annotation details for review."""
        print("\n" + "=" * 60)
        print(f"ANNOTATION {idx + 1}/{total}")
        print("=" * 60)
        print(f"  Video: {ann.get('video_id', 'N/A')}")
        print(f"  Label: {ann.get('label', 'N/A')}")
        print(f"  Confidence: {ann.get('confidence', 0):.2f}")
        print(f"  Category: {ann.get('category', 'N/A')}")

        timestamps = ann.get('two_fps_timestamps', [])
        print(f"  Timestamps: {timestamps[:5]}{'...' if len(timestamps) > 5 else ''}")

        points = ann.get('points', [[]])
        if points and points[0]:
            pt = points[0][0]
            print(f"  Point: ({pt.get('x', 0):.1f}, {pt.get('y', 0):.1f})")

        print("-" * 60)

    def interactive_review(self):
        """Run interactive review session."""
        print("\n" + "=" * 60)
        print("ANATOMY ANNOTATION REVIEW")
        print("=" * 60)
        print("\nCommands:")
        print("  [A]ccept  - Annotation is correct")
        print("  [R]eject  - Annotation is incorrect")
        print("  [M]odify  - Modify point location")
        print("  [S]kip    - Skip for now")
        print("  [V]iew    - Visualize frame (if available)")
        print("  [Q]uit    - Save and quit")
        print("=" * 60)

        # Load dataset for visualization
        self.load_dataset()

        # Create queue if needed
        if not self.review_queue:
            self.create_review_queue()

        if self.current_idx >= len(self.review_queue):
            print("\nAll annotations in queue have been reviewed!")
            return

        total = len(self.review_queue)

        while self.current_idx < total:
            orig_idx, ann = self.review_queue[self.current_idx]
            ann_id = self._get_annotation_id(ann)

            # Skip if already reviewed
            if ann_id in self.reviewed:
                self.current_idx += 1
                continue

            self.print_annotation_info(ann, self.current_idx, total)

            while True:
                try:
                    cmd = input("\nAction [A/R/M/S/V/Q]: ").strip().upper()
                except (KeyboardInterrupt, EOFError):
                    cmd = 'Q'

                if cmd == 'A':
                    self.reviewed[ann_id] = {
                        'decision': 'accepted',
                        'original': ann,
                        'timestamp': datetime.now().isoformat(),
                    }
                    print("Accepted.")
                    self.current_idx += 1
                    break

                elif cmd == 'R':
                    reason = input("Reason (optional): ").strip()
                    self.reviewed[ann_id] = {
                        'decision': 'rejected',
                        'original': ann,
                        'reason': reason,
                        'timestamp': datetime.now().isoformat(),
                    }
                    print("Rejected.")
                    self.current_idx += 1
                    break

                elif cmd == 'M':
                    print("Enter new coordinates (0-100 scale):")
                    try:
                        new_x = float(input("  X: ").strip())
                        new_y = float(input("  Y: ").strip())

                        modified_ann = ann.copy()
                        if modified_ann.get('points') and modified_ann['points'][0]:
                            modified_ann['points'][0][0] = {'x': new_x, 'y': new_y}

                        self.reviewed[ann_id] = {
                            'decision': 'modified',
                            'original': ann,
                            'modified': modified_ann,
                            'new_point': {'x': new_x, 'y': new_y},
                            'timestamp': datetime.now().isoformat(),
                        }
                        print(f"Modified to ({new_x}, {new_y}).")
                        self.current_idx += 1
                        break
                    except ValueError:
                        print("Invalid coordinates. Try again.")

                elif cmd == 'S':
                    print("Skipped.")
                    self.current_idx += 1
                    break

                elif cmd == 'V':
                    print("Generating visualization...")
                    self.visualize_annotation(ann)

                elif cmd == 'Q':
                    self.save_review_state()
                    print(f"\nProgress saved. Reviewed {len(self.reviewed)} annotations.")
                    return

                else:
                    print("Unknown command. Use A/R/M/S/V/Q.")

            # Auto-save every 10 reviews
            if len(self.reviewed) % 10 == 0:
                self.save_review_state()

        # Save final state
        self.save_review_state()
        print(f"\nReview complete! {len(self.reviewed)} annotations reviewed.")

    def show_status(self):
        """Show review progress status."""
        self.load_annotations()
        self.load_review_state()

        # Count by decision
        decisions = defaultdict(int)
        for ann_id, review in self.reviewed.items():
            decisions[review.get('decision', 'unknown')] += 1

        # Calculate acceptance rate
        total_reviewed = len(self.reviewed)
        accepted = decisions.get('accepted', 0)
        rejected = decisions.get('rejected', 0)
        modified = decisions.get('modified', 0)

        print("\n" + "=" * 60)
        print("REVIEW STATUS")
        print("=" * 60)
        print(f"\nTotal anatomy annotations: {len(self.annotations)}")
        print(f"Total reviewed: {total_reviewed}")

        if self.review_queue:
            remaining = len(self.review_queue) - self.current_idx
            print(f"Remaining in queue: {remaining}")

        print(f"\nDecision breakdown:")
        print(f"  Accepted: {accepted} ({accepted/total_reviewed*100:.1f}%)" if total_reviewed else "  Accepted: 0")
        print(f"  Rejected: {rejected} ({rejected/total_reviewed*100:.1f}%)" if total_reviewed else "  Rejected: 0")
        print(f"  Modified: {modified} ({modified/total_reviewed*100:.1f}%)" if total_reviewed else "  Modified: 0")

        if total_reviewed > 0:
            acceptance_rate = (accepted + modified) / total_reviewed * 100
            print(f"\nAcceptance rate: {acceptance_rate:.1f}%")

            if acceptance_rate >= 80:
                print("Status: GOOD - Meets 80% threshold for publication")
            else:
                print("Status: NEEDS ATTENTION - Below 80% threshold")

        print("=" * 60)

    def export_reviewed(self):
        """Export reviewed annotations to clean format."""
        self.load_annotations()
        self.load_review_state()

        if not self.reviewed:
            print("No reviewed annotations to export.")
            return

        # Build clean export
        export = {
            'metadata': {
                'total_reviewed': len(self.reviewed),
                'export_date': datetime.now().isoformat(),
                'source': 'gemini-2.5-pro',
            },
            'accepted': [],
            'rejected': [],
            'modified': [],
            'summary': {},
        }

        for ann_id, review in self.reviewed.items():
            decision = review.get('decision', 'unknown')

            if decision == 'accepted':
                export['accepted'].append(review['original'])
            elif decision == 'rejected':
                export['rejected'].append({
                    'annotation': review['original'],
                    'reason': review.get('reason', ''),
                })
            elif decision == 'modified':
                export['modified'].append({
                    'original': review['original'],
                    'modified': review.get('modified', review['original']),
                    'new_point': review.get('new_point', {}),
                })

        export['summary'] = {
            'accepted': len(export['accepted']),
            'rejected': len(export['rejected']),
            'modified': len(export['modified']),
            'acceptance_rate': (len(export['accepted']) + len(export['modified'])) / len(self.reviewed) * 100,
        }

        with open(REVIEWED_OUTPUT_FILE, 'w') as f:
            json.dump(export, f, indent=2)

        print(f"\nExported to: {REVIEWED_OUTPUT_FILE}")
        print(f"  Accepted: {export['summary']['accepted']}")
        print(f"  Rejected: {export['summary']['rejected']}")
        print(f"  Modified: {export['summary']['modified']}")
        print(f"  Acceptance rate: {export['summary']['acceptance_rate']:.1f}%")

    def generate_visualization_report(self):
        """Generate HTML report with sample visualizations."""
        import base64
        from io import BytesIO

        self.load_annotations()
        self.load_review_state()
        self.load_dataset()

        os.makedirs(VISUALIZATION_DIR, exist_ok=True)

        # Sample annotations for visualization
        samples_per_decision = 5

        accepted_samples = []
        rejected_samples = []
        modified_samples = []

        for ann_id, review in self.reviewed.items():
            decision = review.get('decision', '')
            if decision == 'accepted' and len(accepted_samples) < samples_per_decision:
                accepted_samples.append(review)
            elif decision == 'rejected' and len(rejected_samples) < samples_per_decision:
                rejected_samples.append(review)
            elif decision == 'modified' and len(modified_samples) < samples_per_decision:
                modified_samples.append(review)

        def ann_to_img_base64(ann):
            """Convert annotation visualization to base64."""
            try:
                from PIL import Image
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                video_id = ann.get('video_id', '')
                timestamps = ann.get('two_fps_timestamps', [0])
                points = ann.get('points', [[]])

                ts = timestamps[0] if timestamps else 0
                image = self.get_frame_image(video_id, ts)

                if image is None:
                    return None

                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(image)

                if points and points[0]:
                    for pt in points[0]:
                        x = pt.get('x', 50) / 100 * image.width
                        y = pt.get('y', 50) / 100 * image.height
                        ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2)

                ax.axis('off')

                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                buf.seek(0)
                return base64.b64encode(buf.read()).decode()

            except Exception as e:
                return None

        # Generate HTML
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Anatomy Review Report</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .summary { background: #ecf0f1; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .summary-item { text-align: center; padding: 15px; background: white; border-radius: 5px; }
        .summary-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .summary-label { font-size: 12px; color: #7f8c8d; }
        .sample-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .sample { border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
        .sample img { width: 100%; border-radius: 4px; }
        .sample-info { margin-top: 10px; font-size: 12px; }
        .accepted { border-color: #27ae60; }
        .rejected { border-color: #e74c3c; }
        .modified { border-color: #f39c12; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 10px; font-weight: bold; }
        .badge-accepted { background: #27ae60; color: white; }
        .badge-rejected { background: #e74c3c; color: white; }
        .badge-modified { background: #f39c12; color: white; }
    </style>
</head>
<body>
    <h1>Anatomy Annotation Review Report</h1>
"""

        # Summary section
        total = len(self.reviewed)
        accepted = sum(1 for r in self.reviewed.values() if r.get('decision') == 'accepted')
        rejected = sum(1 for r in self.reviewed.values() if r.get('decision') == 'rejected')
        modified = sum(1 for r in self.reviewed.values() if r.get('decision') == 'modified')
        rate = (accepted + modified) / total * 100 if total > 0 else 0

        html += f"""
    <div class="summary">
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-value">{total}</div>
                <div class="summary-label">Total Reviewed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value" style="color: #27ae60;">{accepted}</div>
                <div class="summary-label">Accepted</div>
            </div>
            <div class="summary-item">
                <div class="summary-value" style="color: #e74c3c;">{rejected}</div>
                <div class="summary-label">Rejected</div>
            </div>
            <div class="summary-item">
                <div class="summary-value" style="color: {'#27ae60' if rate >= 80 else '#e74c3c'};">{rate:.1f}%</div>
                <div class="summary-label">Acceptance Rate</div>
            </div>
        </div>
    </div>
"""

        # Sample sections
        for section_name, samples, badge_class in [
            ("Accepted Samples", accepted_samples, "badge-accepted"),
            ("Rejected Samples", rejected_samples, "badge-rejected"),
            ("Modified Samples", modified_samples, "badge-modified"),
        ]:
            html += f"<h2>{section_name}</h2>\n<div class=\"sample-grid\">\n"

            for review in samples:
                ann = review.get('original', {})
                img_b64 = ann_to_img_base64(ann)

                if img_b64:
                    label = ann.get('label', 'N/A')
                    conf = ann.get('confidence', 0)
                    video = ann.get('video_id', 'N/A')

                    html += f"""
        <div class="sample {badge_class.replace('badge-', '')}">
            <img src="data:image/png;base64,{img_b64}" alt="{label}">
            <div class="sample-info">
                <span class="badge {badge_class}">{section_name.split()[0]}</span><br>
                <strong>{label}</strong> (conf: {conf:.2f})<br>
                {video}
            </div>
        </div>
"""

            html += "</div>\n"

        html += """
</body>
</html>
"""

        report_path = os.path.join(VISUALIZATION_DIR, "review_report.html")
        with open(report_path, 'w') as f:
            f.write(html)

        print(f"\nVisualization report saved to: {report_path}")
        print("Open in browser to view samples.")


def main():
    parser = argparse.ArgumentParser(description="Anatomy Annotation Review Tool")
    parser.add_argument('--status', action='store_true', help='Show review progress')
    parser.add_argument('--export', action='store_true', help='Export reviewed annotations')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization report')
    parser.add_argument('--reset', action='store_true', help='Reset review state (start fresh)')

    args = parser.parse_args()

    tool = AnatomyReviewTool()

    if args.status:
        tool.show_status()
    elif args.export:
        tool.export_reviewed()
    elif args.visualize:
        tool.generate_visualization_report()
    elif args.reset:
        if os.path.exists(REVIEW_STATE_FILE):
            os.remove(REVIEW_STATE_FILE)
            print("Review state reset. Starting fresh next time.")
        else:
            print("No review state to reset.")
    else:
        # Interactive review
        if not tool.load_annotations():
            return
        tool.load_review_state()
        tool.interactive_review()


if __name__ == "__main__":
    main()
