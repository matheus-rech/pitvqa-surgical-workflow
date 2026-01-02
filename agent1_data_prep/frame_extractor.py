"""
PitVQA Frame Extractor Module

Extracts frames from surgical videos with blur detection and quality filtering.
Designed for efficient processing of large surgical video datasets.

Features:
- Configurable FPS extraction
- Laplacian variance blur detection
- Automatic filtering of blurry frames
- Batch processing support
- Memory-efficient frame-by-frame processing
- Comprehensive statistics and logging
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import cv2
import numpy as np
from tqdm import tqdm

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ExtractionStats:
    """Statistics from frame extraction."""
    video_path: str
    total_frames_in_video: int = 0
    frames_sampled: int = 0
    frames_blurry: int = 0
    frames_saved: int = 0
    extraction_fps: float = 0.0
    video_fps: float = 0.0
    video_duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def blur_rate(self) -> float:
        """Percentage of frames filtered due to blur."""
        if self.frames_sampled == 0:
            return 0.0
        return (self.frames_blurry / self.frames_sampled) * 100

    def to_dict(self) -> Dict:
        """Convert stats to dictionary."""
        return {
            'video_path': self.video_path,
            'total_frames_in_video': self.total_frames_in_video,
            'frames_sampled': self.frames_sampled,
            'frames_blurry': self.frames_blurry,
            'frames_saved': self.frames_saved,
            'blur_rate_percent': round(self.blur_rate, 2),
            'extraction_fps': self.extraction_fps,
            'video_fps': self.video_fps,
            'video_duration_seconds': round(self.video_duration_seconds, 2),
            'errors': self.errors
        }


@dataclass
class BatchExtractionStats:
    """Aggregated statistics from batch extraction."""
    total_videos: int = 0
    videos_processed: int = 0
    videos_failed: int = 0
    total_frames_sampled: int = 0
    total_frames_blurry: int = 0
    total_frames_saved: int = 0
    per_video_stats: List[Dict] = field(default_factory=list)
    failed_videos: List[str] = field(default_factory=list)

    @property
    def overall_blur_rate(self) -> float:
        """Overall percentage of frames filtered due to blur."""
        if self.total_frames_sampled == 0:
            return 0.0
        return (self.total_frames_blurry / self.total_frames_sampled) * 100

    def to_dict(self) -> Dict:
        """Convert stats to dictionary."""
        return {
            'total_videos': self.total_videos,
            'videos_processed': self.videos_processed,
            'videos_failed': self.videos_failed,
            'total_frames_sampled': self.total_frames_sampled,
            'total_frames_blurry': self.total_frames_blurry,
            'total_frames_saved': self.total_frames_saved,
            'overall_blur_rate_percent': round(self.overall_blur_rate, 2),
            'per_video_stats': self.per_video_stats,
            'failed_videos': self.failed_videos
        }


class FrameExtractor:
    """
    Extract frames from surgical videos with blur detection.

    This class provides memory-efficient frame extraction with automatic
    filtering of blurry frames using Laplacian variance analysis.

    Attributes:
        extraction_fps: Target frames per second to extract (default: 1.0)
        blur_threshold: Laplacian variance threshold for blur detection (default: 100.0)
        min_frame_size: Minimum frame dimension to process (default: 64)

    Example:
        >>> extractor = FrameExtractor(extraction_fps=1.0, blur_threshold=100.0)
        >>> stats = extractor.extract_from_video("surgery.mp4", "output/frames")
        >>> print(f"Saved {stats['frames_saved']} frames")

        >>> # Batch processing
        >>> batch_stats = extractor.extract_from_directory("videos/", "output/")
        >>> print(f"Processed {batch_stats['videos_processed']} videos")
    """

    # Supported video extensions
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

    def __init__(
        self,
        extraction_fps: float = 1.0,
        blur_threshold: float = 100.0,
        min_frame_size: int = 64
    ):
        """
        Initialize the frame extractor.

        Args:
            extraction_fps: Target frames per second to extract.
                           Default is 1 fps (1 frame per second).
            blur_threshold: Laplacian variance threshold for blur detection.
                           Higher values = more strict (filters more frames).
                           Lower values = more permissive.
                           Default is 100.0.
            min_frame_size: Minimum frame dimension (width/height) to process.
                           Frames smaller than this are skipped.
                           Default is 64 pixels.

        Raises:
            ValueError: If extraction_fps <= 0 or blur_threshold < 0.
        """
        if extraction_fps <= 0:
            raise ValueError(f"extraction_fps must be positive, got {extraction_fps}")
        if blur_threshold < 0:
            raise ValueError(f"blur_threshold must be non-negative, got {blur_threshold}")

        self.extraction_fps = extraction_fps
        self.blur_threshold = blur_threshold
        self.min_frame_size = min_frame_size

        logger.info(
            f"FrameExtractor initialized: fps={extraction_fps}, "
            f"blur_threshold={blur_threshold}"
        )

    def is_blurry(
        self,
        frame: np.ndarray,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Detect if a frame is blurry using Laplacian variance.

        The Laplacian operator highlights regions of rapid intensity change.
        Blurry images have low variance in their Laplacian, as they lack
        sharp edges.

        Args:
            frame: Input frame as BGR or grayscale numpy array.
            threshold: Custom threshold for this check.
                      If None, uses instance threshold.

        Returns:
            True if the frame is considered blurry, False otherwise.

        Raises:
            ValueError: If frame is None or has invalid shape.
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        if len(frame.shape) < 2:
            raise ValueError(f"Invalid frame shape: {frame.shape}")

        threshold = threshold if threshold is not None else self.blur_threshold

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        is_blur = variance < threshold

        logger.debug(
            f"Blur check: variance={variance:.2f}, "
            f"threshold={threshold}, is_blurry={is_blur}"
        )

        return is_blur

    def get_laplacian_variance(self, frame: np.ndarray) -> float:
        """
        Calculate the Laplacian variance of a frame.

        Useful for analyzing blur levels without making a binary decision.

        Args:
            frame: Input frame as BGR or grayscale numpy array.

        Returns:
            Laplacian variance value. Higher = sharper, Lower = blurrier.
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _get_video_id(self, video_path: Union[str, Path]) -> str:
        """Extract video ID from file path (filename without extension)."""
        return Path(video_path).stem

    def _generate_frame_filename(
        self,
        video_id: str,
        frame_number: int
    ) -> str:
        """Generate frame filename following naming convention."""
        return f"{video_id}_frame_{frame_number:06d}.png"

    def extract_from_video(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        filter_blur: bool = True,
        video_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        show_progress: bool = True
    ) -> Dict:
        """
        Extract frames from a single video file.

        Extracts frames at the configured FPS rate, optionally filtering
        out blurry frames. Frames are saved as PNG files with the naming
        convention: {video_id}_frame_{frame_number:06d}.png

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save extracted frames.
            filter_blur: Whether to filter out blurry frames.
                        Default is True.
            video_id: Custom video ID for naming. If None, uses filename.
            start_time: Start extraction from this time (seconds).
                       None means start from beginning.
            end_time: Stop extraction at this time (seconds).
                     None means process until end.
            show_progress: Show tqdm progress bar. Default is True.

        Returns:
            Dictionary containing extraction statistics:
            - video_path: Path to processed video
            - total_frames_in_video: Total frames in video
            - frames_sampled: Frames sampled at target FPS
            - frames_blurry: Frames filtered due to blur
            - frames_saved: Frames successfully saved
            - blur_rate_percent: Percentage of blurry frames
            - extraction_fps: FPS used for extraction
            - video_fps: Original video FPS
            - video_duration_seconds: Video duration
            - errors: List of any errors encountered

        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video cannot be opened or has no frames.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        # Validate input
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if video_id is None:
            video_id = self._get_video_id(video_path)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stats
        stats = ExtractionStats(video_path=str(video_path))

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            # Get video properties
            stats.video_fps = cap.get(cv2.CAP_PROP_FPS)
            stats.total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if stats.video_fps <= 0:
                logger.warning(f"Invalid FPS detected ({stats.video_fps}), using 30")
                stats.video_fps = 30.0

            if stats.total_frames_in_video <= 0:
                raise ValueError(f"Video has no frames: {video_path}")

            stats.video_duration_seconds = stats.total_frames_in_video / stats.video_fps
            stats.extraction_fps = self.extraction_fps

            # Calculate frame interval
            frame_interval = max(1, int(stats.video_fps / self.extraction_fps))

            # Calculate start and end frames
            start_frame = 0
            if start_time is not None:
                start_frame = int(start_time * stats.video_fps)

            end_frame = stats.total_frames_in_video
            if end_time is not None:
                end_frame = min(int(end_time * stats.video_fps), stats.total_frames_in_video)

            # Estimate frames to process
            estimated_frames = (end_frame - start_frame) // frame_interval

            logger.info(
                f"Processing {video_path.name}: "
                f"{stats.total_frames_in_video} frames @ {stats.video_fps:.1f} fps, "
                f"extracting every {frame_interval} frames"
            )

            # Set video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Progress bar setup
            pbar = None
            if show_progress:
                pbar = tqdm(
                    total=estimated_frames,
                    desc=f"Extracting {video_id}",
                    unit="frames"
                )

            frame_count = start_frame
            output_frame_number = 0

            while frame_count < end_frame:
                ret, frame = cap.read()

                if not ret:
                    logger.debug(f"Failed to read frame {frame_count}")
                    break

                # Only process frames at target FPS
                if (frame_count - start_frame) % frame_interval == 0:
                    stats.frames_sampled += 1

                    # Check frame validity
                    if frame is None or frame.size == 0:
                        stats.errors.append(f"Empty frame at position {frame_count}")
                        frame_count += 1
                        continue

                    # Check minimum size
                    h, w = frame.shape[:2]
                    if h < self.min_frame_size or w < self.min_frame_size:
                        stats.errors.append(
                            f"Frame {frame_count} too small: {w}x{h}"
                        )
                        frame_count += 1
                        continue

                    # Blur detection
                    if filter_blur and self.is_blurry(frame):
                        stats.frames_blurry += 1
                        if pbar:
                            pbar.update(1)
                        frame_count += 1
                        continue

                    # Save frame
                    filename = self._generate_frame_filename(video_id, output_frame_number)
                    output_path = output_dir / filename

                    try:
                        success = cv2.imwrite(str(output_path), frame)
                        if success:
                            stats.frames_saved += 1
                            output_frame_number += 1
                        else:
                            stats.errors.append(f"Failed to save frame: {filename}")
                    except Exception as e:
                        stats.errors.append(f"Error saving frame {filename}: {str(e)}")
                        logger.error(f"Error saving frame: {e}")

                    if pbar:
                        pbar.update(1)

                frame_count += 1

            if pbar:
                pbar.close()

        finally:
            cap.release()

        # Log summary
        logger.info(
            f"Completed {video_id}: "
            f"sampled={stats.frames_sampled}, "
            f"blurry={stats.frames_blurry} ({stats.blur_rate:.1f}%), "
            f"saved={stats.frames_saved}"
        )

        if stats.errors:
            logger.warning(f"Encountered {len(stats.errors)} errors during extraction")

        return stats.to_dict()

    def extract_from_directory(
        self,
        video_dir: Union[str, Path],
        output_dir: Union[str, Path],
        filter_blur: bool = True,
        recursive: bool = False,
        show_progress: bool = True
    ) -> Dict:
        """
        Extract frames from all videos in a directory.

        Processes all supported video files in the specified directory,
        creating a subdirectory for each video's frames.

        Args:
            video_dir: Directory containing video files.
            output_dir: Base directory for extracted frames.
            filter_blur: Whether to filter blurry frames.
            recursive: Search for videos recursively in subdirectories.
            show_progress: Show progress bars.

        Returns:
            Dictionary containing aggregate statistics:
            - total_videos: Number of videos found
            - videos_processed: Successfully processed videos
            - videos_failed: Videos that failed processing
            - total_frames_sampled: Total frames sampled across all videos
            - total_frames_blurry: Total blurry frames filtered
            - total_frames_saved: Total frames saved
            - overall_blur_rate_percent: Overall blur rate
            - per_video_stats: List of per-video statistics
            - failed_videos: List of videos that failed

        Raises:
            FileNotFoundError: If video_dir doesn't exist.
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)

        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        # Find all video files
        if recursive:
            video_files = []
            for ext in self.VIDEO_EXTENSIONS:
                video_files.extend(video_dir.rglob(f"*{ext}"))
                video_files.extend(video_dir.rglob(f"*{ext.upper()}"))
        else:
            video_files = []
            for ext in self.VIDEO_EXTENSIONS:
                video_files.extend(video_dir.glob(f"*{ext}"))
                video_files.extend(video_dir.glob(f"*{ext.upper()}"))

        video_files = sorted(set(video_files))

        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return BatchExtractionStats().to_dict()

        logger.info(f"Found {len(video_files)} videos to process")

        # Initialize batch stats
        batch_stats = BatchExtractionStats(total_videos=len(video_files))

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each video
        video_iterator = video_files
        if show_progress:
            video_iterator = tqdm(
                video_files,
                desc="Processing videos",
                unit="video"
            )

        for video_path in video_iterator:
            video_id = self._get_video_id(video_path)
            video_output_dir = output_dir / video_id

            try:
                stats = self.extract_from_video(
                    video_path=video_path,
                    output_dir=video_output_dir,
                    filter_blur=filter_blur,
                    video_id=video_id,
                    show_progress=False  # Disable nested progress bars
                )

                batch_stats.videos_processed += 1
                batch_stats.total_frames_sampled += stats['frames_sampled']
                batch_stats.total_frames_blurry += stats['frames_blurry']
                batch_stats.total_frames_saved += stats['frames_saved']
                batch_stats.per_video_stats.append(stats)

            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                batch_stats.videos_failed += 1
                batch_stats.failed_videos.append(str(video_path))

        # Log summary
        logger.info(
            f"Batch extraction complete: "
            f"{batch_stats.videos_processed}/{batch_stats.total_videos} videos, "
            f"{batch_stats.total_frames_saved} frames saved, "
            f"overall blur rate: {batch_stats.overall_blur_rate:.1f}%"
        )

        return batch_stats.to_dict()

    def analyze_blur_distribution(
        self,
        video_path: Union[str, Path],
        sample_rate: float = 1.0,
        show_progress: bool = True
    ) -> Dict:
        """
        Analyze blur distribution in a video without saving frames.

        Useful for determining optimal blur threshold for a dataset.

        Args:
            video_path: Path to video file.
            sample_rate: Frames per second to sample for analysis.
            show_progress: Show progress bar.

        Returns:
            Dictionary containing:
            - min_variance: Minimum Laplacian variance
            - max_variance: Maximum Laplacian variance
            - mean_variance: Mean Laplacian variance
            - std_variance: Standard deviation of variance
            - percentiles: 25th, 50th, 75th, 90th percentiles
            - histogram: Histogram of variance values
            - suggested_threshold: Recommended blur threshold
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if video_fps <= 0:
                video_fps = 30.0

            frame_interval = max(1, int(video_fps / sample_rate))
            estimated_samples = total_frames // frame_interval

            variances = []

            pbar = None
            if show_progress:
                pbar = tqdm(
                    total=estimated_samples,
                    desc="Analyzing blur",
                    unit="frames"
                )

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    if frame is not None and frame.size > 0:
                        variance = self.get_laplacian_variance(frame)
                        variances.append(variance)

                    if pbar:
                        pbar.update(1)

                frame_count += 1

            if pbar:
                pbar.close()

        finally:
            cap.release()

        if not variances:
            return {'error': 'No frames could be analyzed'}

        variances = np.array(variances)

        # Calculate percentiles
        percentiles = {
            'p25': float(np.percentile(variances, 25)),
            'p50': float(np.percentile(variances, 50)),
            'p75': float(np.percentile(variances, 75)),
            'p90': float(np.percentile(variances, 90))
        }

        # Suggested threshold (below 25th percentile = blurry)
        suggested_threshold = percentiles['p25']

        # Create histogram
        hist, bin_edges = np.histogram(variances, bins=50)

        return {
            'frames_analyzed': len(variances),
            'min_variance': float(variances.min()),
            'max_variance': float(variances.max()),
            'mean_variance': float(variances.mean()),
            'std_variance': float(variances.std()),
            'percentiles': percentiles,
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'suggested_threshold': suggested_threshold,
            'current_threshold': self.blur_threshold
        }


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the frame extractor module.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional file path to write logs to.
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


if __name__ == "__main__":
    import argparse
    import json

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract frames from surgical videos with blur filtering"
    )
    parser.add_argument(
        'input',
        help="Input video file or directory"
    )
    parser.add_argument(
        'output',
        help="Output directory for extracted frames"
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help="Target extraction FPS (default: 1.0)"
    )
    parser.add_argument(
        '--blur-threshold',
        type=float,
        default=100.0,
        help="Blur detection threshold (default: 100.0)"
    )
    parser.add_argument(
        '--no-blur-filter',
        action='store_true',
        help="Disable blur filtering"
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help="Search for videos recursively"
    )
    parser.add_argument(
        '--analyze-blur',
        action='store_true',
        help="Analyze blur distribution instead of extracting"
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help="Path to log file"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)

    # Create extractor
    extractor = FrameExtractor(
        extraction_fps=args.fps,
        blur_threshold=args.blur_threshold
    )

    input_path = Path(args.input)

    if args.analyze_blur:
        # Analyze blur distribution
        if not input_path.is_file():
            print("Error: --analyze-blur requires a video file as input")
            exit(1)

        print(f"\nAnalyzing blur distribution in {input_path.name}...")
        analysis = extractor.analyze_blur_distribution(input_path)

        print("\n=== Blur Analysis Results ===")
        print(f"Frames analyzed: {analysis['frames_analyzed']}")
        print(f"Variance range: {analysis['min_variance']:.2f} - {analysis['max_variance']:.2f}")
        print(f"Mean variance: {analysis['mean_variance']:.2f} (+/- {analysis['std_variance']:.2f})")
        print(f"\nPercentiles:")
        for p, v in analysis['percentiles'].items():
            print(f"  {p}: {v:.2f}")
        print(f"\nSuggested threshold: {analysis['suggested_threshold']:.2f}")
        print(f"Current threshold: {analysis['current_threshold']:.2f}")

    elif input_path.is_file():
        # Extract from single video
        print(f"\nExtracting frames from {input_path.name}...")
        stats = extractor.extract_from_video(
            video_path=input_path,
            output_dir=args.output,
            filter_blur=not args.no_blur_filter
        )

        print("\n=== Extraction Results ===")
        print(f"Video: {stats['video_path']}")
        print(f"Video duration: {stats['video_duration_seconds']:.1f}s @ {stats['video_fps']:.1f} fps")
        print(f"Frames sampled: {stats['frames_sampled']}")
        print(f"Frames filtered (blur): {stats['frames_blurry']} ({stats['blur_rate_percent']:.1f}%)")
        print(f"Frames saved: {stats['frames_saved']}")

        if stats['errors']:
            print(f"\nEncountered {len(stats['errors'])} errors")

    elif input_path.is_dir():
        # Extract from directory
        print(f"\nProcessing videos in {input_path}...")
        stats = extractor.extract_from_directory(
            video_dir=input_path,
            output_dir=args.output,
            filter_blur=not args.no_blur_filter,
            recursive=args.recursive
        )

        print("\n=== Batch Extraction Results ===")
        print(f"Videos found: {stats['total_videos']}")
        print(f"Videos processed: {stats['videos_processed']}")
        print(f"Videos failed: {stats['videos_failed']}")
        print(f"Total frames sampled: {stats['total_frames_sampled']}")
        print(f"Total frames filtered (blur): {stats['total_frames_blurry']} ({stats['overall_blur_rate_percent']:.1f}%)")
        print(f"Total frames saved: {stats['total_frames_saved']}")

        if stats['failed_videos']:
            print(f"\nFailed videos:")
            for v in stats['failed_videos']:
                print(f"  - {v}")

        # Save detailed stats to JSON
        stats_file = Path(args.output) / "extraction_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nDetailed stats saved to: {stats_file}")

    else:
        print(f"Error: Input path does not exist: {input_path}")
        exit(1)

    print("\nDone!")
