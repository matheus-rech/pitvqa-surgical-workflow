#!/usr/bin/env python3
"""
Multi-Agent Surgical Annotation Pipeline
=========================================
Uses Claude Opus 4.5, Gemini 3 Pro, and GPT-5.2 for consensus-based
annotation of surgical instruments and anatomy in pituitary surgery frames.

Output format: Molmo2-VideoPoint compatible
"""

import os
import json
import base64
import asyncio
import aiohttp
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from PIL import Image
import io
from datetime import datetime

# API Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Surgical annotation categories
SURGICAL_CATEGORIES = {
    "instruments": [
        "pituitary_forceps", "suction_cannula", "curette", "endoscope",
        "bipolar_cautery", "drill", "dissector", "scissors", "speculum",
        "doppler_probe", "micro_hook", "tumor_forceps", "ring_curette"
    ],
    "anatomy": [
        "tumor", "pituitary_gland", "carotid_artery", "optic_nerve",
        "sella_turcica", "sphenoid_sinus", "dura_mater", "diaphragma_sellae",
        "clivus", "posterior_clinoid", "suprasellar_cistern", "arachnoid"
    ],
    "events": [
        "active_bleeding", "tumor_removal", "cauterization", "irrigation",
        "dissection", "drilling", "hemostasis", "tissue_retraction"
    ]
}

@dataclass
class PointAnnotation:
    """Single point annotation with metadata"""
    x: float  # 0-100 normalized
    y: float  # 0-100 normalized
    label: str
    category: str  # instruments, anatomy, events
    confidence: float
    annotator: str  # claude, gemini, gpt, consensus

@dataclass
class FrameAnnotation:
    """Complete annotation for a single frame"""
    video_id: str
    frame_id: int
    question: str
    label: str
    count: int
    points: list  # List of {x, y} dicts
    category: str
    timestamp: float
    confidence: float
    consensus_method: str  # unanimous, majority, tiebreaker
    annotators_agreed: list


class ClaudeAnnotator:
    """Claude Opus 4.5 with thinking for primary annotation"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "claude-opus-4-5-20250514"
        self.base_url = "https://api.anthropic.com/v1/messages"

    async def annotate_frame(self, image_base64: str, session: aiohttp.ClientSession) -> dict:
        """Annotate surgical frame with instruments and anatomy points"""

        prompt = """You are an expert neurosurgical annotator analyzing pituitary surgery endoscopic images.

TASK: Identify and localize ALL visible surgical instruments and anatomical structures.

For EACH identified object, provide:
1. Label (specific name from the categories below)
2. Point coordinates (x, y) as percentage of image (0-100)
3. Category (instruments, anatomy, or events)
4. Confidence (0.0-1.0)

INSTRUMENT CATEGORIES:
- pituitary_forceps, suction_cannula, curette, endoscope, bipolar_cautery
- drill, dissector, scissors, speculum, doppler_probe, micro_hook
- tumor_forceps, ring_curette

ANATOMY CATEGORIES:
- tumor, pituitary_gland, carotid_artery, optic_nerve, sella_turcica
- sphenoid_sinus, dura_mater, diaphragma_sellae, clivus, arachnoid

EVENT CATEGORIES:
- active_bleeding, tumor_removal, cauterization, irrigation, dissection

OUTPUT FORMAT (JSON):
{
    "annotations": [
        {
            "label": "pituitary_forceps",
            "x": 45.2,
            "y": 62.8,
            "category": "instruments",
            "confidence": 0.95,
            "reasoning": "Metal grasping instrument visible in center-right"
        }
    ],
    "surgical_phase": "tumor_removal",
    "overall_confidence": 0.92
}

Think step by step about what you see, then provide precise coordinates."""

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        try:
            async with session.post(self.base_url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    content = result["content"][0]["text"]
                    # Extract JSON from response
                    return self._parse_json_response(content, "claude")
                else:
                    error = await resp.text()
                    print(f"Claude API error: {resp.status} - {error}")
                    return {"annotations": [], "error": error, "annotator": "claude"}
        except Exception as e:
            print(f"Claude annotation error: {e}")
            return {"annotations": [], "error": str(e), "annotator": "claude"}

    def _parse_json_response(self, text: str, annotator: str) -> dict:
        """Extract JSON from model response"""
        try:
            # Try to find JSON in the response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                result = json.loads(json_str)
                result["annotator"] = annotator
                return result
        except json.JSONDecodeError:
            pass
        return {"annotations": [], "raw_response": text, "annotator": annotator}


class GeminiValidator:
    """Gemini 3 Pro for validation and classification"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gemini-2.5-pro-preview-06-05"  # Latest Gemini
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    async def validate_annotations(self, image_base64: str, claude_annotations: dict,
                                   session: aiohttp.ClientSession) -> dict:
        """Validate Claude's annotations and provide independent assessment"""

        prompt = f"""You are an expert surgical image analyst validating annotations.

PREVIOUS ANNOTATIONS (from another model):
{json.dumps(claude_annotations.get('annotations', []), indent=2)}

YOUR TASK:
1. VERIFY each annotation - is the label correct? Are coordinates accurate?
2. IDENTIFY any MISSED objects (instruments, anatomy, events)
3. CORRECT any misclassifications
4. Provide your OWN independent annotations

For each object, provide:
- label: specific instrument/anatomy name
- x, y: coordinates as percentage (0-100)
- category: instruments, anatomy, or events
- confidence: 0.0-1.0
- validation_status: "confirmed", "corrected", "rejected", or "new"

OUTPUT FORMAT (JSON):
{{
    "validated_annotations": [
        {{
            "label": "pituitary_forceps",
            "x": 45.5,
            "y": 63.0,
            "category": "instruments",
            "confidence": 0.93,
            "validation_status": "confirmed",
            "original_label": "pituitary_forceps",
            "notes": "Coordinates slightly adjusted for center of instrument"
        }}
    ],
    "new_annotations": [],
    "rejected_annotations": [],
    "overall_agreement": 0.85
}}

Think carefully about surgical anatomy and instrument identification."""

        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 4096
            }
        }

        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    return self._parse_json_response(content, "gemini")
                else:
                    error = await resp.text()
                    print(f"Gemini API error: {resp.status} - {error}")
                    return {"validated_annotations": [], "error": error, "annotator": "gemini"}
        except Exception as e:
            print(f"Gemini validation error: {e}")
            return {"validated_annotations": [], "error": str(e), "annotator": "gemini"}

    def _parse_json_response(self, text: str, annotator: str) -> dict:
        """Extract JSON from model response"""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                result = json.loads(json_str)
                result["annotator"] = annotator
                return result
        except json.JSONDecodeError:
            pass
        return {"validated_annotations": [], "raw_response": text, "annotator": annotator}


class GPTTiebreaker:
    """GPT-5.2 for resolving conflicts between Claude and Gemini"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "gpt-4o"  # Use latest available
        self.base_url = "https://api.openai.com/v1/chat/completions"

    async def resolve_conflict(self, image_base64: str, claude_result: dict,
                               gemini_result: dict, session: aiohttp.ClientSession) -> dict:
        """Resolve annotation conflicts between Claude and Gemini"""

        prompt = f"""You are an expert image annotator resolving conflicts between two annotation systems.

TWO SYSTEMS PROVIDED DIFFERENT ANNOTATIONS for objects in this endoscopic frame:

SYSTEM A ANNOTATIONS:
{json.dumps(claude_result.get('annotations', []), indent=2)}

SYSTEM B VALIDATION:
{json.dumps(gemini_result.get('validated_annotations', []), indent=2)}

YOUR TASK:
1. Analyze the image independently
2. For EACH point of disagreement, determine the CORRECT annotation
3. Provide final consensus annotations

Consider:
- Label accuracy (is the label correct for what's visible?)
- Coordinate precision (which system's coordinates are more accurate?)
- Completeness (are there objects both systems missed?)

OUTPUT FORMAT (JSON):
{{
    "final_annotations": [
        {{
            "label": "object_name",
            "x": 45.3,
            "y": 62.9,
            "category": "instruments|anatomy|events",
            "confidence": 0.95,
            "source": "system_a|system_b|independent",
            "resolution_reason": "Why this annotation was chosen"
        }}
    ],
    "conflicts_resolved": 2,
    "consensus_confidence": 0.91
}}"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.2
        }

        try:
            async with session.post(self.base_url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    content = result["choices"][0]["message"]["content"]
                    return self._parse_json_response(content, "gpt")
                else:
                    error = await resp.text()
                    print(f"GPT API error: {resp.status} - {error}")
                    return {"final_annotations": [], "error": error, "annotator": "gpt"}
        except Exception as e:
            print(f"GPT tiebreaker error: {e}")
            return {"final_annotations": [], "error": str(e), "annotator": "gpt"}

    def _parse_json_response(self, text: str, annotator: str) -> dict:
        """Extract JSON from model response"""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                result = json.loads(json_str)
                result["annotator"] = annotator
                return result
        except json.JSONDecodeError:
            pass
        return {"final_annotations": [], "raw_response": text, "annotator": annotator}


class ConsensusEngine:
    """Builds consensus from multiple annotators"""

    def __init__(self, agreement_threshold: float = 0.8, distance_threshold: float = 5.0):
        self.agreement_threshold = agreement_threshold
        self.distance_threshold = distance_threshold  # Max distance for same point (in %)

    def build_consensus(self, claude_result: dict, gemini_result: dict,
                        gpt_result: Optional[dict] = None) -> dict:
        """Build consensus annotations from multiple sources"""

        claude_annotations = claude_result.get("annotations", [])
        gemini_annotations = gemini_result.get("validated_annotations", [])

        consensus_annotations = []
        consensus_method = "unanimous"
        annotators_agreed = []

        # Match annotations by proximity and label
        matched_pairs = self._match_annotations(claude_annotations, gemini_annotations)

        for match in matched_pairs:
            if match["agreement"] >= self.agreement_threshold:
                # High agreement - use average
                consensus_annotations.append({
                    "label": match["label"],
                    "x": (match["claude_x"] + match["gemini_x"]) / 2,
                    "y": (match["claude_y"] + match["gemini_y"]) / 2,
                    "category": match["category"],
                    "confidence": match["agreement"],
                    "source": "consensus"
                })
                annotators_agreed.append(["claude", "gemini"])
            elif gpt_result and match["needs_tiebreaker"]:
                # Use GPT tiebreaker result
                gpt_annotations = gpt_result.get("final_annotations", [])
                for gpt_ann in gpt_annotations:
                    if self._is_same_object(match, gpt_ann):
                        consensus_annotations.append(gpt_ann)
                        annotators_agreed.append(["gpt_tiebreaker"])
                        consensus_method = "tiebreaker"
                        break

        # Add unique annotations from each source with high confidence
        for ann in claude_annotations:
            if not self._in_consensus(ann, consensus_annotations) and ann.get("confidence", 0) > 0.9:
                ann["source"] = "claude_unique"
                consensus_annotations.append(ann)

        for ann in gemini_annotations:
            if not self._in_consensus(ann, consensus_annotations) and ann.get("confidence", 0) > 0.9:
                ann["source"] = "gemini_unique"
                consensus_annotations.append(ann)

        return {
            "consensus_annotations": consensus_annotations,
            "consensus_method": consensus_method,
            "total_annotations": len(consensus_annotations),
            "agreement_scores": [m["agreement"] for m in matched_pairs]
        }

    def _match_annotations(self, claude_anns: list, gemini_anns: list) -> list:
        """Match annotations from different sources by proximity and label"""
        matches = []
        used_gemini = set()

        for c_ann in claude_anns:
            best_match = None
            best_distance = float('inf')

            for i, g_ann in enumerate(gemini_anns):
                if i in used_gemini:
                    continue

                distance = self._point_distance(
                    c_ann.get("x", 0), c_ann.get("y", 0),
                    g_ann.get("x", 0), g_ann.get("y", 0)
                )

                label_match = c_ann.get("label", "").lower() == g_ann.get("label", "").lower()

                if distance < self.distance_threshold and distance < best_distance:
                    best_match = (i, g_ann, distance, label_match)
                    best_distance = distance

            if best_match:
                i, g_ann, distance, label_match = best_match
                used_gemini.add(i)

                agreement = 1.0 - (distance / self.distance_threshold)
                if label_match:
                    agreement = min(1.0, agreement + 0.2)

                matches.append({
                    "label": c_ann.get("label"),
                    "category": c_ann.get("category"),
                    "claude_x": c_ann.get("x", 0),
                    "claude_y": c_ann.get("y", 0),
                    "gemini_x": g_ann.get("x", 0),
                    "gemini_y": g_ann.get("y", 0),
                    "agreement": agreement,
                    "needs_tiebreaker": not label_match and distance < self.distance_threshold
                })

        return matches

    def _point_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Euclidean distance between two points"""
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def _is_same_object(self, match: dict, annotation: dict) -> bool:
        """Check if annotation refers to same object as match"""
        avg_x = (match["claude_x"] + match["gemini_x"]) / 2
        avg_y = (match["claude_y"] + match["gemini_y"]) / 2
        distance = self._point_distance(avg_x, avg_y,
                                        annotation.get("x", 0),
                                        annotation.get("y", 0))
        return distance < self.distance_threshold

    def _in_consensus(self, annotation: dict, consensus: list) -> bool:
        """Check if annotation already in consensus list"""
        for c in consensus:
            distance = self._point_distance(
                annotation.get("x", 0), annotation.get("y", 0),
                c.get("x", 0), c.get("y", 0)
            )
            if distance < self.distance_threshold:
                return True
        return False


class SurgicalAnnotationPipeline:
    """Main pipeline orchestrating multi-agent annotation"""

    def __init__(self):
        self.claude = ClaudeAnnotator(ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        self.gemini = GeminiValidator(GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        self.gpt = GPTTiebreaker(OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.consensus = ConsensusEngine()

        # Verify API keys
        if not self.claude:
            print("Warning: ANTHROPIC_API_KEY not set - Claude annotator disabled")
        if not self.gemini:
            print("Warning: GOOGLE_API_KEY not set - Gemini validator disabled")
        if not self.gpt:
            print("Warning: OPENAI_API_KEY not set - GPT tiebreaker disabled")

    async def annotate_frame(self, image: Image.Image, video_id: str,
                             frame_id: int, timestamp: float = 0.0) -> FrameAnnotation:
        """Run full annotation pipeline on a single frame"""

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        async with aiohttp.ClientSession() as session:
            # Step 1: Claude primary annotation
            print(f"  [1/3] Claude Opus 4.5 annotating frame {frame_id}...")
            claude_result = await self.claude.annotate_frame(image_base64, session)

            # Step 2: Gemini validation
            print(f"  [2/3] Gemini 3 Pro validating annotations...")
            gemini_result = await self.gemini.validate_annotations(
                image_base64, claude_result, session
            )

            # Step 3: Check if tiebreaker needed
            gpt_result = None
            preliminary_consensus = self.consensus.build_consensus(claude_result, gemini_result)

            if any(score < self.consensus.agreement_threshold
                   for score in preliminary_consensus.get("agreement_scores", [])):
                print(f"  [3/3] GPT-5.2 resolving conflicts...")
                gpt_result = await self.gpt.resolve_conflict(
                    image_base64, claude_result, gemini_result, session
                )
            else:
                print(f"  [3/3] High agreement - no tiebreaker needed")

            # Build final consensus
            final_consensus = self.consensus.build_consensus(
                claude_result, gemini_result, gpt_result
            )

        # Convert to FrameAnnotation
        annotations = final_consensus.get("consensus_annotations", [])

        # Group by label for VideoPoint format
        if annotations:
            primary_label = annotations[0].get("label", "surgical_target")
            primary_category = annotations[0].get("category", "instruments")
            points = [{"x": a.get("x", 0), "y": a.get("y", 0)} for a in annotations]
            avg_confidence = sum(a.get("confidence", 0.5) for a in annotations) / len(annotations)
        else:
            primary_label = "no_annotation"
            primary_category = "unknown"
            points = []
            avg_confidence = 0.0

        return FrameAnnotation(
            video_id=video_id,
            frame_id=frame_id,
            question=f"Point to the {primary_label}",
            label=primary_label,
            count=len(points),
            points=points,
            category=primary_category,
            timestamp=timestamp,
            confidence=avg_confidence,
            consensus_method=final_consensus.get("consensus_method", "unknown"),
            annotators_agreed=["claude", "gemini"] + (["gpt"] if gpt_result else [])
        )

    async def annotate_batch(self, frames: list, output_path: str = None) -> list:
        """Annotate a batch of frames"""
        results = []

        for i, frame_data in enumerate(frames):
            print(f"\nProcessing frame {i+1}/{len(frames)}...")

            image = frame_data.get("image")
            video_id = frame_data.get("video_id", f"video_{i}")
            frame_id = frame_data.get("frame_id", i)
            timestamp = frame_data.get("timestamp", i * 0.5)  # Assume 2 FPS

            result = await self.annotate_frame(image, video_id, frame_id, timestamp)
            results.append(asdict(result))

            # Save intermediate results
            if output_path and (i + 1) % 10 == 0:
                self._save_results(results, output_path)

        if output_path:
            self._save_results(results, output_path)

        return results

    def _save_results(self, results: list, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} annotations to {output_path}")

    def to_molmo_videopoint_format(self, annotations: list) -> list:
        """Convert annotations to Molmo2-VideoPoint format"""
        molmo_format = []

        for ann in annotations:
            molmo_entry = {
                "video_id": ann["video_id"],
                "question": ann["question"],
                "label": ann["label"],
                "count": ann["count"],
                "two_fps_timestamps": [ann["timestamp"]],
                "points": [ann["points"]],  # Nested list per timestamp
                "raw_frames": [ann["frame_id"]],
                "raw_timestamps": [ann["timestamp"]],
                "annotator_unsure": ann["confidence"] < 0.7,
                "category": ann["category"],
                "video_source": "pitvqa_surgical"
            }
            molmo_format.append(molmo_entry)

        return molmo_format


async def main():
    """Test the annotation pipeline"""
    print("=" * 60)
    print("Surgical Annotation Pipeline - Multi-Agent Consensus System")
    print("=" * 60)

    # Check API keys
    print("\nAPI Key Status:")
    print(f"  ANTHROPIC_API_KEY: {'✓ Set' if ANTHROPIC_API_KEY else '✗ Not set'}")
    print(f"  GOOGLE_API_KEY: {'✓ Set' if GOOGLE_API_KEY else '✗ Not set'}")
    print(f"  OPENAI_API_KEY: {'✓ Set' if OPENAI_API_KEY else '✗ Not set'}")

    if not all([ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY]):
        print("\nWarning: Not all API keys are set. Pipeline may not work fully.")
        print("Set the following environment variables:")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  export GOOGLE_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")

    print("\nPipeline initialized successfully!")
    print("\nUsage:")
    print("  from surgical_annotation_pipeline import SurgicalAnnotationPipeline")
    print("  pipeline = SurgicalAnnotationPipeline()")
    print("  result = await pipeline.annotate_frame(image, video_id, frame_id)")


if __name__ == "__main__":
    asyncio.run(main())
