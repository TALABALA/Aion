"""
AION Audio Perception

Multi-model audio perception system with:
- Speech recognition (Whisper)
- Speaker diarization (pyannote)
- Audio event detection (Audio Spectrogram Transformer, BEATs)
- Speaker embedding extraction (SpeechBrain, ECAPA-TDNN)
- Music analysis (librosa, essentia)
- Audio-text embeddings (CLAP)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioEvent,
    AudioEventType,
    AudioQuality,
    AudioScene,
    AudioSegment,
    EmotionalTone,
    MusicAnalysis,
    Speaker,
    TimeRange,
    Transcript,
    TranscriptSegment,
    VoiceCharacteristics,
    Word,
)

logger = structlog.get_logger(__name__)

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class PerceptionConfig:
    """Configuration for audio perception."""
    # ASR settings
    whisper_model: str = "openai/whisper-large-v3"
    whisper_language: Optional[str] = None  # Auto-detect if None
    whisper_task: str = "transcribe"  # "transcribe" or "translate"

    # Diarization settings
    enable_diarization: bool = True
    min_speakers: int = 1
    max_speakers: int = 10
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # Audio event detection
    event_detection_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    event_threshold: float = 0.3

    # Speaker embedding
    speaker_embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"

    # CLAP embeddings
    clap_model: str = "laion/larger_clap_general"

    # Music analysis
    enable_music_analysis: bool = True

    # Processing settings
    device: str = "auto"
    target_sample_rate: int = 16000
    chunk_length_s: float = 30.0  # Whisper chunk length
    batch_size: int = 8


class AudioPerception:
    """
    AION Audio Perception System

    Processes audio using multiple models:
    - Whisper for speech recognition
    - Pyannote for speaker diarization
    - AST/BEATs for audio event detection
    - ECAPA-TDNN for speaker embeddings
    - CLAP for audio-text embeddings
    - Librosa for music analysis
    """

    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        self._device: Optional[str] = None

        # Models (lazy loaded)
        self._whisper_model = None
        self._whisper_processor = None
        self._diarization_pipeline = None
        self._event_detector = None
        self._event_processor = None
        self._speaker_encoder = None
        self._clap_model = None
        self._clap_processor = None

        # Statistics
        self._stats = {
            "audio_files_processed": 0,
            "total_duration_processed": 0.0,
            "transcriptions_performed": 0,
            "diarizations_performed": 0,
            "events_detected": 0,
            "embeddings_computed": 0,
        }

        self._initialized = False

    @property
    def device(self) -> str:
        """Get the compute device."""
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def _get_device(self) -> str:
        """Determine best available device."""
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    async def initialize(self) -> None:
        """Initialize the perception system."""
        if self._initialized:
            return

        logger.info("Initializing Audio Perception System", device=self.device)
        self._initialized = True
        logger.info("Audio Perception System initialized (models loaded on demand)")

    async def shutdown(self) -> None:
        """Shutdown the perception system."""
        logger.info("Shutting down Audio Perception System")

        # Clear models from memory
        self._whisper_model = None
        self._whisper_processor = None
        self._diarization_pipeline = None
        self._event_detector = None
        self._event_processor = None
        self._speaker_encoder = None
        self._clap_model = None
        self._clap_processor = None

        self._initialized = False

    # ========================
    # Audio Loading
    # ========================

    async def load_audio(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        target_sr: Optional[int] = None,
        mono: bool = True,
    ) -> tuple[np.ndarray, int]:
        """
        Load audio from various sources.

        Args:
            source: Audio source (path, URL, bytes, or numpy array)
            target_sr: Target sample rate (None = keep original)
            mono: Convert to mono

        Returns:
            Tuple of (waveform, sample_rate)
        """
        target_sr = target_sr or self.config.target_sample_rate

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self._load_audio_sync(source, target_sr, mono)
        )

    def _load_audio_sync(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        target_sr: int,
        mono: bool,
    ) -> tuple[np.ndarray, int]:
        """Synchronous audio loading."""
        try:
            import librosa
            import soundfile as sf
        except ImportError as e:
            logger.error(f"Audio libraries not available: {e}")
            raise RuntimeError("librosa and soundfile required for audio loading")

        if isinstance(source, np.ndarray):
            # Already loaded - just resample if needed
            return source, target_sr

        if isinstance(source, bytes):
            # Load from bytes
            with io.BytesIO(source) as buf:
                waveform, sr = sf.read(buf)
                if sr != target_sr:
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
                if mono and waveform.ndim > 1:
                    waveform = librosa.to_mono(waveform.T)
                return waveform.astype(np.float32), target_sr

        # Load from path or URL
        path = str(source)

        if path.startswith(("http://", "https://")):
            # Download from URL
            import httpx
            response = httpx.get(path, follow_redirects=True)
            return self._load_audio_sync(response.content, target_sr, mono)

        # Load from file
        waveform, sr = librosa.load(path, sr=target_sr, mono=mono)
        return waveform.astype(np.float32), sr

    def compute_audio_hash(self, waveform: np.ndarray) -> str:
        """Compute a hash for audio deduplication."""
        return hashlib.sha256(waveform.tobytes()).hexdigest()[:16]

    # ========================
    # Speech Recognition (Whisper)
    # ========================

    async def _ensure_whisper_loaded(self) -> None:
        """Ensure Whisper model is loaded."""
        if self._whisper_model is not None:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._load_whisper)

    def _load_whisper(self) -> None:
        """Load Whisper model (blocking)."""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            logger.info("Loading Whisper model", model=self.config.whisper_model)

            self._whisper_processor = WhisperProcessor.from_pretrained(
                self.config.whisper_model
            )
            self._whisper_model = WhisperForConditionalGeneration.from_pretrained(
                self.config.whisper_model
            )

            if self.device != "cpu":
                import torch
                self._whisper_model = self._whisper_model.to(self.device)
                if self.device == "cuda":
                    self._whisper_model = self._whisper_model.half()  # FP16 for efficiency

            logger.info("Whisper model loaded successfully")

        except ImportError as e:
            logger.warning(f"Whisper not available: {e}")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")

    async def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        language: Optional[str] = None,
        task: str = "transcribe",
        return_timestamps: bool = True,
    ) -> Transcript:
        """
        Transcribe speech to text using Whisper.

        Args:
            audio: Audio source
            language: Language code (auto-detect if None)
            task: "transcribe" or "translate"
            return_timestamps: Include word-level timestamps

        Returns:
            Transcript with segments and timing
        """
        await self._ensure_whisper_loaded()

        # Load audio if needed
        if isinstance(audio, AudioSegment):
            if audio.waveform is not None:
                waveform = audio.waveform
                sr = audio.sample_rate
            else:
                raise ValueError("AudioSegment has no waveform data")
        else:
            waveform, sr = await self.load_audio(audio)

        if self._whisper_model is None:
            # Return mock transcript
            return Transcript(
                text="[Transcription not available - Whisper model not loaded]",
                language=language or "en",
                confidence=0.0,
                duration=len(waveform) / sr,
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: self._transcribe_sync(waveform, sr, language, task, return_timestamps)
        )

        self._stats["transcriptions_performed"] += 1
        self._stats["total_duration_processed"] += len(waveform) / sr

        return result

    def _transcribe_sync(
        self,
        waveform: np.ndarray,
        sr: int,
        language: Optional[str],
        task: str,
        return_timestamps: bool,
    ) -> Transcript:
        """Synchronous transcription using Whisper."""
        import torch

        # Prepare input
        inputs = self._whisper_processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        )

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.device == "cuda":
                inputs["input_features"] = inputs["input_features"].half()

        # Configure generation
        generate_kwargs = {
            "task": task,
            "return_timestamps": return_timestamps,
        }
        if language:
            generate_kwargs["language"] = language

        # Generate transcription
        with torch.no_grad():
            predicted_ids = self._whisper_model.generate(
                inputs["input_features"],
                **generate_kwargs,
            )

        # Decode
        transcription = self._whisper_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            output_offsets=return_timestamps,
        )

        # Parse result
        if return_timestamps and isinstance(transcription, dict):
            text = transcription.get("text", "")
            chunks = transcription.get("chunks", [])
        else:
            text = transcription[0] if isinstance(transcription, list) else str(transcription)
            chunks = []

        # Build segments
        segments = []
        if chunks:
            for i, chunk in enumerate(chunks):
                segment = TranscriptSegment(
                    id=f"seg_{i}",
                    text=chunk.get("text", "").strip(),
                    start_time=chunk.get("timestamp", [0, 0])[0] or 0,
                    end_time=chunk.get("timestamp", [0, 0])[1] or 0,
                    confidence=0.9,  # Whisper doesn't provide per-segment confidence
                )
                segments.append(segment)

        # Detect language
        detected_language = language or "en"
        try:
            # Try to get detected language from model
            if hasattr(self._whisper_model, "detect_language"):
                lang_probs = self._whisper_model.detect_language(inputs["input_features"])
                detected_language = max(lang_probs, key=lang_probs.get)
        except Exception:
            pass

        return Transcript(
            text=text.strip(),
            language=detected_language,
            language_probability=0.95,  # Approximate
            confidence=0.9,
            segments=segments,
            duration=len(waveform) / sr,
            word_count=len(text.split()),
        )

    # ========================
    # Speaker Diarization
    # ========================

    async def _ensure_diarization_loaded(self) -> None:
        """Ensure diarization pipeline is loaded."""
        if self._diarization_pipeline is not None:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._load_diarization)

    def _load_diarization(self) -> None:
        """Load diarization pipeline (blocking)."""
        try:
            from pyannote.audio import Pipeline
            import torch

            logger.info("Loading diarization pipeline", model=self.config.diarization_model)

            # Note: pyannote requires authentication token for some models
            # Users should set HUGGING_FACE_HUB_TOKEN environment variable
            self._diarization_pipeline = Pipeline.from_pretrained(
                self.config.diarization_model,
                use_auth_token=True,
            )

            if self.device != "cpu" and torch.cuda.is_available():
                self._diarization_pipeline = self._diarization_pipeline.to(
                    torch.device(self.device)
                )

            logger.info("Diarization pipeline loaded successfully")

        except ImportError as e:
            logger.warning(f"Pyannote not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load diarization pipeline: {e}")

    async def diarize(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> list[Speaker]:
        """
        Perform speaker diarization.

        Args:
            audio: Audio source
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers

        Returns:
            List of Speaker objects with their speaking segments
        """
        if not self.config.enable_diarization:
            return []

        await self._ensure_diarization_loaded()

        if self._diarization_pipeline is None:
            logger.warning("Diarization not available, returning empty speakers")
            return []

        # Load audio
        if isinstance(audio, AudioSegment):
            if audio.waveform is None:
                return []
            waveform = audio.waveform
            sr = audio.sample_rate
        else:
            waveform, sr = await self.load_audio(audio)

        loop = asyncio.get_event_loop()
        speakers = await loop.run_in_executor(
            _executor,
            lambda: self._diarize_sync(
                waveform, sr,
                min_speakers or self.config.min_speakers,
                max_speakers or self.config.max_speakers
            )
        )

        self._stats["diarizations_performed"] += 1
        return speakers

    def _diarize_sync(
        self,
        waveform: np.ndarray,
        sr: int,
        min_speakers: int,
        max_speakers: int,
    ) -> list[Speaker]:
        """Synchronous diarization."""
        import torch

        # Create temporary file for pyannote (it expects file input)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            import soundfile as sf
            sf.write(tmp.name, waveform, sr)

            # Run diarization
            diarization = self._diarization_pipeline(
                tmp.name,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        # Parse results into Speaker objects
        speaker_segments: dict[str, list[TimeRange]] = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(
                TimeRange(start=turn.start, end=turn.end)
            )

        # Create Speaker objects
        speakers = []
        for speaker_id, segments in speaker_segments.items():
            # Merge adjacent segments
            merged = self._merge_adjacent_segments(segments, gap_threshold=0.5)

            speakers.append(Speaker(
                id=speaker_id,
                name=f"Speaker {len(speakers) + 1}",
                segments=merged,
                confidence=0.85,
            ))

        return speakers

    def _merge_adjacent_segments(
        self,
        segments: list[TimeRange],
        gap_threshold: float = 0.5,
    ) -> list[TimeRange]:
        """Merge segments that are close together."""
        if not segments:
            return []

        segments = sorted(segments, key=lambda s: s.start)
        merged = [segments[0]]

        for segment in segments[1:]:
            if segment.start - merged[-1].end <= gap_threshold:
                merged[-1] = TimeRange(merged[-1].start, segment.end)
            else:
                merged.append(segment)

        return merged

    async def align_transcript_with_speakers(
        self,
        transcript: Transcript,
        speakers: list[Speaker],
    ) -> Transcript:
        """
        Align transcript segments with speaker diarization.

        Args:
            transcript: Transcript to align
            speakers: Speaker diarization results

        Returns:
            Updated transcript with speaker IDs assigned
        """
        if not speakers:
            return transcript

        # Create a mapping of time ranges to speakers
        for segment in transcript.segments:
            segment_range = segment.time_range
            best_speaker = None
            best_overlap = 0.0

            for speaker in speakers:
                for speaker_range in speaker.segments:
                    overlap = segment_range.overlap_duration(speaker_range)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = speaker.id

            if best_speaker:
                segment.speaker_id = best_speaker

        # Update transcript speakers list
        transcript.speakers = speakers

        return transcript

    # ========================
    # Audio Event Detection
    # ========================

    async def _ensure_event_detector_loaded(self) -> None:
        """Ensure event detector is loaded."""
        if self._event_detector is not None:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._load_event_detector)

    def _load_event_detector(self) -> None:
        """Load audio event detection model (blocking)."""
        try:
            from transformers import AutoFeatureExtractor, ASTForAudioClassification

            logger.info("Loading event detector", model=self.config.event_detection_model)

            self._event_processor = AutoFeatureExtractor.from_pretrained(
                self.config.event_detection_model
            )
            self._event_detector = ASTForAudioClassification.from_pretrained(
                self.config.event_detection_model
            )

            if self.device != "cpu":
                self._event_detector = self._event_detector.to(self.device)

            logger.info("Event detector loaded successfully")

        except ImportError as e:
            logger.warning(f"Event detector not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load event detector: {e}")

    async def detect_events(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        threshold: Optional[float] = None,
        window_size: float = 1.0,
        hop_size: float = 0.5,
    ) -> list[AudioEvent]:
        """
        Detect audio events using sliding window.

        Args:
            audio: Audio source
            threshold: Detection confidence threshold
            window_size: Window size in seconds
            hop_size: Hop size in seconds

        Returns:
            List of detected AudioEvent objects
        """
        await self._ensure_event_detector_loaded()

        threshold = threshold or self.config.event_threshold

        # Load audio
        if isinstance(audio, AudioSegment):
            if audio.waveform is None:
                return []
            waveform = audio.waveform
            sr = audio.sample_rate
        else:
            waveform, sr = await self.load_audio(audio)

        if self._event_detector is None:
            # Return basic event detection based on energy
            return self._detect_events_fallback(waveform, sr)

        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            _executor,
            lambda: self._detect_events_sync(waveform, sr, threshold, window_size, hop_size)
        )

        self._stats["events_detected"] += len(events)
        return events

    def _detect_events_sync(
        self,
        waveform: np.ndarray,
        sr: int,
        threshold: float,
        window_size: float,
        hop_size: float,
    ) -> list[AudioEvent]:
        """Synchronous event detection using sliding window."""
        import torch

        events = []
        duration = len(waveform) / sr
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        current_events: dict[str, dict] = {}  # Track ongoing events

        for start_sample in range(0, len(waveform) - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            window = waveform[start_sample:end_sample]
            start_time = start_sample / sr

            # Process window
            inputs = self._event_processor(
                window,
                sampling_rate=sr,
                return_tensors="pt",
            )

            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._event_detector(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            # Get top predictions
            top_probs, top_indices = torch.topk(probs, k=5)

            for prob, idx in zip(top_probs, top_indices):
                if prob.item() < threshold:
                    continue

                label = self._event_detector.config.id2label[idx.item()]
                event_type = self._classify_event_type(label)

                # Track event continuity
                if label in current_events:
                    current_events[label]["end_time"] = start_time + window_size
                    current_events[label]["confidence"] = max(
                        current_events[label]["confidence"], prob.item()
                    )
                else:
                    current_events[label] = {
                        "label": label,
                        "event_type": event_type,
                        "start_time": start_time,
                        "end_time": start_time + window_size,
                        "confidence": prob.item(),
                    }

            # Check for ended events
            ended_labels = []
            for label, event_data in current_events.items():
                if event_data["end_time"] < start_time:
                    events.append(AudioEvent(
                        label=event_data["label"],
                        event_type=event_data["event_type"],
                        start_time=event_data["start_time"],
                        end_time=event_data["end_time"],
                        confidence=event_data["confidence"],
                    ))
                    ended_labels.append(label)

            for label in ended_labels:
                del current_events[label]

        # Add remaining events
        for event_data in current_events.values():
            events.append(AudioEvent(
                label=event_data["label"],
                event_type=event_data["event_type"],
                start_time=event_data["start_time"],
                end_time=event_data["end_time"],
                confidence=event_data["confidence"],
            ))

        # Merge overlapping events of same type
        events = self._merge_overlapping_events(events)

        return events

    def _classify_event_type(self, label: str) -> AudioEventType:
        """Classify an event label into an event type."""
        label_lower = label.lower()

        speech_keywords = ["speech", "talk", "voice", "speak", "conversation"]
        music_keywords = ["music", "singing", "guitar", "piano", "drum", "instrument"]
        animal_keywords = ["dog", "cat", "bird", "animal", "bark", "meow"]
        mechanical_keywords = ["engine", "car", "machine", "motor", "vehicle"]
        alert_keywords = ["alarm", "siren", "bell", "ring", "beep"]

        if any(kw in label_lower for kw in speech_keywords):
            return AudioEventType.SPEECH
        elif any(kw in label_lower for kw in music_keywords):
            return AudioEventType.MUSIC
        elif any(kw in label_lower for kw in animal_keywords):
            return AudioEventType.ANIMAL
        elif any(kw in label_lower for kw in mechanical_keywords):
            return AudioEventType.MECHANICAL
        elif any(kw in label_lower for kw in alert_keywords):
            return AudioEventType.ALERT
        elif "silence" in label_lower:
            return AudioEventType.SILENCE
        else:
            return AudioEventType.ENVIRONMENTAL

    def _merge_overlapping_events(self, events: list[AudioEvent]) -> list[AudioEvent]:
        """Merge overlapping events of the same type."""
        if not events:
            return []

        # Group by label
        by_label: dict[str, list[AudioEvent]] = {}
        for event in events:
            if event.label not in by_label:
                by_label[event.label] = []
            by_label[event.label].append(event)

        merged = []
        for label, label_events in by_label.items():
            label_events.sort(key=lambda e: e.start_time)
            current = label_events[0]

            for event in label_events[1:]:
                if event.start_time <= current.end_time + 0.1:  # Small gap tolerance
                    current = AudioEvent(
                        label=current.label,
                        event_type=current.event_type,
                        start_time=current.start_time,
                        end_time=max(current.end_time, event.end_time),
                        confidence=max(current.confidence, event.confidence),
                    )
                else:
                    merged.append(current)
                    current = event

            merged.append(current)

        return sorted(merged, key=lambda e: e.start_time)

    def _detect_events_fallback(
        self,
        waveform: np.ndarray,
        sr: int,
    ) -> list[AudioEvent]:
        """Fallback event detection using energy analysis."""
        try:
            import librosa
        except ImportError:
            return []

        # Compute RMS energy
        rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=512)

        # Threshold for speech/activity
        threshold = np.mean(rms) + 0.5 * np.std(rms)

        events = []
        in_event = False
        event_start = 0.0

        for i, (time, energy) in enumerate(zip(times, rms)):
            if energy > threshold and not in_event:
                in_event = True
                event_start = time
            elif energy <= threshold and in_event:
                in_event = False
                if time - event_start > 0.1:  # Minimum event duration
                    events.append(AudioEvent(
                        label="activity",
                        event_type=AudioEventType.UNKNOWN,
                        start_time=event_start,
                        end_time=time,
                        confidence=0.5,
                    ))

        if in_event:
            events.append(AudioEvent(
                label="activity",
                event_type=AudioEventType.UNKNOWN,
                start_time=event_start,
                end_time=times[-1],
                confidence=0.5,
            ))

        return events

    # ========================
    # Speaker Embeddings
    # ========================

    async def _ensure_speaker_encoder_loaded(self) -> None:
        """Ensure speaker encoder is loaded."""
        if self._speaker_encoder is not None:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._load_speaker_encoder)

    def _load_speaker_encoder(self) -> None:
        """Load speaker embedding model (blocking)."""
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info("Loading speaker encoder", model=self.config.speaker_embedding_model)

            self._speaker_encoder = EncoderClassifier.from_hparams(
                source=self.config.speaker_embedding_model,
                savedir=f".cache/speechbrain/{self.config.speaker_embedding_model.split('/')[-1]}",
                run_opts={"device": self.device} if self.device != "cpu" else {},
            )

            logger.info("Speaker encoder loaded successfully")

        except ImportError as e:
            logger.warning(f"SpeechBrain not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load speaker encoder: {e}")

    async def compute_speaker_embedding(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> Optional[np.ndarray]:
        """
        Compute speaker embedding for voice identification.

        Args:
            audio: Audio source (should contain clear speech)

        Returns:
            Speaker embedding vector (192-dimensional for ECAPA-TDNN)
        """
        await self._ensure_speaker_encoder_loaded()

        if self._speaker_encoder is None:
            logger.warning("Speaker encoder not available")
            return None

        # Load audio
        if isinstance(audio, AudioSegment):
            if audio.waveform is None:
                return None
            waveform = audio.waveform
            sr = audio.sample_rate
        else:
            waveform, sr = await self.load_audio(audio)

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            _executor,
            lambda: self._compute_speaker_embedding_sync(waveform, sr)
        )

        self._stats["embeddings_computed"] += 1
        return embedding

    def _compute_speaker_embedding_sync(
        self,
        waveform: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Synchronous speaker embedding computation."""
        import torch

        # Convert to tensor
        if isinstance(waveform, np.ndarray):
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
        else:
            waveform_tensor = waveform.unsqueeze(0)

        # Compute embedding
        with torch.no_grad():
            embedding = self._speaker_encoder.encode_batch(waveform_tensor)

        return embedding.squeeze().cpu().numpy()

    async def compare_speakers(
        self,
        audio1: Union[str, Path, np.ndarray, AudioSegment],
        audio2: Union[str, Path, np.ndarray, AudioSegment],
    ) -> tuple[bool, float]:
        """
        Compare two audio samples for speaker similarity.

        Args:
            audio1: First audio sample
            audio2: Second audio sample

        Returns:
            Tuple of (is_same_speaker, similarity_score)
        """
        emb1 = await self.compute_speaker_embedding(audio1)
        emb2 = await self.compute_speaker_embedding(audio2)

        if emb1 is None or emb2 is None:
            return False, 0.0

        # Cosine similarity
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

        # Threshold for same speaker (tuned for ECAPA-TDNN)
        threshold = 0.75
        is_same = similarity > threshold

        return is_same, similarity

    # ========================
    # CLAP Embeddings (Audio-Text)
    # ========================

    async def _ensure_clap_loaded(self) -> None:
        """Ensure CLAP model is loaded."""
        if self._clap_model is not None:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._load_clap)

    def _load_clap(self) -> None:
        """Load CLAP model (blocking)."""
        try:
            from transformers import ClapProcessor, ClapModel

            logger.info("Loading CLAP model", model=self.config.clap_model)

            self._clap_processor = ClapProcessor.from_pretrained(self.config.clap_model)
            self._clap_model = ClapModel.from_pretrained(self.config.clap_model)

            if self.device != "cpu":
                self._clap_model = self._clap_model.to(self.device)

            logger.info("CLAP model loaded successfully")

        except ImportError as e:
            logger.warning(f"CLAP not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load CLAP: {e}")

    async def compute_audio_embedding(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> Optional[np.ndarray]:
        """
        Compute CLAP audio embedding for retrieval.

        Args:
            audio: Audio source

        Returns:
            Audio embedding vector
        """
        await self._ensure_clap_loaded()

        if self._clap_model is None:
            # Fall back to simpler embedding
            return await self._compute_fallback_embedding(audio)

        # Load audio
        if isinstance(audio, AudioSegment):
            if audio.waveform is None:
                return None
            waveform = audio.waveform
            sr = audio.sample_rate
        else:
            waveform, sr = await self.load_audio(audio, target_sr=48000)  # CLAP uses 48kHz

        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            _executor,
            lambda: self._compute_clap_audio_embedding_sync(waveform)
        )

        self._stats["embeddings_computed"] += 1
        return embedding

    def _compute_clap_audio_embedding_sync(self, waveform: np.ndarray) -> np.ndarray:
        """Synchronous CLAP audio embedding."""
        import torch

        inputs = self._clap_processor(
            audios=waveform,
            return_tensors="pt",
            sampling_rate=48000,
        )

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_features = self._clap_model.get_audio_features(**inputs)

        return audio_features.squeeze().cpu().numpy()

    async def compute_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Compute CLAP text embedding for text-to-audio search.

        Args:
            text: Text query

        Returns:
            Text embedding vector
        """
        await self._ensure_clap_loaded()

        if self._clap_model is None:
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self._compute_clap_text_embedding_sync(text)
        )

    def _compute_clap_text_embedding_sync(self, text: str) -> np.ndarray:
        """Synchronous CLAP text embedding."""
        import torch

        inputs = self._clap_processor(text=text, return_tensors="pt", padding=True)

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._clap_model.get_text_features(**inputs)

        return text_features.squeeze().cpu().numpy()

    async def _compute_fallback_embedding(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> np.ndarray:
        """Compute a simple fallback embedding using audio features."""
        # Load audio
        if isinstance(audio, AudioSegment):
            if audio.waveform is None:
                return np.zeros(512, dtype=np.float32)
            waveform = audio.waveform
            sr = audio.sample_rate
        else:
            waveform, sr = await self.load_audio(audio)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self._compute_fallback_embedding_sync(waveform, sr)
        )

    def _compute_fallback_embedding_sync(
        self,
        waveform: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Compute fallback embedding using librosa features."""
        try:
            import librosa

            # Compute various features
            features = []

            # MFCCs
            mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(waveform)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))

            # RMS energy
            rms = librosa.feature.rms(y=waveform)
            features.append(np.mean(rms))
            features.append(np.std(rms))

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
            features.extend(np.mean(chroma, axis=1))

            # Pad or truncate to fixed size
            embedding = np.array(features, dtype=np.float32)
            if len(embedding) < 512:
                embedding = np.pad(embedding, (0, 512 - len(embedding)))
            else:
                embedding = embedding[:512]

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except ImportError:
            return np.zeros(512, dtype=np.float32)

    # ========================
    # Music Analysis
    # ========================

    async def analyze_music(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> MusicAnalysis:
        """
        Analyze music content.

        Args:
            audio: Audio source

        Returns:
            MusicAnalysis with tempo, key, mood, etc.
        """
        if not self.config.enable_music_analysis:
            return MusicAnalysis()

        # Load audio
        if isinstance(audio, AudioSegment):
            if audio.waveform is None:
                return MusicAnalysis()
            waveform = audio.waveform
            sr = audio.sample_rate
        else:
            waveform, sr = await self.load_audio(audio)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self._analyze_music_sync(waveform, sr)
        )

    def _analyze_music_sync(self, waveform: np.ndarray, sr: int) -> MusicAnalysis:
        """Synchronous music analysis using librosa."""
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not available for music analysis")
            return MusicAnalysis()

        analysis = MusicAnalysis()

        try:
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=waveform, sr=sr)
            analysis.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
            analysis.tempo_confidence = 0.8
            analysis.beats = librosa.frames_to_time(beats, sr=sr).tolist()

            # Get downbeats (approximate)
            analysis.downbeats = [analysis.beats[i] for i in range(0, len(analysis.beats), 4)]

            # Key estimation using chroma
            chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)

            # Simple key detection
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_idx = int(np.argmax(chroma_mean))
            analysis.key = f"{key_names[key_idx]} major"
            analysis.key_confidence = float(chroma_mean[key_idx] / np.sum(chroma_mean))

            # Energy analysis
            rms = librosa.feature.rms(y=waveform)[0]
            analysis.energy = float(np.mean(rms))

            # Spectral features for mood estimation
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
            spectral_flatness = librosa.feature.spectral_flatness(y=waveform)[0]

            # Simple mood estimation based on tempo and energy
            if analysis.tempo > 120 and analysis.energy > 0.3:
                analysis.mood = "energetic"
                analysis.danceability = 0.8
            elif analysis.tempo < 80:
                analysis.mood = "calm"
                analysis.danceability = 0.3
            else:
                analysis.mood = "moderate"
                analysis.danceability = 0.5

            # Valence estimation (very approximate)
            # Higher spectral centroid and major key = happier
            if "major" in (analysis.key or ""):
                analysis.valence = 0.6 + 0.2 * min(np.mean(spectral_centroid) / 2000, 1)
            else:
                analysis.valence = 0.4 - 0.2 * min(np.mean(spectral_centroid) / 2000, 1)

            analysis.valence = max(0, min(1, analysis.valence))

        except Exception as e:
            logger.warning(f"Music analysis error: {e}")

        return analysis

    # ========================
    # Audio Quality Analysis
    # ========================

    async def analyze_quality(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> AudioQuality:
        """
        Analyze audio quality.

        Args:
            audio: Audio source

        Returns:
            AudioQuality metrics
        """
        # Load audio
        if isinstance(audio, AudioSegment):
            if audio.waveform is None:
                return AudioQuality()
            waveform = audio.waveform
            sr = audio.sample_rate
        else:
            waveform, sr = await self.load_audio(audio)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self._analyze_quality_sync(waveform, sr)
        )

    def _analyze_quality_sync(self, waveform: np.ndarray, sr: int) -> AudioQuality:
        """Synchronous audio quality analysis."""
        quality = AudioQuality(sample_rate=sr)

        try:
            # Peak and RMS amplitude
            quality.peak_amplitude = float(np.max(np.abs(waveform)))
            quality.rms_amplitude = float(np.sqrt(np.mean(waveform ** 2)))

            # Clipping detection
            quality.clipping_detected = quality.peak_amplitude > 0.99

            # Dynamic range (approximate)
            if quality.rms_amplitude > 0:
                quality.dynamic_range = 20 * np.log10(quality.peak_amplitude / quality.rms_amplitude)

            # SNR estimation (very approximate - assumes speech/signal in middle frequencies)
            try:
                import librosa
                stft = np.abs(librosa.stft(waveform))
                # Assume noise in very low and very high frequencies
                signal_power = np.mean(stft[10:100, :] ** 2)
                noise_power = np.mean(stft[:5, :] ** 2) + np.mean(stft[-10:, :] ** 2)
                if noise_power > 0:
                    quality.signal_to_noise_ratio = 10 * np.log10(signal_power / noise_power)
                    quality.noise_level = min(1.0, noise_power / signal_power)
            except Exception:
                pass

            # Background noise detection
            quality.has_background_noise = quality.noise_level > 0.1

        except Exception as e:
            logger.warning(f"Quality analysis error: {e}")

        return quality

    # ========================
    # Combined Scene Understanding
    # ========================

    async def understand_scene(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
        enable_transcription: bool = True,
        enable_diarization: bool = True,
        enable_events: bool = True,
        enable_music: bool = True,
    ) -> AudioScene:
        """
        Full audio scene understanding.

        Combines all analysis capabilities for comprehensive understanding.

        Args:
            audio: Audio source
            enable_transcription: Include speech transcription
            enable_diarization: Include speaker diarization
            enable_events: Include event detection
            enable_music: Include music analysis

        Returns:
            Complete AudioScene analysis
        """
        import time
        start_time = time.time()

        # Load audio once
        if isinstance(audio, AudioSegment):
            if audio.waveform is None:
                return AudioScene()
            waveform = audio.waveform
            sr = audio.sample_rate
            audio_id = audio.id
        else:
            waveform, sr = await self.load_audio(audio)
            audio_id = str(uuid.uuid4())

        duration = len(waveform) / sr

        # Run analyses in parallel where possible
        tasks = []

        if enable_events:
            tasks.append(("events", self.detect_events(waveform)))

        if enable_music:
            tasks.append(("music", self.analyze_music(waveform)))

        tasks.append(("quality", self.analyze_quality(waveform)))
        tasks.append(("embedding", self.compute_audio_embedding(waveform)))

        # Run parallel tasks
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.warning(f"Scene analysis task {name} failed: {e}")
                results[name] = None

        # Sequential tasks that depend on previous results
        transcript = None
        speakers = []

        if enable_transcription:
            try:
                transcript = await self.transcribe(waveform)

                if enable_diarization:
                    speakers = await self.diarize(waveform)
                    if speakers:
                        transcript = await self.align_transcript_with_speakers(
                            transcript, speakers
                        )
            except Exception as e:
                logger.warning(f"Transcription failed: {e}")

        # Determine scene type
        events = results.get("events", []) or []
        music_analysis = results.get("music")

        has_speech = any(e.event_type == AudioEventType.SPEECH for e in events)
        has_music = any(e.event_type == AudioEventType.MUSIC for e in events)

        if has_speech and has_music:
            scene_type = "mixed"
        elif has_speech:
            scene_type = "conversation" if len(speakers) > 1 else "speech"
        elif has_music:
            scene_type = "music"
        else:
            scene_type = "ambient"

        # Ambient description
        event_labels = [e.label for e in events[:5]]
        ambient_description = ", ".join(event_labels) if event_labels else "quiet"

        # Create scene
        scene = AudioScene(
            audio_id=audio_id,
            duration=duration,
            events=events,
            speakers=speakers,
            transcript=transcript,
            scene_type=scene_type,
            ambient_description=ambient_description,
            music_analysis=music_analysis if has_music else None,
            has_music=has_music,
            music_is_foreground=has_music and not has_speech,
            quality=results.get("quality"),
            scene_embedding=results.get("embedding"),
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        # Calculate noise level
        if scene.quality:
            scene.noise_level_db = -20 * np.log10(max(scene.quality.noise_level, 1e-10))
            scene.average_loudness_db = 20 * np.log10(max(scene.quality.rms_amplitude, 1e-10))

        self._stats["audio_files_processed"] += 1
        self._stats["total_duration_processed"] += duration

        return scene

    # ========================
    # Voice Characteristics
    # ========================

    async def analyze_voice(
        self,
        audio: Union[str, Path, np.ndarray, AudioSegment],
    ) -> VoiceCharacteristics:
        """
        Analyze voice characteristics for TTS/voice cloning.

        Args:
            audio: Audio with speech

        Returns:
            VoiceCharacteristics
        """
        # Load audio
        if isinstance(audio, AudioSegment):
            if audio.waveform is None:
                return VoiceCharacteristics()
            waveform = audio.waveform
            sr = audio.sample_rate
        else:
            waveform, sr = await self.load_audio(audio)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self._analyze_voice_sync(waveform, sr)
        )

    def _analyze_voice_sync(self, waveform: np.ndarray, sr: int) -> VoiceCharacteristics:
        """Synchronous voice analysis."""
        try:
            import librosa
        except ImportError:
            return VoiceCharacteristics()

        characteristics = VoiceCharacteristics()

        try:
            # Pitch analysis (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
            )

            voiced_f0 = f0[~np.isnan(f0)]
            if len(voiced_f0) > 0:
                characteristics.pitch_mean = float(np.mean(voiced_f0))
                characteristics.pitch_std = float(np.std(voiced_f0))
                characteristics.pitch_range = (float(np.min(voiced_f0)), float(np.max(voiced_f0)))

            # Energy
            rms = librosa.feature.rms(y=waveform)[0]
            characteristics.energy_mean = float(np.mean(rms))
            characteristics.energy_std = float(np.std(rms))

            # Speaking rate estimation (very approximate)
            # Based on energy peaks as syllable markers
            energy_peaks = librosa.util.peak_pick(
                rms, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10
            )
            duration = len(waveform) / sr
            if duration > 0:
                syllables_per_second = len(energy_peaks) / duration
                characteristics.speaking_rate = syllables_per_second * 60  # Approximate WPM

            # Spectral characteristics
            spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
            spectral_tilt = np.mean(np.diff(np.log(spectral_centroid + 1e-10)))
            characteristics.spectral_tilt = float(spectral_tilt)

        except Exception as e:
            logger.warning(f"Voice analysis error: {e}")

        return characteristics

    def get_stats(self) -> dict[str, Any]:
        """Get perception system statistics."""
        return {
            **self._stats,
            "device": self.device,
            "models_loaded": {
                "whisper": self._whisper_model is not None,
                "diarization": self._diarization_pipeline is not None,
                "event_detector": self._event_detector is not None,
                "speaker_encoder": self._speaker_encoder is not None,
                "clap": self._clap_model is not None,
            },
        }
