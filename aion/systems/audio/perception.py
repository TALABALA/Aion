"""
AION Audio Perception

Multi-model audio processing system with:
- Speech recognition (Whisper)
- Speaker diarization (pyannote)
- Speaker embeddings (SpeechBrain)
- Audio event detection (Audio Spectrogram Transformer)
- Music analysis (librosa/essentia)
- Audio-text embeddings (CLAP)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Union
from pathlib import Path
import tempfile

import numpy as np
import structlog

from aion.systems.audio.models import (
    AudioSegment,
    Transcript,
    TranscriptSegment,
    Word,
    Speaker,
    AudioEvent,
    MusicAnalysis,
    TimeRange,
    FrequencyRange,
    VoiceProfile,
)

logger = structlog.get_logger(__name__)


# Audio event categories and labels
AUDIO_EVENT_CATEGORIES = {
    "speech": ["speech", "male_speech", "female_speech", "child_speech", "narration"],
    "music": ["music", "singing", "musical_instrument", "guitar", "piano", "drums"],
    "ambient": ["silence", "noise", "white_noise", "wind", "rain", "thunder", "water"],
    "action": [
        "footsteps", "door", "knock", "clap", "cough", "laughter", "crying",
        "applause", "glass_breaking", "car_horn", "siren", "alarm", "bell",
        "keyboard_typing", "phone_ringing", "dog_bark", "cat_meow", "bird",
    ],
    "vehicle": ["engine", "car", "motorcycle", "airplane", "helicopter", "train"],
}

# AudioSet label to category mapping
LABEL_TO_CATEGORY = {}
for category, labels in AUDIO_EVENT_CATEGORIES.items():
    for label in labels:
        LABEL_TO_CATEGORY[label.lower()] = category


class AudioPerception:
    """
    AION Audio Perception System

    Processes audio using multiple models:
    - Whisper for speech recognition
    - Pyannote for speaker diarization
    - SpeechBrain for speaker embeddings
    - Audio Spectrogram Transformer for event detection
    - CLAP for audio-text embeddings
    """

    def __init__(
        self,
        whisper_model: str = "openai/whisper-large-v3",
        speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        event_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        clap_model: str = "laion/larger_clap_general",
        device: str = "auto",
        sample_rate: int = 16000,
    ):
        self.whisper_model_name = whisper_model
        self.speaker_model_name = speaker_model
        self.event_model_name = event_model
        self.clap_model_name = clap_model
        self.sample_rate = sample_rate

        # Device selection
        if device == "auto":
            self.device = self._get_device()
        else:
            self.device = device

        # Models (lazy loaded)
        self._whisper_processor = None
        self._whisper_model = None
        self._speaker_model = None
        self._event_processor = None
        self._event_model = None
        self._clap_processor = None
        self._clap_model = None
        self._diarization_pipeline = None

        # Model availability flags
        self._whisper_available = False
        self._speaker_available = False
        self._event_available = False
        self._clap_available = False
        self._diarization_available = False

        self._initialized = False

    @staticmethod
    def _get_device() -> str:
        """Determine best available device."""
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

        logger.info("Initializing Audio Perception System")

        # Initialize models in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models)

        self._initialized = True
        logger.info(
            "Audio Perception System initialized",
            whisper=self._whisper_available,
            speaker=self._speaker_available,
            events=self._event_available,
            clap=self._clap_available,
            diarization=self._diarization_available,
            device=self.device,
        )

    def _load_models(self) -> None:
        """Load audio models (blocking)."""
        # Load Whisper for speech recognition
        self._load_whisper()

        # Load speaker embedding model
        self._load_speaker_model()

        # Load audio event detection model
        self._load_event_model()

        # Load CLAP for audio-text embeddings
        self._load_clap()

        # Load diarization pipeline
        self._load_diarization()

    def _load_whisper(self) -> None:
        """Load Whisper model."""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            import torch

            logger.debug("Loading Whisper model", model=self.whisper_model_name)

            self._whisper_processor = WhisperProcessor.from_pretrained(
                self.whisper_model_name
            )
            self._whisper_model = WhisperForConditionalGeneration.from_pretrained(
                self.whisper_model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            self._whisper_model.to(self.device)
            self._whisper_available = True

            logger.info("Whisper model loaded successfully")

        except ImportError as e:
            logger.warning(f"Whisper not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load Whisper: {e}")

    def _load_speaker_model(self) -> None:
        """Load speaker embedding model."""
        try:
            from speechbrain.inference import EncoderClassifier

            logger.debug("Loading speaker model", model=self.speaker_model_name)

            self._speaker_model = EncoderClassifier.from_hparams(
                source=self.speaker_model_name,
                run_opts={"device": self.device},
            )
            self._speaker_available = True

            logger.info("Speaker model loaded successfully")

        except ImportError as e:
            logger.warning(f"SpeechBrain not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load speaker model: {e}")

    def _load_event_model(self) -> None:
        """Load audio event detection model."""
        try:
            from transformers import AutoFeatureExtractor, ASTForAudioClassification
            import torch

            logger.debug("Loading event detection model", model=self.event_model_name)

            self._event_processor = AutoFeatureExtractor.from_pretrained(
                self.event_model_name
            )
            self._event_model = ASTForAudioClassification.from_pretrained(
                self.event_model_name
            )
            self._event_model.to(self.device)
            self._event_available = True

            logger.info("Audio event model loaded successfully")

        except ImportError as e:
            logger.warning(f"Audio event model not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load event model: {e}")

    def _load_clap(self) -> None:
        """Load CLAP model for audio-text embeddings."""
        try:
            from transformers import ClapProcessor, ClapModel

            logger.debug("Loading CLAP model", model=self.clap_model_name)

            self._clap_processor = ClapProcessor.from_pretrained(self.clap_model_name)
            self._clap_model = ClapModel.from_pretrained(self.clap_model_name)
            self._clap_model.to(self.device)
            self._clap_available = True

            logger.info("CLAP model loaded successfully")

        except ImportError as e:
            logger.warning(f"CLAP not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load CLAP: {e}")

    def _load_diarization(self) -> None:
        """Load speaker diarization pipeline."""
        try:
            from pyannote.audio import Pipeline
            import os

            # Pyannote requires HuggingFace token for some models
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

            logger.debug("Loading diarization pipeline")

            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            if self.device != "cpu":
                import torch
                self._diarization_pipeline.to(torch.device(self.device))

            self._diarization_available = True
            logger.info("Diarization pipeline loaded successfully")

        except ImportError as e:
            logger.warning(f"Pyannote not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to load diarization: {e}")

    async def shutdown(self) -> None:
        """Shutdown the perception system."""
        self._whisper_model = None
        self._whisper_processor = None
        self._speaker_model = None
        self._event_model = None
        self._event_processor = None
        self._clap_model = None
        self._clap_processor = None
        self._diarization_pipeline = None
        self._initialized = False
        logger.info("Audio Perception System shutdown")

    # ==================== Audio Loading ====================

    async def load_audio(
        self,
        source: Union[str, Path, bytes, np.ndarray],
        target_sr: Optional[int] = None,
        mono: bool = True,
    ) -> tuple[np.ndarray, int]:
        """
        Load and preprocess audio from various sources.

        Args:
            source: Audio file path, bytes, or numpy array
            target_sr: Target sample rate (defaults to self.sample_rate)
            mono: Convert to mono

        Returns:
            Tuple of (waveform, sample_rate)
        """
        target_sr = target_sr or self.sample_rate

        if isinstance(source, np.ndarray):
            waveform = source
            sr = target_sr  # Assume target SR if numpy array
        else:
            loop = asyncio.get_event_loop()
            waveform, sr = await loop.run_in_executor(
                None,
                lambda: self._load_audio_sync(source, target_sr, mono),
            )

        return waveform, sr

    def _load_audio_sync(
        self,
        source: Union[str, Path, bytes],
        target_sr: int,
        mono: bool,
    ) -> tuple[np.ndarray, int]:
        """Synchronous audio loading."""
        try:
            import librosa

            if isinstance(source, bytes):
                # Load from bytes
                audio_file = io.BytesIO(source)
                waveform, sr = librosa.load(audio_file, sr=target_sr, mono=mono)
            else:
                # Load from file path
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"Audio file not found: {path}")
                waveform, sr = librosa.load(str(path), sr=target_sr, mono=mono)

            return waveform, sr

        except ImportError:
            logger.warning("librosa not available, using soundfile")
            import soundfile as sf

            if isinstance(source, bytes):
                audio_file = io.BytesIO(source)
                waveform, sr = sf.read(audio_file)
            else:
                waveform, sr = sf.read(str(source))

            # Resample if needed
            if sr != target_sr:
                import scipy.signal
                waveform = scipy.signal.resample(
                    waveform,
                    int(len(waveform) * target_sr / sr)
                )
                sr = target_sr

            # Convert to mono if needed
            if mono and waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            return waveform, sr

    def compute_audio_hash(self, waveform: np.ndarray) -> str:
        """Compute a hash for audio deduplication."""
        return hashlib.sha256(waveform.tobytes()).hexdigest()[:16]

    # ==================== Speech Recognition ====================

    async def transcribe(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = None,
        return_timestamps: bool = True,
    ) -> Transcript:
        """
        Transcribe speech to text.

        Args:
            audio: Audio source
            language: Language code (auto-detect if None)
            return_timestamps: Include word-level timestamps

        Returns:
            Transcript with segments and timing
        """
        if not self._initialized:
            await self.initialize()

        waveform, sr = await self.load_audio(audio)

        if not self._whisper_available:
            return self._mock_transcript(waveform, sr)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(waveform, sr, language, return_timestamps),
        )

    def _transcribe_sync(
        self,
        waveform: np.ndarray,
        sr: int,
        language: Optional[str],
        return_timestamps: bool,
    ) -> Transcript:
        """Synchronous transcription."""
        import torch

        # Prepare inputs
        inputs = self._whisper_processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with timestamps
        generate_kwargs = {
            "return_timestamps": return_timestamps,
            "task": "transcribe",
        }
        if language:
            generate_kwargs["language"] = language

        with torch.no_grad():
            outputs = self._whisper_model.generate(
                **inputs,
                **generate_kwargs,
            )

        # Decode with timestamps
        if return_timestamps:
            result = self._whisper_processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                output_offsets=True,
            )[0]
        else:
            result = {
                "text": self._whisper_processor.decode(outputs[0], skip_special_tokens=True),
                "offsets": [],
            }

        # Build transcript
        transcript_id = str(uuid.uuid4())
        duration = len(waveform) / sr

        segments = []
        if isinstance(result, dict) and "offsets" in result:
            for i, offset in enumerate(result.get("offsets", [])):
                seg_id = f"{transcript_id}_seg_{i}"
                segments.append(TranscriptSegment(
                    id=seg_id,
                    text=offset.get("text", "").strip(),
                    start_time=offset.get("timestamp", (0, 0))[0] or 0,
                    end_time=offset.get("timestamp", (0, duration))[1] or duration,
                    confidence=0.9,  # Whisper doesn't provide per-segment confidence
                ))

        full_text = result.get("text", "") if isinstance(result, dict) else str(result)

        # Detect language
        detected_lang = language or "en"
        if hasattr(self._whisper_model, "detect_language"):
            try:
                detected_lang = self._whisper_model.detect_language(inputs["input_features"])[0]
            except Exception:
                pass

        return Transcript(
            id=transcript_id,
            text=full_text.strip(),
            language=detected_lang,
            confidence=0.9,
            segments=segments,
            speakers=[],  # Will be filled by diarization
            audio_id=self.compute_audio_hash(waveform),
            duration=duration,
        )

    def _mock_transcript(self, waveform: np.ndarray, sr: int) -> Transcript:
        """Return mock transcript when Whisper is unavailable."""
        duration = len(waveform) / sr
        return Transcript(
            id=str(uuid.uuid4()),
            text="[Transcription unavailable - Whisper model not loaded]",
            language="en",
            confidence=0.0,
            segments=[],
            speakers=[],
            audio_id=self.compute_audio_hash(waveform),
            duration=duration,
        )

    # ==================== Speaker Diarization ====================

    async def diarize(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        num_speakers: Optional[int] = None,
    ) -> list[Speaker]:
        """
        Perform speaker diarization.

        Args:
            audio: Audio source
            num_speakers: Expected number of speakers (auto-detect if None)

        Returns:
            List of identified speakers with their speaking segments
        """
        if not self._initialized:
            await self.initialize()

        if not self._diarization_available:
            logger.warning("Diarization not available")
            return []

        waveform, sr = await self.load_audio(audio)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._diarize_sync(waveform, sr, num_speakers),
        )

    def _diarize_sync(
        self,
        waveform: np.ndarray,
        sr: int,
        num_speakers: Optional[int],
    ) -> list[Speaker]:
        """Synchronous diarization."""
        import torch

        # Create temporary file for pyannote
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import soundfile as sf
            sf.write(f.name, waveform, sr)
            temp_path = f.name

        try:
            # Run diarization
            kwargs = {}
            if num_speakers:
                kwargs["num_speakers"] = num_speakers

            diarization = self._diarization_pipeline(temp_path, **kwargs)

            # Extract speakers
            speakers = {}
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                if speaker_label not in speakers:
                    speakers[speaker_label] = Speaker(
                        id=f"speaker_{len(speakers)}",
                        name=speaker_label,
                    )
                speakers[speaker_label].add_segment(turn.start, turn.end)

            return list(speakers.values())

        finally:
            import os
            os.unlink(temp_path)

    async def transcribe_with_diarization(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        language: Optional[str] = None,
    ) -> Transcript:
        """
        Transcribe with speaker diarization.

        Args:
            audio: Audio source
            language: Language code (auto-detect if None)

        Returns:
            Transcript with speaker labels
        """
        # Load audio once
        waveform, sr = await self.load_audio(audio)

        # Run transcription and diarization in parallel
        transcript_task = asyncio.create_task(
            self.transcribe(waveform, language=language)
        )
        diarization_task = asyncio.create_task(self.diarize(waveform))

        transcript, speakers = await asyncio.gather(
            transcript_task, diarization_task
        )

        # Assign speakers to segments
        if speakers:
            for segment in transcript.segments:
                seg_range = segment.time_range
                best_speaker = None
                best_overlap = 0.0

                for speaker in speakers:
                    for spk_range in speaker.segments:
                        intersection = seg_range.intersection(spk_range)
                        if intersection and intersection.duration > best_overlap:
                            best_overlap = intersection.duration
                            best_speaker = speaker

                if best_speaker:
                    segment.speaker_id = best_speaker.id

            transcript.speakers = speakers

        return transcript

    # ==================== Speaker Embeddings ====================

    async def get_speaker_embedding(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
    ) -> np.ndarray:
        """
        Extract speaker embedding from audio.

        Args:
            audio: Audio source (should contain single speaker)

        Returns:
            Speaker embedding vector
        """
        if not self._initialized:
            await self.initialize()

        waveform, sr = await self.load_audio(audio)

        if not self._speaker_available:
            # Return random embedding as fallback
            return np.random.randn(192).astype(np.float32)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._get_speaker_embedding_sync(waveform, sr),
        )

    def _get_speaker_embedding_sync(
        self,
        waveform: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Synchronous speaker embedding extraction."""
        import torch

        # SpeechBrain expects torch tensor
        signal = torch.from_numpy(waveform).unsqueeze(0)

        # Extract embedding
        with torch.no_grad():
            embedding = self._speaker_model.encode_batch(signal)

        return embedding.squeeze().cpu().numpy()

    async def identify_speaker(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        known_speakers: list[VoiceProfile],
        threshold: float = 0.7,
    ) -> tuple[Optional[VoiceProfile], float]:
        """
        Identify speaker against known profiles.

        Args:
            audio: Audio source
            known_speakers: List of known voice profiles
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (matched_profile, similarity) or (None, 0.0)
        """
        if not known_speakers:
            return None, 0.0

        embedding = await self.get_speaker_embedding(audio)

        best_match = None
        best_similarity = 0.0

        for profile in known_speakers:
            similarity = profile.similarity(embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = profile

        if best_similarity >= threshold:
            return best_match, best_similarity

        return None, best_similarity

    async def verify_speaker(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        claimed_profile: VoiceProfile,
        threshold: float = 0.75,
    ) -> tuple[bool, float]:
        """
        Verify if audio matches claimed speaker profile.

        Args:
            audio: Audio source
            claimed_profile: The speaker profile to verify against
            threshold: Verification threshold

        Returns:
            Tuple of (is_verified, similarity)
        """
        embedding = await self.get_speaker_embedding(audio)
        similarity = claimed_profile.similarity(embedding)
        return similarity >= threshold, similarity

    # ==================== Audio Event Detection ====================

    async def detect_events(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        threshold: float = 0.3,
        top_k: int = 10,
    ) -> list[AudioEvent]:
        """
        Detect audio events using Audio Spectrogram Transformer.

        Args:
            audio: Audio source
            threshold: Detection confidence threshold
            top_k: Maximum number of events per segment

        Returns:
            List of detected audio events
        """
        if not self._initialized:
            await self.initialize()

        waveform, sr = await self.load_audio(audio)

        if not self._event_available:
            return self._mock_events(waveform, sr)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._detect_events_sync(waveform, sr, threshold, top_k),
        )

    def _detect_events_sync(
        self,
        waveform: np.ndarray,
        sr: int,
        threshold: float,
        top_k: int,
    ) -> list[AudioEvent]:
        """Synchronous event detection."""
        import torch

        events = []
        duration = len(waveform) / sr

        # Process in 10-second chunks with 2-second overlap
        chunk_duration = 10.0
        overlap = 2.0
        chunk_samples = int(chunk_duration * sr)
        hop_samples = int((chunk_duration - overlap) * sr)

        for start_sample in range(0, len(waveform), hop_samples):
            end_sample = min(start_sample + chunk_samples, len(waveform))
            chunk = waveform[start_sample:end_sample]

            if len(chunk) < sr:  # Skip very short chunks
                continue

            # Prepare input
            inputs = self._event_processor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self._event_model(**inputs)

            probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

            # Get top-k predictions above threshold
            indices = np.where(probs > threshold)[0]
            for idx in indices[:top_k]:
                label = self._event_model.config.id2label[idx]
                confidence = float(probs[idx])

                category = LABEL_TO_CATEGORY.get(label.lower(), "other")

                events.append(AudioEvent(
                    id=f"event_{len(events)}",
                    label=label,
                    category=category,
                    start_time=start_sample / sr,
                    end_time=end_sample / sr,
                    confidence=confidence,
                ))

        # Merge overlapping events with same label
        events = self._merge_events(events)

        return events

    def _merge_events(self, events: list[AudioEvent]) -> list[AudioEvent]:
        """Merge overlapping events with same label."""
        if not events:
            return events

        # Group by label
        by_label: dict[str, list[AudioEvent]] = {}
        for event in events:
            if event.label not in by_label:
                by_label[event.label] = []
            by_label[event.label].append(event)

        merged = []
        for label, label_events in by_label.items():
            # Sort by start time
            label_events.sort(key=lambda e: e.start_time)

            current = label_events[0]
            for event in label_events[1:]:
                if event.start_time <= current.end_time:
                    # Merge
                    current = AudioEvent(
                        id=current.id,
                        label=current.label,
                        category=current.category,
                        start_time=current.start_time,
                        end_time=max(current.end_time, event.end_time),
                        confidence=max(current.confidence, event.confidence),
                    )
                else:
                    merged.append(current)
                    current = event

            merged.append(current)

        # Sort by start time
        merged.sort(key=lambda e: e.start_time)
        return merged

    def _mock_events(self, waveform: np.ndarray, sr: int) -> list[AudioEvent]:
        """Return mock events when model is unavailable."""
        duration = len(waveform) / sr

        # Detect if there's significant audio content
        rms = np.sqrt(np.mean(waveform ** 2))

        events = []
        if rms > 0.01:
            events.append(AudioEvent(
                id="event_0",
                label="audio_content",
                category="ambient",
                start_time=0,
                end_time=duration,
                confidence=0.5,
            ))

        return events

    # ==================== Audio Embeddings (CLAP) ====================

    async def get_audio_embedding(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
    ) -> np.ndarray:
        """
        Get audio embedding using CLAP.

        Args:
            audio: Audio source

        Returns:
            Audio embedding vector
        """
        if not self._initialized:
            await self.initialize()

        waveform, sr = await self.load_audio(audio)

        if not self._clap_available:
            # Return hash-based embedding as fallback
            return self._compute_fallback_embedding(waveform)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._get_audio_embedding_sync(waveform, sr),
        )

    def _get_audio_embedding_sync(
        self,
        waveform: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Synchronous CLAP embedding extraction."""
        import torch

        inputs = self._clap_processor(
            audios=waveform,
            sampling_rate=sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_features = self._clap_model.get_audio_features(**inputs)

        return audio_features.squeeze().cpu().numpy()

    async def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding using CLAP.

        Args:
            text: Text to embed

        Returns:
            Text embedding vector
        """
        if not self._initialized:
            await self.initialize()

        if not self._clap_available:
            # Return hash-based embedding as fallback
            return self._compute_text_fallback_embedding(text)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._get_text_embedding_sync(text),
        )

    def _get_text_embedding_sync(self, text: str) -> np.ndarray:
        """Synchronous text embedding."""
        import torch

        inputs = self._clap_processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._clap_model.get_text_features(**inputs)

        return text_features.squeeze().cpu().numpy()

    def _compute_fallback_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """Compute simple spectral embedding as fallback."""
        # Use simple spectral features
        embedding = np.zeros(512, dtype=np.float32)

        # RMS energy
        rms = np.sqrt(np.mean(waveform ** 2))
        embedding[0] = rms

        # Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(waveform))))
        embedding[1] = zcr

        # Spectral centroid (rough approximation)
        if len(waveform) > 0:
            fft = np.abs(np.fft.rfft(waveform))
            freqs = np.fft.rfftfreq(len(waveform))
            centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)
            embedding[2] = centroid

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _compute_text_fallback_embedding(self, text: str) -> np.ndarray:
        """Compute simple text embedding as fallback."""
        embedding = np.zeros(512, dtype=np.float32)

        words = text.lower().split()
        for i, word in enumerate(words[:100]):
            idx = hash(word) % 512
            embedding[idx] += 1.0 / (i + 1)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    # ==================== Music Analysis ====================

    async def analyze_music(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
    ) -> Optional[MusicAnalysis]:
        """
        Analyze musical features.

        Args:
            audio: Audio source

        Returns:
            MusicAnalysis or None if not music
        """
        if not self._initialized:
            await self.initialize()

        waveform, sr = await self.load_audio(audio)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._analyze_music_sync(waveform, sr),
        )

    def _analyze_music_sync(
        self,
        waveform: np.ndarray,
        sr: int,
    ) -> Optional[MusicAnalysis]:
        """Synchronous music analysis."""
        try:
            import librosa

            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
            tempo_bpm = float(tempo) if not hasattr(tempo, '__iter__') else float(tempo[0])

            # Key estimation using chroma features
            chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr)
            key_idx = int(np.argmax(np.mean(chroma, axis=1)))
            keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            key = keys[key_idx]

            # Mode estimation (major/minor)
            # Simplified: use energy ratio of major vs minor third
            major_third_energy = np.mean(chroma[4])  # E for C major
            minor_third_energy = np.mean(chroma[3])  # Eb for C minor
            mode = "major" if major_third_energy > minor_third_energy else "minor"

            # Spectral features for mood/energy
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr))
            rms = np.mean(librosa.feature.rms(y=waveform))

            # Normalize to 0-1 scale
            energy = min(1.0, rms * 10)
            brightness = min(1.0, spectral_centroid / 4000)

            # Estimate mood based on features
            if energy > 0.6 and tempo_bpm > 120:
                mood = "energetic"
            elif energy < 0.3 and tempo_bpm < 80:
                mood = "calm"
            elif mode == "minor" and energy < 0.5:
                mood = "melancholic"
            elif mode == "major" and energy > 0.4:
                mood = "happy"
            else:
                mood = "neutral"

            return MusicAnalysis(
                tempo_bpm=tempo_bpm,
                key=key,
                mode=mode,
                time_signature="4/4",  # Would need more analysis
                mood=mood,
                energy=energy,
                danceability=min(1.0, tempo_bpm / 180),
                instrumentalness=0.5,  # Would need vocal detection
                valence=0.5 + 0.5 * (1 if mode == "major" else -1) * energy,
                confidence=0.7,
            )

        except ImportError:
            logger.warning("librosa not available for music analysis")
            return None
        except Exception as e:
            logger.warning(f"Music analysis failed: {e}")
            return None

    # ==================== Utilities ====================

    async def compute_spectrogram(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        n_mels: int = 128,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Compute mel spectrogram."""
        waveform, sr = await self.load_audio(audio)

        try:
            import librosa

            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=sr,
                n_mels=n_mels,
                hop_length=hop_length,
            )
            return librosa.power_to_db(mel_spec, ref=np.max)

        except ImportError:
            # Simple FFT-based spectrogram
            n_fft = 2048
            hop = hop_length

            num_frames = (len(waveform) - n_fft) // hop + 1
            spec = np.zeros((n_fft // 2 + 1, num_frames))

            for i in range(num_frames):
                start = i * hop
                frame = waveform[start:start + n_fft]
                spec[:, i] = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))

            return 20 * np.log10(spec + 1e-8)

    def get_capabilities(self) -> dict[str, bool]:
        """Get available capabilities."""
        return {
            "transcription": self._whisper_available,
            "diarization": self._diarization_available,
            "speaker_embedding": self._speaker_available,
            "event_detection": self._event_available,
            "audio_embedding": self._clap_available,
            "music_analysis": True,  # librosa-based, always available
        }

    def get_stats(self) -> dict[str, Any]:
        """Get perception system statistics."""
        return {
            "initialized": self._initialized,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "capabilities": self.get_capabilities(),
        }
