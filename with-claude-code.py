from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Iterator
import logging
import os
from abc import ABC, abstractmethod

import whisper
from pydub import AudioSegment
from docx import Document
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Counter, Histogram

# Metrics
PROCESSING_TIME = Histogram("audio_processing_seconds", "Time spent processing audio")
CHUNKS_PROCESSED = Counter("audio_chunks_processed", "Number of audio chunks processed")
ERRORS = Counter("processing_errors", "Number of processing errors", ["type"])

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats"""

    MP3 = "mp3"
    MP4 = "mp4"
    WAV = "wav"


@dataclass
class TranscriptionConfig:
    """Configuration for transcription service"""

    chunk_length_ms: int = 10 * 60 * 1000  # 10 minutes
    whisper_model: str = "medium.en"
    device: str = "cpu"
    output_format: str = "docx"
    max_retries: int = 3
    line_length: int = 80
    combine_keywords: List[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TranscriptionConfig":
        """Load configuration from YAML file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class TranscriptionFormatter:
    """Handles formatting of transcription text"""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.combine_keywords = config.combine_keywords or [
            "Yes",
            "Yep",
            "Perfect",
            "Okay",
            "Fantastic",
            "All right",
            "Cool",
        ]

    def format_text(self, text: str) -> str:
        """Format transcription text for readability"""
        from textwrap import fill

        sentences = text.replace("\n", " ").split(". ")
        formatted_lines = []
        buffer = ""

        for sentence in sentences:
            sentence = sentence.strip()

            # Combine short responses with previous sentence
            if any(sentence.startswith(keyword) for keyword in self.combine_keywords):
                if buffer:
                    buffer = f"{buffer} {sentence}"
                else:
                    buffer = sentence
            else:
                if buffer:
                    formatted_lines.append(fill(buffer, width=self.config.line_length))
                    buffer = ""
                formatted_lines.append(fill(sentence, width=self.config.line_length))

        if buffer:
            formatted_lines.append(fill(buffer, width=self.config.line_length))

        return "\n\n".join(formatted_lines)


class AudioProcessor(ABC):
    """Abstract base class for audio processing"""

    @abstractmethod
    def process(self, audio: AudioSegment) -> AudioSegment:
        pass


class NormalizeAudioProcessor(AudioProcessor):
    """Normalizes audio volume"""

    def process(self, audio: AudioSegment) -> AudioSegment:
        return audio.normalize()


class AudioChunker:
    """Handles splitting audio into chunks"""

    def __init__(self, config: TranscriptionConfig):
        self.chunk_length_ms = config.chunk_length_ms

    def split(self, audio: AudioSegment) -> Iterator[AudioSegment]:
        """Split audio into chunks"""
        for i in range(0, len(audio), self.chunk_length_ms):
            chunk = audio[i : i + self.chunk_length_ms]
            CHUNKS_PROCESSED.inc()
            yield chunk


class WhisperTranscriber:
    """Handles transcription using Whisper"""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.model = self._load_model()
        self.formatter = TranscriptionFormatter(config)

    def _load_model(self) -> whisper.Whisper:
        """Load Whisper model with retry logic"""

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )
        def load() -> whisper.Whisper:
            logger.info(f"Loading Whisper model {self.config.whisper_model}...")
            return whisper.load_model(
                self.config.whisper_model, device=self.config.device
            )

        return load()

    @retry(stop=stop_after_attempt(3))
    def transcribe_chunk(self, chunk: AudioSegment, chunk_index: int) -> str:
        """Transcribe audio chunk with retry logic"""
        temp_path = Path(f"temp_chunk_{chunk_index}.wav")
        try:
            chunk.export(temp_path, format="wav")
            result = self.model.transcribe(str(temp_path))
            return self.formatter.format_text(result["text"])
        finally:
            temp_path.unlink(missing_ok=True)


class DocumentWriter(ABC):
    """Abstract base class for document writing"""

    @abstractmethod
    def write(self, text: str, output_path: Path) -> None:
        pass


class DocxWriter(DocumentWriter):
    """Writes transcription to DOCX format"""

    def write(self, text: str, output_path: Path) -> None:
        doc = Document()
        doc.add_paragraph(text)
        doc.save(output_path)


class TranscriptionService:
    """Main service class for audio transcription"""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.audio_processor = NormalizeAudioProcessor()
        self.chunker = AudioChunker(config)
        self.transcriber = WhisperTranscriber(config)
        self.writer = DocxWriter()

    def process_file(self, input_path: Path, output_dir: Path) -> Optional[Path]:
        """Process a single audio file"""
        with PROCESSING_TIME.time():
            try:
                logger.info(f"Processing file: {input_path}")

                # Load and normalize audio
                audio = AudioSegment.from_file(input_path)
                audio = self.audio_processor.process(audio)

                # Process chunks
                transcriptions = []
                for i, chunk in enumerate(self.chunker.split(audio)):
                    logger.info(f"Processing chunk {i+1}")
                    trans = self.transcriber.transcribe_chunk(chunk, i)
                    transcriptions.append(f"[Chunk {i+1}]\n{trans}")

                # Write output
                full_text = "\n\n".join(transcriptions)
                output_path = (
                    output_dir / f"{input_path.stem}.{self.config.output_format}"
                )
                self.writer.write(full_text, output_path)

                logger.info(f"Successfully processed {input_path}")
                return output_path

            except Exception as e:
                logger.error(f"Error processing {input_path}: {str(e)}")
                ERRORS.labels(type=type(e).__name__).inc()
                raise


def main():
    """Main entry point"""
    try:
        # Use default configuration
        config = TranscriptionConfig()  # Use default configuration
        service = TranscriptionService(config)

        # Define input and output directories
        input_dir = Path(r"C:\Users\Jonathan\Documents\kbdocs\text_result")
        output_dir = Path(r"C:\Users\Jonathan\Documents\kbdocs\text_result")
        output_dir.mkdir(exist_ok=True)  # Ensure output directory exists

        # Process all audio files in the input directory
        for file_path in input_dir.glob("*.*"):
            if file_path.suffix[1:].lower() in [f.value for f in AudioFormat]:
                service.process_file(file_path, output_dir)  # Process each file

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()


# PATH를 동적으로 처리 그니까 documents/kbdocs에 있는거 documents/kbdocs.text_result말고
# def main():
#     """Main entry point"""
#     try:
#         # Load configuration
#         config = TranscriptionConfig.from_yaml(Path("config.yaml"))
#         service = TranscriptionService(config)

#         # Process files
#         input_dir = Path(os.getenv("INPUT_DIR", "input"))
#         output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
#         output_dir.mkdir(exist_ok=True)

#         for file_path in input_dir.glob("*.*"):
#             if file_path.suffix[1:] in [f.value for f in AudioFormat]:
#                 service.process_file(file_path, output_dir)

#     except Exception as e:
#         logger.error(f"Fatal error: {str(e)}")
#         raise


# if __name__ == "__main__":
#     main()
