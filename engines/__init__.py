"""
EVA Engine Package.

Contains the main processing engines:
- Audio processing
- Speech recognition
- Emotion intelligence
- Memory
- Emotion fusion
- LLM
- RAG
"""

from .audio_engine import AudioEngine
from .speech_engine import SpeechEngine
from .emotion_engine import EmotionEngine
from .memory_engine import MemoryEngine
from .fusion import confidence_weighted_fusion
from .llm_engine import LLMEngine
from .rag_engine import RAGEngine
from .tts_engine import TTSEngine
