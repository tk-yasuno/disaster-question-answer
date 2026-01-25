"""
Disaster QA Pipeline - MVP Implementation
災害質問応答パイプライン MVP版

This package implements a minimum viable product for disaster-related question answering
using Japanese disaster documentation and multilingual language models.

主要モジュール:
- glossary: 用語理解・正規化モジュール
- data_processor: PDF文書処理・前処理モジュール  
- qa_pipeline: RAGベース質問応答パイプライン
- fine_tuning: LoRAファインチューニングモジュール
- api: FastAPI RESTエンドポイント
"""

__version__ = "0.1.0"
__author__ = "Disaster QA Team"

from .modules.glossary import GlossaryManager
from .modules.data_processor import DisasterDataProcessor
from .modules.qa_pipeline import DisasterQAPipeline
from .modules.fine_tuning import LoRATrainer  # 有効化

__all__ = [
    "GlossaryManager",
    "DisasterDataProcessor",
    "DisasterQAPipeline",
    "LoRATrainer",  # 有効化
]