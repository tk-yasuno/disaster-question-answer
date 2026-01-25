"""
災害質問応答パイプライン - RAGベース実装

機能:
- FAISS semantic search
- BERT-base日本語モデルによる抽出型QA
- WhatIf/WhatAct分類
- 用語正規化との統合
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering, 
    pipeline, BertTokenizer, BertForQuestionAnswering
)
import torch
import logging
from dataclasses import dataclass
from enum import Enum

from .glossary import GlossaryManager
from .data_processor import DisasterDataProcessor
from ..utils import load_config

logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """質問タイプ分類"""
    WHAT_IF = "WhatIf"      # 状況分析
    WHAT_ACT = "WhatAct"    # 行動提案  
    WHAT_IS = "WhatIs"      # 事実質問
    HOW_TO = "HowTo"        # 手順質問
    GENERAL = "General"     # 一般質問

@dataclass
class QAResult:
    """QA結果データクラス"""
    answer: str
    confidence: float
    question_type: QuestionType
    disaster_category: str
    sources: List[Dict]
    normalized_query: str
    raw_query: str

class DisasterQAPipeline:
    """
    災害質問応答パイプライン
    
    統合機能:
    - 用語正規化 (GlossaryManager)
    - 文書検索 (DisasterDataProcessor + FAISS)
    - 抽出型QA (BERT Japanese)
    - 質問分類・回答整形
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.qa_model_name = self.config['qa_model']
        self.max_seq_length = self.config['max_seq_length']
        self.similarity_threshold = self.config['retrieval']['similarity_threshold']
        self.top_k = self.config['retrieval']['top_k']
        
        # デバイス設定
        self.device = torch.device(self.config.get('device', 'cpu'))
        logger.info(f"Using device: {self.device}")
        
        # モジュール初期化
        self._init_components()
        
    def _init_components(self):
        """各コンポーネントの初期化"""
        
        # 1. 用語辞書
        logger.info("Initializing Glossary Manager...")
        self.glossary = GlossaryManager(self.config)
        
        # 2. データ処理器
        logger.info("Initializing Data Processor...")
        self.data_processor = DisasterDataProcessor(self.config)
        
        # 3. QAモデル
        logger.info(f"Loading QA model: {self.qa_model_name}")
        self._load_qa_model()
        
        # 4. 質問分類器
        self._init_question_classifier()
    
    def _load_qa_model(self):
        """BERT-base日本語QAモデル読み込み"""
        try:
            self.qa_tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
            self.qa_model.to(self.device)
            
            # pipeline作成
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.qa_model,
                tokenizer=self.qa_tokenizer,
                device=0 if self.device.type == 'cuda' else -1
            )
            
            logger.info("QA model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load QA model: {e}")
            # フォールバック: 軽量モデル
            self.qa_pipeline = pipeline("question-answering", 
                                      model="deepset/bert-base-cased-squad2")
            logger.warning("Using fallback English QA model")
    
    def _init_question_classifier(self):
        """質問分類器の初期化"""
        
        # キーワードベース分類器（MVP版）
        self.question_patterns = {
            QuestionType.WHAT_IF: [
                r'もし.*たら', r'.*場合', r'.*とき', r'.*状況', r'.*scenario',
                r'if.*', r'when.*', r'.*なら'
            ],
            QuestionType.WHAT_ACT: [
                r'どうすれば', r'何をすべき', r'どう行動', r'対処法', r'対応方法',
                r'what.*should', r'how.*to', r'何をする', r'action'
            ],
            QuestionType.WHAT_IS: [
                r'とは', r'とは何', r'どんな', r'.*について', r'definition',
                r'what.*is', r'explain'
            ],
            QuestionType.HOW_TO: [
                r'どのように', r'やり方', r'方法', r'手順', r'procedure',
                r'how.*to', r'steps'
            ]
        }
    
    def answer_question(self, 
                       question: str, 
                       language: str = 'auto',
                       top_k: int = None) -> QAResult:
        """
        質問に対する回答を生成
        
        Args:
            question: ユーザーの質問
            language: 言語 ('ja', 'en', 'auto')
            top_k: 検索する文書数
        
        Returns:
            QAResult: 回答結果
        """
        
        if not question.strip():
            return self._create_empty_result(question, "質問が空です")
        
        try:
            # 1. 用語正規化
            normalization_result = self.glossary.normalize_query(question, language)
            normalized_query = normalization_result['normalized_query']
            disaster_category = self.glossary.categorize_query(question)
            
            logger.info(f"Normalized query: {normalized_query}")
            logger.info(f"Disaster category: {disaster_category}")
            
            # 2. 質問タイプ分類
            question_type = self._classify_question(question)
            logger.info(f"Question type: {question_type}")
            
            # 3. 関連文書検索
            top_k = top_k or self.top_k
            relevant_chunks = self._retrieve_relevant_documents(
                normalized_query, top_k)
            
            if not relevant_chunks:
                return self._create_empty_result(
                    question, 
                    "関連する情報が見つかりませんでした", 
                    question_type, 
                    disaster_category
                )
            
            # 4. 抽出型QA実行
            qa_result = self._extract_answer(question, relevant_chunks)
            
            # 5. 結果整形
            final_result = self._format_final_answer(
                qa_result,
                question,
                normalized_query,
                question_type,
                disaster_category,
                relevant_chunks
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"QA pipeline error: {e}")
            return self._create_empty_result(
                question, 
                f"エラーが発生しました: {str(e)}"
            )
    
    def _classify_question(self, question: str) -> QuestionType:
        """質問をタイプ別に分類"""
        
        question_lower = question.lower()
        
        for qtype, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return qtype
        
        return QuestionType.GENERAL
    
    def _retrieve_relevant_documents(self, query: str, top_k: int) -> List[Dict]:
        """関連文書を検索"""
        
        try:
            # FAISS検索
            similar_chunks = self.data_processor.search_similar_chunks(query, top_k)
            
            # 閾値フィルタリング
            filtered_chunks = [
                chunk for chunk in similar_chunks
                if chunk['similarity'] >= self.similarity_threshold
            ]
            
            logger.info(f"Found {len(filtered_chunks)} relevant chunks")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return []
    
    def _extract_answer(self, question: str, contexts: List[Dict]) -> Dict:
        """複数コンテキストから最適回答を抽出"""
        
        best_answer = None
        best_score = 0.0
        
        for context in contexts[:3]:  # 上位3件で試行
            try:
                context_text = context['text']
                
                # トークン制限チェック
                if len(context_text) > self.max_seq_length * 4:  # 概算
                    context_text = context_text[:self.max_seq_length * 4]
                
                # QA実行
                qa_input = {
                    "question": question,
                    "context": context_text
                }
                
                result = self.qa_pipeline(qa_input)
                
                if result['score'] > best_score:
                    best_answer = result
                    best_answer['source_chunk'] = context
                    best_score = result['score']
                
            except Exception as e:
                logger.warning(f"QA extraction failed for chunk: {e}")
                continue
        
        return best_answer or {"answer": "回答を生成できませんでした", "score": 0.0}
    
    def _format_final_answer(self, 
                           qa_result: Dict,
                           original_question: str,
                           normalized_query: str, 
                           question_type: QuestionType,
                           disaster_category: str,
                           sources: List[Dict]) -> QAResult:
        """最終回答をフォーマット"""
        
        # 基本回答
        answer = qa_result.get('answer', '').strip()
        confidence = qa_result.get('score', 0.0)
        
        # 質問タイプ別の後処理
        if question_type == QuestionType.WHAT_ACT and answer:
            answer = self._enhance_action_answer(answer, disaster_category)
        elif question_type == QuestionType.WHAT_IF and answer:
            answer = self._enhance_situation_answer(answer, disaster_category)
        
        # ソース情報
        source_info = []
        for source in sources[:3]:
            source_info.append({
                'document': source.get('document_filename', 'unknown'),
                'title': source.get('document_title', ''),
                'disaster_type': source.get('disaster_type', ''),
                'relevance': source.get('similarity', 0.0),
                'chunk_id': source.get('chunk_id', 0)
            })
        
        return QAResult(
            answer=answer,
            confidence=confidence,
            question_type=question_type,
            disaster_category=disaster_category,
            sources=source_info,
            normalized_query=normalized_query,
            raw_query=original_question
        )
    
    def _enhance_action_answer(self, answer: str, disaster_category: str) -> str:
        """行動提案回答を強化"""
        
        action_templates = {
            'earthquake': ['直ちに安全な場所に避難する', '余震に注意する', '火の元を確認する'],
            'tsunami': ['高台または津波避難ビルに避難する', '海岸から離れる', 'ラジオで情報を確認する'],
            'typhoon': ['屋内に避難する', '窓を補強する', '停電に備える'],
            'flood': ['高い場所に避難する', '地下を避ける', '増水情報を確認する']
        }
        
        if disaster_category in action_templates:
            suggested_actions = action_templates[disaster_category]
            answer += "\\n\\n追加の推奨行動:\\n" + "\\n".join(f"• {action}" for action in suggested_actions)
        
        return answer
    
    def _enhance_situation_answer(self, answer: str, disaster_category: str) -> str:
        """状況分析回答を強化"""
        
        answer += f"\\n\\n[{disaster_category}災害の一般的な状況分析に基づいています]"
        return answer
    
    def _create_empty_result(self, 
                           question: str, 
                           message: str = "回答できませんでした",
                           question_type: QuestionType = QuestionType.GENERAL,
                           disaster_category: str = "general") -> QAResult:
        """空の結果を作成"""
        
        return QAResult(
            answer=message,
            confidence=0.0,
            question_type=question_type,
            disaster_category=disaster_category,
            sources=[],
            normalized_query=question,
            raw_query=question
        )
    
    def batch_answer(self, questions: List[str]) -> List[QAResult]:
        """バッチ処理で複数質問に回答"""
        
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)
        
        return results
    
    def get_pipeline_info(self) -> Dict:
        """パイプライン情報を取得"""
        
        return {
            "qa_model": self.qa_model_name,
            "embedding_model": self.config['embedding_model'],
            "device": str(self.device),
            "glossary_terms": len(self.glossary.terms_df) if self.glossary.terms_df is not None else 0,
            "indexed_documents": getattr(self.data_processor.faiss_index, 'ntotal', 0),
            "similarity_threshold": self.similarity_threshold,
            "max_seq_length": self.max_seq_length
        }


def main():
    """CLI実行用メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Disaster QA Pipeline CLI')
    parser.add_argument('--question', type=str, 
                      help='Question to answer')
    parser.add_argument('--batch-file', type=str,
                      help='File with questions (one per line)')
    parser.add_argument('--info', action='store_true',
                      help='Show pipeline information')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # パイプライン初期化
    logger.info("Initializing QA Pipeline...")
    pipeline = DisasterQAPipeline()
    
    if args.info:
        info = pipeline.get_pipeline_info()
        print("🔍 Pipeline Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    if args.question:
        logger.info(f"Processing question: {args.question}")
        result = pipeline.answer_question(args.question)
        
        print(f"\\n❓ Question: {result.raw_query}")
        print(f"🔄 Normalized: {result.normalized_query}")
        print(f"📝 Answer: {result.answer}")
        print(f"🎯 Type: {result.question_type.value}")
        print(f"🏷️ Category: {result.disaster_category}")
        print(f"📊 Confidence: {result.confidence:.3f}")
        print(f"📚 Sources: {len(result.sources)} documents")
    
    if args.batch_file:
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(questions)} questions from file")
        results = pipeline.batch_answer(questions)
        
        for i, result in enumerate(results, 1):
            print(f"\\n--- Question {i} ---")
            print(f"Q: {result.raw_query}")
            print(f"A: {result.answer}")
            print(f"Confidence: {result.confidence:.3f}")


if __name__ == '__main__':
    main()