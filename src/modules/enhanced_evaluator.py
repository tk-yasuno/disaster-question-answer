#!/usr/bin/env python3
"""
v0.2: 評価指標実装 (F1スコア、BLEUスコア、Exact Match)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from transformers import EvalPrediction
import evaluate
from collections import Counter
import re


class DisasterQAEvaluator:
    """災害QA評価指標実装クラス"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bleu_metric = evaluate.load("bleu")
        
    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """包括的な評価メトリクスを計算"""
        
        predictions, labels = eval_pred
        
        # 予測結果の処理
        if isinstance(predictions, tuple):
            start_predictions, end_predictions = predictions
        else:
            start_predictions = predictions[0]
            end_predictions = predictions[1]
            
        # ラベルの処理  
        if isinstance(labels, tuple):
            start_labels, end_labels = labels[0], labels[1]
        else:
            start_labels = labels[:, 0] if labels.ndim > 1 else labels
            end_labels = labels[:, 1] if labels.ndim > 1 else labels
        
        # 予測位置を取得
        start_pred_positions = np.argmax(start_predictions, axis=-1)
        end_pred_positions = np.argmax(end_predictions, axis=-1)
        
        # 各種メトリクス計算
        metrics = {}
        
        # 1. Position-based accuracy
        start_accuracy = np.mean(start_pred_positions == start_labels)
        end_accuracy = np.mean(end_pred_positions == end_labels)
        metrics["start_position_accuracy"] = float(start_accuracy)
        metrics["end_position_accuracy"] = float(end_accuracy)
        
        # 2. Joint position accuracy (both start and end correct)
        joint_accuracy = np.mean((start_pred_positions == start_labels) & 
                                (end_pred_positions == end_labels))
        metrics["joint_position_accuracy"] = float(joint_accuracy)
        
        # 3. Span-based F1 score
        span_f1_scores = []
        exact_matches = []
        
        for i in range(len(start_pred_positions)):
            pred_start = start_pred_positions[i]
            pred_end = end_pred_positions[i]
            true_start = start_labels[i]
            true_end = end_labels[i]
            
            # 予測スパンと真のスパンの重複を計算
            pred_span = set(range(pred_start, min(pred_end + 1, len(self.tokenizer.decode([1])))))
            true_span = set(range(true_start, true_end + 1))
            
            if len(pred_span) == 0 and len(true_span) == 0:
                f1 = 1.0
                exact = 1.0
            elif len(pred_span) == 0 or len(true_span) == 0:
                f1 = 0.0
                exact = 0.0
            else:
                overlap = len(pred_span.intersection(true_span))
                precision = overlap / len(pred_span) if len(pred_span) > 0 else 0
                recall = overlap / len(true_span) if len(true_span) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                exact = 1.0 if pred_span == true_span else 0.0
            
            span_f1_scores.append(f1)
            exact_matches.append(exact)
        
        metrics["span_f1"] = float(np.mean(span_f1_scores))
        metrics["exact_match"] = float(np.mean(exact_matches))
        
        # 4. Overall F1 (weighted average)
        metrics["overall_f1"] = float((metrics["span_f1"] + metrics["joint_position_accuracy"]) / 2)
        
        return metrics
    
    def compute_text_metrics(self, 
                           predictions: List[str], 
                           references: List[str]) -> Dict[str, float]:
        """テキストベースの評価メトリクス"""
        
        metrics = {}
        
        # BLEU score calculation
        try:
            # BLEUスコア用にリファレンスを適切な形式に変換
            references_bleu = [[ref.split()] for ref in references]
            predictions_bleu = [pred.split() for pred in predictions]
            
            bleu_result = self.bleu_metric.compute(
                predictions=predictions_bleu, 
                references=references_bleu
            )
            metrics["bleu"] = float(bleu_result["bleu"])
        except Exception as e:
            print(f"BLEU calculation failed: {e}")
            metrics["bleu"] = 0.0
        
        # Exact Match (text-based)
        exact_matches = []
        for pred, ref in zip(predictions, references):
            pred_normalized = self._normalize_text(pred)
            ref_normalized = self._normalize_text(ref)
            exact_matches.append(1.0 if pred_normalized == ref_normalized else 0.0)
        
        metrics["text_exact_match"] = float(np.mean(exact_matches))
        
        # Token-level F1
        token_f1_scores = []
        for pred, ref in zip(predictions, references):
            f1 = self._compute_token_f1(pred, ref)
            token_f1_scores.append(f1)
        
        metrics["token_f1"] = float(np.mean(token_f1_scores))
        
        return metrics
    
    def _normalize_text(self, text: str) -> str:
        """テキスト正規化"""
        # 空白の統一、句読点の処理
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[。、]', '', text)
        return text.lower()
    
    def _compute_token_f1(self, prediction: str, reference: str) -> float:
        """トークンレベルのF1スコア計算"""
        
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        overlap = sum((pred_counter & ref_counter).values())
        
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1


def create_enhanced_compute_metrics(tokenizer):
    """強化されたcompute_metrics関数を作成"""
    
    evaluator = DisasterQAEvaluator(tokenizer)
    
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """Enhanced metrics computation for fine-tuning"""
        return evaluator.compute_metrics(eval_pred)
    
    return compute_metrics


# テスト用のサンプル実行
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # テスト用のデータ
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
    evaluator = DisasterQAEvaluator(tokenizer)
    
    # テキストメトリクスのテスト
    predictions = [
        "地震が発生した場合は机の下に隠れてください",
        "津波警報が出たら高台に避難してください"
    ]
    references = [
        "地震の時は机の下に隠れることが大切です",
        "津波警報では高台への避難が重要です"
    ]
    
    text_metrics = evaluator.compute_text_metrics(predictions, references)
    
    print("📊 Enhanced Evaluation Metrics Test")
    print("=" * 40)
    for metric, value in text_metrics.items():
        print(f"{metric}: {value:.4f}")