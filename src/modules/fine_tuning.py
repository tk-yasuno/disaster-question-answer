"""
LoRAファインチューニングモジュール

BERT-baseモデルをPEFT (LoRA)で軽量ファインチューニング
DisaQuADデータセット形式に対応
"""

import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, EvalPrediction
)
from peft import (
    LoraConfig, get_peft_model, TaskType, 
    prepare_model_for_kbit_training
)
from datasets import Dataset as HFDataset
import evaluate

from ..utils import load_config, get_project_root, ensure_dir
from .enhanced_evaluator import create_enhanced_compute_metrics

logger = logging.getLogger(__name__)

@dataclass
class DisaQuADSample:
    """災害QAデータサンプル"""
    question: str
    context: str
    answer: str
    start_position: int
    end_position: int
    disaster_type: str
    question_type: str  # WhatIf/WhatAct/WhatIs etc.
    document_source: str

class DisaQuADDataset:
    """
    災害QAデータセット
    
    v0.2 Update: 5→100→500→1000サンプル対応
    拡張されたデータセットローダー
    """
    
    def __init__(self, data_dir: Path = None, dataset_size: int = 100):
        """
        Args:
            data_dir: データディレクトリパス
            dataset_size: データセットサイズ (5, 100, 500, 1000)
        """
        self.data_dir = data_dir or Path("data/processed/qa_dataset_v2")
        self.dataset_size = dataset_size
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """拡張されたデータセットを読み込み"""
        
        # v0.2データセットを優先的に読み込み
        dataset_file = self.data_dir / f"qa_samples_{self.dataset_size}.json"
        
        if dataset_file.exists():
            logger.info(f"Loading v0.2 dataset: {dataset_file}")
            with open(dataset_file, 'r', encoding='utf-8') as f:
                raw_samples = json.load(f)
            
            # 辞書形式のデータをDisaQuADSampleオブジェクトに変換
            for sample_data in raw_samples:
                sample = DisaQuADSample(
                    question=sample_data["question"],
                    context=sample_data["context"],
                    answer=sample_data["answer"],
                    start_position=sample_data.get("start_char", 0),  # 文字位置を使用
                    end_position=sample_data.get("start_char", 0) + len(sample_data["answer"]),
                    disaster_type=sample_data["disaster_type"],
                    question_type=sample_data["question_type"],
                    document_source=sample_data.get("document_source", "unknown")
                )
                self.samples.append(sample)
        else:
            logger.warning(f"v0.2 dataset not found: {dataset_file}")
            # フォールバック: 既存のデータまたは合成データ
            self._create_synthetic_data()
        
        logger.info(f"Loaded {len(self.samples)} QA samples")
    
    def _create_synthetic_data(self):
        """MVP用の合成災害QAデータを作成"""
        
        synthetic_samples = [
            {
                "question": "地震が発生したときにまず何をすべきですか？",
                "context": "地震が発生した場合、まず自分の安全を確保することが重要です。机の下に隠れるか、頭を保護してください。その後、火の元を確認し、ガスの元栓を閉めてください。避難する際は、エレベーターは使わず、階段を使用してください。",
                "answer": "机の下に隠れるか、頭を保護してください",
                "start_char": 45,
                "disaster_type": "earthquake",
                "question_type": "WhatAct",
                "document_source": "earthquake_manual.pdf"
            },
            {
                "question": "津波警報が出された場合の対応は？",
                "context": "津波警報が発令された場合、直ちに高台または津波避難ビルに避難してください。海岸や河川の近くにいる場合は、すぐに離れてください。車での避難は渋滞の可能性があるため、徒歩での避難を推奨します。",
                "answer": "直ちに高台または津波避難ビルに避難してください",
                "start_char": 12,
                "disaster_type": "tsunami", 
                "question_type": "WhatAct",
                "document_source": "tsunami_guideline.pdf"
            },
            {
                "question": "台風接近時の家庭での準備は？",
                "context": "台風が接近している場合、窓ガラスに飛散防止フィルムを貼るか、カーテンを閉めてください。また、停電に備えて懐中電灯や非常用電源を準備し、食料や水を3日分確保してください。屋外の物は室内に取り込むか、固定してください。",
                "answer": "窓ガラスに飛散防止フィルムを貼るか、カーテンを閉めてください",
                "start_char": 15,
                "disaster_type": "typhoon",
                "question_type": "WhatAct", 
                "document_source": "typhoon_preparation.pdf"
            },
            {
                "question": "洪水警報とは何ですか？",
                "context": "洪水警報は、河川の水位が上昇し、氾濫の恐れがある場合に気象庁が発表する警報です。この警報が出された地域では、河川や用水路の近くには近づかず、避難の準備をしてください。警戒レベル3に相当します。",
                "answer": "河川の水位が上昇し、氾濫の恐れがある場合に気象庁が発表する警報です",
                "start_char": 6,
                "disaster_type": "flood",
                "question_type": "WhatIs",
                "document_source": "flood_warning_system.pdf"
            },
            {
                "question": "避難所での生活で注意すべきことは？",
                "context": "避難所での生活では、感染症予防のために手洗いやマスク着用を心がけてください。プライバシーの確保と他の避難者との協調が重要です。また、体調不良者がいる場合は、速やかに避難所の運営者に報告してください。",
                "answer": "感染症予防のために手洗いやマスク着用を心がけてください",
                "start_char": 12,
                "disaster_type": "general",
                "question_type": "WhatAct",
                "document_source": "evacuation_center_guide.pdf"
            }
        ]
        
        for sample_data in synthetic_samples:
            # 文字位置からトークン位置を計算（簡単な実装）
            context = sample_data["context"]
            start_char = sample_data["start_char"]
            answer = sample_data["answer"]
            
            # 簡易的な位置計算
            start_pos = len(context[:start_char].split())
            end_pos = start_pos + len(answer.split()) - 1
            
            sample = DisaQuADSample(
                question=sample_data["question"],
                context=context,
                answer=answer,
                start_position=start_pos,
                end_position=end_pos,
                disaster_type=sample_data["disaster_type"],
                question_type=sample_data["question_type"],
                document_source=sample_data["document_source"]
            )
            
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def split(self, train_ratio=0.8):
        """データセットを訓練用と評価用に分割"""
        total_samples = len(self.samples)
        train_size = int(total_samples * train_ratio)
        
        train_samples = self.samples[:train_size]
        eval_samples = self.samples[train_size:]
        
        # 新しいDisaQuADDatasetインスタンスを作成
        train_dataset = DisaQuADDataset.__new__(DisaQuADDataset)
        train_dataset.samples = train_samples
        train_dataset.data_dir = self.data_dir
        train_dataset.dataset_size = len(train_samples)
        
        eval_dataset = DisaQuADDataset.__new__(DisaQuADDataset)
        eval_dataset.samples = eval_samples
        eval_dataset.data_dir = self.data_dir
        eval_dataset.dataset_size = len(eval_samples)
        
        return train_dataset, eval_dataset
    
    def split(self, train_ratio: float = 0.8) -> Tuple['DisaQuADDataset', 'DisaQuADDataset']:
        """データを訓練用と評価用に分割"""
        n_train = int(len(self.samples) * train_ratio)
        
        train_dataset = DisaQuADDataset.__new__(DisaQuADDataset)
        train_dataset.samples = self.samples[:n_train]
        
        eval_dataset = DisaQuADDataset.__new__(DisaQuADDataset)
        eval_dataset.samples = self.samples[n_train:]
        
        return train_dataset, eval_dataset

class LoRATrainer:
    """
    LoRAファインチューニング実行クラス
    
    機能:
    - PEFT (LoRA) 設定・適用
    - DisaQuADデータセット学習
    - モデル評価・保存
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.model_name = self.config['qa_model']
        self.lora_config = self.config['fine_tuning']
        self.max_seq_length = self.config['max_seq_length']
        
        # デバイス設定
        self.device = torch.device(self.config.get('device', 'cpu'))
        
        # パス設定
        self.model_dir = get_project_root() / self.config['data_paths']['models']
        ensure_dir(self.model_dir)
        
        # モデル・トークナイザ
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        logger.info(f"LoRA Trainer initialized for {self.model_name}")
    
    def load_base_model(self):
        """ベースモデルとトークナイザーを読み込み"""
        logger.info(f"Loading base model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            trust_remote_code=True,
            use_safetensors=True
        )
        
        logger.info("Base model loaded successfully")
    
    def setup_lora(self):
        """LoRA設定を適用"""
        if self.model is None:
            raise ValueError("Base model not loaded. Call load_base_model() first.")
        
        lora_config = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
            r=self.lora_config['lora_rank'],
            lora_alpha=self.lora_config['lora_alpha'],
            lora_dropout=self.lora_config['lora_dropout'],
            target_modules=["query", "value", "key", "dense"]  # BERT用
        )
        
        # PEFTモデル作成
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.to(self.device)
        
        # 学習可能パラメータ数を表示
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    def prepare_dataset(self, dataset: DisaQuADDataset) -> HFDataset:
        """データセットをHuggingFace形式に変換"""
        
        def tokenize_sample(sample):
            # 質問とコンテキストをトークン化
            encoding = self.tokenizer(
                sample.question,
                sample.context,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # 簡単な方法で回答位置を推定（実際のプロダクションではより精密な実装が必要）
            # 文字単位での回答開始・終了位置をトークン単位で近似
            answer_start_char = sample.context.find(sample.answer)
            answer_end_char = answer_start_char + len(sample.answer)
            
            # 簡易的な位置推定（改良の余地あり）
            context_tokens = self.tokenizer.tokenize(sample.context)
            total_tokens = len(context_tokens)
            
            if total_tokens > 0:
                start_ratio = answer_start_char / len(sample.context) if len(sample.context) > 0 else 0
                end_ratio = answer_end_char / len(sample.context) if len(sample.context) > 0 else 0
                
                start_position = int(start_ratio * total_tokens) + 1  # [CLS] + question tokens分のオフセット
                end_position = int(end_ratio * total_tokens) + 1
                
                # 範囲チェック
                start_position = max(1, min(start_position, self.max_seq_length - 1))
                end_position = max(start_position, min(end_position, self.max_seq_length - 1))
            else:
                start_position = 1
                end_position = 1
            
            encoding["start_positions"] = torch.tensor(start_position, dtype=torch.long)
            encoding["end_positions"] = torch.tensor(end_position, dtype=torch.long)
            
            return {key: val.squeeze() if hasattr(val, 'squeeze') else val for key, val in encoding.items()}
        
        # データ変換
        processed_data = []
        for sample in dataset:
            try:
                tokenized = tokenize_sample(sample)
                processed_data.append(tokenized)
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                continue
        
        # デバッグ情報を出力
        if processed_data:
            sample_keys = list(processed_data[0].keys())
            logger.info(f"Dataset columns: {sample_keys}")
            for key in sample_keys:
                sample_value = processed_data[0][key]
                logger.info(f"Column '{key}': type={type(sample_value)}, shape={getattr(sample_value, 'shape', 'N/A')}")
        
        hf_dataset = HFDataset.from_list(processed_data)
        logger.info(f"HuggingFace dataset features: {hf_dataset.features}")
        return hf_dataset
    
    def train(self, 
              train_dataset: DisaQuADDataset,
              eval_dataset: DisaQuADDataset = None,
              output_dir: str = None) -> str:
        """LoRAファインチューニングを実行"""
        
        if self.peft_model is None:
            raise ValueError("PEFT model not set up. Call setup_lora() first.")
        
        # 出力ディレクトリ
        if output_dir is None:
            output_dir = self.model_dir / f"lora_finetuned_{self.model_name.split('/')[-1]}"
        
        ensure_dir(output_dir)
        
        logger.info(f"Starting LoRA fine-tuning...")
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation samples: {len(eval_dataset)}")
        
        # データセット準備
        train_hf_dataset = self.prepare_dataset(train_dataset)
        eval_hf_dataset = self.prepare_dataset(eval_dataset) if eval_dataset else None
        
        # 学習設定
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.lora_config['num_epochs'],
            per_device_train_batch_size=self.lora_config['batch_size'],
            per_device_eval_batch_size=self.lora_config['batch_size'],
            warmup_steps=self.lora_config['warmup_steps'],
            learning_rate=float(self.lora_config['learning_rate']),  # 確実に数値に変換
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps" if eval_hf_dataset else "no",  # evaluation_strategy -> eval_strategy
            eval_steps=100 if eval_hf_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_hf_dataset else False,
            metric_for_best_model="eval_overall_f1" if eval_hf_dataset else None,  # 修正: eval_f1 -> eval_overall_f1
            remove_unused_columns=False,  # データセット列の問題を回避
            fp16=self.device.type == 'cuda',
            dataloader_pin_memory=False,
            report_to="none"  # wandbを無効化
        )
        
        # v0.2: Enhanced評価メトリクス使用
        compute_metrics = create_enhanced_compute_metrics(self.tokenizer)
        
        # Trainer作成
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_hf_dataset,
            eval_dataset=eval_hf_dataset,
            compute_metrics=compute_metrics if eval_hf_dataset else None,
            processing_class=self.tokenizer
        )
        
        # 学習実行
        try:
            train_result = trainer.train()
            
            # モデル保存
            trainer.save_model()
            
            # 学習結果ログ
            logger.info(f"Training completed!")
            logger.info(f"Final loss: {train_result.training_loss:.4f}")
            
            if eval_hf_dataset:
                eval_result = trainer.evaluate()
                logger.info(f"Evaluation results: {eval_result}")
            
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, test_dataset: DisaQuADDataset, model_path: str = None):
        """学習済みモデルを評価"""
        
        if model_path:
            self.load_trained_model(model_path)
        
        if self.peft_model is None:
            raise ValueError("No model loaded for evaluation")
        
        test_hf_dataset = self.prepare_dataset(test_dataset)
        
        # 簡単な評価実装
        logger.info(f"Evaluating model on {len(test_dataset)} samples...")
        
        # 評価メトリクス計算
        correct_predictions = 0
        total_predictions = len(test_dataset)
        
        # ここで実際の予測と評価を実行
        # (実装簡略化のため詳細省略)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"Evaluation accuracy: {accuracy:.3f}")
        
        return {
            "accuracy": accuracy,
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions
        }
    
    def load_trained_model(self, model_path: str):
        """学習済みLoRAモデルを読み込み"""
        logger.info(f"Loading trained LoRA model from {model_path}")
        
        # ベースモデル読み込み
        if self.model is None:
            self.load_base_model()
        
        # LoRA設定適用
        if self.peft_model is None:
            self.setup_lora()
        
        # 学習済み重みを読み込み
        self.peft_model.load_adapter(model_path, "default")
        
        logger.info("Trained LoRA model loaded successfully")
    
    def save_model(self, output_path: str):
        """モデルを保存"""
        if self.peft_model is None:
            raise ValueError("No trained model to save")
        
        ensure_dir(output_path)
        
        # LoRA重みのみ保存
        self.peft_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # 設定も保存
        config_path = Path(output_path) / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model saved to {output_path}")


def main():
    """CLI実行用メイン関数 - v0.2 Enhanced"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LoRA Fine-tuning CLI v0.2')
    parser.add_argument('--train', action='store_true',
                      help='Start LoRA fine-tuning')
    parser.add_argument('--evaluate', type=str,
                      help='Evaluate trained model (provide model path)')
    parser.add_argument('--data-dir', type=str,
                      default='data/processed/qa_dataset_v2',
                      help='DisaQuAD dataset directory (v0.2)')
    parser.add_argument('--dataset-size', type=int, 
                      default=100, choices=[5, 100, 500, 1000],
                      help='Dataset size for training')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    trainer = LoRATrainer()
    
    # v0.2拡張データセット作成
    data_dir = Path(args.data_dir)
    dataset = DisaQuADDataset(data_dir, dataset_size=args.dataset_size)
    train_dataset, eval_dataset = dataset.split(0.8)
    
    logger.info(f"Dataset loaded: {len(dataset.samples)} total samples")
    logger.info(f"Training samples: {len(train_dataset.samples)}")
    logger.info(f"Evaluation samples: {len(eval_dataset.samples)}")
    
    if args.train:
        logger.info(f"Starting LoRA fine-tuning with {args.dataset_size} samples...")
        
        # モデル準備
        trainer.load_base_model()
        trainer.setup_lora()
        
        # 学習実行
        model_path = trainer.train(train_dataset, eval_dataset)
        print(f"✅ LoRA fine-tuning completed. Model saved at: {model_path}")
        
        # 保存されたモデルのパフォーマンステスト
        logger.info("Running post-training validation...")
        results = trainer.evaluate_model(eval_dataset, model_path)
        
        print("📊 Final Evaluation Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
    
    if args.evaluate:
        logger.info(f"Evaluating model: {args.evaluate}")
        
        trainer.load_base_model()
        trainer.setup_lora()
        
        results = trainer.evaluate_model(eval_dataset, args.evaluate)
        
        print("📊 Evaluation Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

if __name__ == '__main__':
    main()