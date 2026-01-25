"""
v0.3 Enhanced LoRA Fine-tuning (Simplified BiLSTM Version)
End Position精度向上に特化、CRF無しでBiLSTMのみ使用

主な改善点:
1. End Position Loss重み付け (2.5倍)
2. BiLSTM layer追加でspan detection強化  
3. 終了位置特化データ拡張対応
"""

import json
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering, AutoModel,
    TrainingArguments, Trainer, EvalPrediction
)
from peft import (
    LoraConfig, get_peft_model, TaskType, 
    prepare_model_for_kbit_training
)
from datasets import Dataset as HFDataset

from ..utils import load_config, get_project_root, ensure_dir
from .enhanced_evaluator import create_enhanced_compute_metrics
from .fine_tuning import DisaQuADDataset, DisaQuADSample, LoRATrainer

logger = logging.getLogger(__name__)


class BiLSTMHead(nn.Module):
    """
    BiLSTM Head for enhanced span boundary detection (without CRF)
    
    BERT出力 → BiLSTM → Position Predictions
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 lstm_hidden_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced QA heads with BiLSTM features
        self.qa_outputs = nn.Linear(lstm_hidden_size * 2, 2)  # start/end logits
        
        # Additional position-specific heads for enhanced end position detection
        self.start_head = nn.Linear(lstm_hidden_size * 2, 1)
        self.end_head = nn.Linear(lstm_hidden_size * 2, 1)
        
    def forward(self, sequence_output, attention_mask=None):
        """
        Args:
            sequence_output: BERT output [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            start_logits, end_logits
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # BiLSTM processing
        lstm_output, _ = self.bilstm(sequence_output)  # [batch, seq_len, lstm_hidden*2]
        lstm_output = self.dropout(lstm_output)
        
        # Enhanced position predictions
        start_logits = self.start_head(lstm_output).squeeze(-1)  # [batch, seq_len]
        end_logits = self.end_head(lstm_output).squeeze(-1)      # [batch, seq_len]
        
        # Apply attention mask
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(~attention_mask.bool(), -1e9)
            end_logits = end_logits.masked_fill(~attention_mask.bool(), -1e9)
        
        return start_logits, end_logits


class EnhancedQAModelV3Simple(nn.Module):
    """
    v0.3 Enhanced QA Model with BiLSTM (Simplified without CRF)
    
    Architecture: BERT → BiLSTM → Enhanced Position Heads
    """
    
    def __init__(self, 
                 model_name: str = "cl-tohoku/bert-base-japanese-v3",
                 lstm_hidden_size: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        self.model_name = model_name
        
        # Base BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Enhanced BiLSTM head
        self.qa_head = BiLSTMHead(
            hidden_size=self.bert.config.hidden_size,
            lstm_hidden_size=lstm_hidden_size,
            dropout=dropout
        )
        
    def forward(self, 
                input_ids,
                attention_mask=None,
                start_positions=None,
                end_positions=None):
        
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Enhanced QA head with BiLSTM
        start_logits, end_logits = self.qa_head(
            sequence_output, attention_mask
        )
        
        outputs = {
            "start_logits": start_logits,
            "end_logits": end_logits
        }
        
        # Loss computation during training
        if start_positions is not None and end_positions is not None:
            loss = self.compute_weighted_loss(
                start_logits, end_logits,
                start_positions, end_positions
            )
            outputs["loss"] = loss
        
        return outputs
    
    def compute_weighted_loss(self,
                            start_logits, end_logits,
                            start_positions, end_positions):
        """
        v0.3 核心: End Position重み付き損失関数
        
        start_weight: 1.0 (base)
        end_weight: 2.5 (enhanced for v0.3)
        """
        
        # Position-based loss
        loss_fct = nn.CrossEntropyLoss()
        
        # Start position loss
        start_loss = loss_fct(start_logits, start_positions)
        
        # End position loss with 2.5x weight
        end_loss = loss_fct(end_logits, end_positions) 
        
        # v0.3 重み付け: end position を2.5倍強調
        START_WEIGHT = 1.0
        END_WEIGHT = 2.5
        
        total_loss = START_WEIGHT * start_loss + END_WEIGHT * end_loss
        
        return total_loss


class LoRATrainerV3Simple(LoRATrainer):
    """
    v0.3 Enhanced LoRA Trainer with BiLSTM and Weighted Loss (Simplified)
    """
    
    def __init__(self, config_path: str = None):
        # Load base config from parent
        super().__init__(config_path)
        
        # v0.3 specific enhancements
        self.v3_config = {
            "lstm_hidden_size": 256,
            "end_position_weight": 2.5,
            "dropout": 0.1
        }
        
        self.enhanced_model = None
    
    def load_base_model(self):
        """v0.3 Enhanced model loading with BiLSTM"""
        logger.info(f"Loading v0.3 simplified enhanced model: {self.model_name}")
        
        # Enhanced model instead of standard QA model
        self.enhanced_model = EnhancedQAModelV3Simple(
            model_name=self.model_name,
            lstm_hidden_size=self.v3_config["lstm_hidden_size"],
            dropout=self.v3_config["dropout"]
        )
        
        # Set as main model for LoRA
        self.model = self.enhanced_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("v0.3 Simplified Enhanced model loaded with BiLSTM")
    
    def setup_lora(self):
        """v0.3 Enhanced LoRA setup"""
        if self.enhanced_model is None:
            raise ValueError("Enhanced model not loaded. Call load_base_model() first.")
        
        # LoRA config for enhanced model
        lora_config = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
            r=self.lora_config['lora_rank'],
            lora_alpha=self.lora_config['lora_alpha'],
            lora_dropout=self.lora_config['lora_dropout'],
            # Enhanced target modules for BiLSTM
            target_modules=[
                "query", "value", "key", "dense",  # BERT layers
                "qa_outputs", "start_head", "end_head"  # Enhanced heads
            ]
        )
        
        # Apply LoRA to enhanced model
        self.peft_model = get_peft_model(self.enhanced_model, lora_config)
        self.peft_model.to(self.device)
        
        # Enhanced parameter statistics
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        logger.info(f"v0.3 Enhanced Trainable parameters: {trainable_params:,}")
        logger.info(f"v0.3 Total parameters: {total_params:,}")
        logger.info(f"v0.3 Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        # Show BiLSTM specific params
        bilstm_params = sum(p.numel() for p in self.enhanced_model.qa_head.bilstm.parameters())
        
        logger.info(f"BiLSTM parameters: {bilstm_params:,}")


def create_end_position_augmented_dataset(base_dataset_path: str, 
                                        output_path: str,
                                        augmentation_factor: int = 2):
    """
    v0.3 データ拡張: 終了位置周辺のコンテキスト強化
    
    既存データセットから終了位置検出に特化した追加サンプルを生成
    """
    logger.info(f"Creating end-position augmented dataset from {base_dataset_path}")
    
    with open(base_dataset_path, 'r', encoding='utf-8') as f:
        base_samples = json.load(f)
    
    augmented_samples = []
    
    for sample in base_samples:
        # Original sample
        augmented_samples.append(sample)
        
        # Create augmented samples focused on end position
        context = sample["context"]
        answer = sample["answer"]
        
        answer_start_char = context.find(answer)
        if answer_start_char == -1:
            continue
        
        answer_end_char = answer_start_char + len(answer)
        
        # Create variations with different context windows around answer
        for i in range(augmentation_factor):
            # Expand context around answer end position
            expand_start = 30 + i * 15  # Variable expansion around start
            expand_end = 50 + i * 20    # Larger expansion around end position
            
            context_start = max(0, answer_start_char - expand_start)
            context_end = min(len(context), answer_end_char + expand_end)
            
            augmented_context = context[context_start:context_end]
            
            # Adjust answer position in new context
            new_answer_start = augmented_context.find(answer)
            if new_answer_start != -1:
                augmented_sample = {
                    **sample,
                    "context": augmented_context,
                    "start_char": new_answer_start,
                    "augmented": True,
                    "augmentation_type": f"end_position_focused_{i}",
                    "v3_enhanced": True
                }
                augmented_samples.append(augmented_sample)
    
    # Save augmented dataset
    ensure_dir(Path(output_path).parent)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created augmented dataset: {len(base_samples)} → {len(augmented_samples)} samples")
    logger.info(f"Augmented dataset saved to {output_path}")
    
    return output_path


def main():
    """v0.3 Simplified Enhanced Fine-tuning CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='v0.3 End Position Enhanced LoRA Fine-tuning (Simplified)')
    parser.add_argument('--train', action='store_true',
                      help='Start v0.3 enhanced fine-tuning')
    parser.add_argument('--augment-data', action='store_true',
                      help='Create end-position augmented dataset')
    parser.add_argument('--base-dataset', type=str,
                      default='data/processed/qa_dataset_v2/qa_samples_500.json',
                      help='Base dataset for augmentation')
    parser.add_argument('--dataset-size', type=int, 
                      default=500, choices=[100, 500, 1000],
                      help='Base dataset size')
    parser.add_argument('--output-dir', type=str,
                      default='models/lora_v03_enhanced_simple',
                      help='Output directory for v0.3 model')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger.info("🚀 v0.3 Simplified End Position Enhancement Starting...")
    
    if args.augment_data:
        # Create augmented dataset
        augmented_path = f"data/processed/qa_dataset_v3/qa_samples_{args.dataset_size}_augmented.json"
        create_end_position_augmented_dataset(
            args.base_dataset, 
            augmented_path,
            augmentation_factor=2
        )
        print(f"✅ Augmented dataset created: {augmented_path}")
        return
    
    if args.train:
        # v0.3 Enhanced Training
        trainer = LoRATrainerV3Simple()
        
        # Load augmented dataset if available, otherwise use base
        augmented_path = f"data/processed/qa_dataset_v3/qa_samples_{args.dataset_size}_augmented.json"
        if Path(augmented_path).exists():
            logger.info(f"Using augmented dataset: {augmented_path}")
            # Load augmented dataset manually
            dataset = DisaQuADDataset(dataset_size=args.dataset_size)
            # Replace with augmented samples
            with open(augmented_path, 'r', encoding='utf-8') as f:
                raw_samples = json.load(f)
            
            dataset.samples = []
            for sample_data in raw_samples:
                sample = DisaQuADSample(
                    question=sample_data["question"],
                    context=sample_data["context"],
                    answer=sample_data["answer"],
                    start_position=sample_data.get("start_char", 0),
                    end_position=sample_data.get("start_char", 0) + len(sample_data["answer"]),
                    disaster_type=sample_data["disaster_type"],
                    question_type=sample_data["question_type"],
                    document_source=sample_data.get("document_source", "unknown")
                )
                dataset.samples.append(sample)
        else:
            logger.info(f"Using base dataset: {args.dataset_size} samples")
            dataset = DisaQuADDataset(dataset_size=args.dataset_size)
        
        train_dataset, eval_dataset = dataset.split(0.8)
        
        # Model setup
        trainer.load_base_model()
        trainer.setup_lora()
        
        # Enhanced training
        logger.info("🔧 Starting v0.3 Simplified End Position Enhanced Training...")
        model_path = trainer.train(train_dataset, eval_dataset, args.output_dir)
        
        print("✅ v0.3 Simplified Enhanced LoRA Fine-tuning Completed!")
        print(f"📁 Model saved at: {model_path}")
        print("🎯 Expected: End Position Accuracy 8.7% → 25%+ (with BiLSTM enhancement)")


if __name__ == '__main__':
    main()