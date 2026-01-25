#!/usr/bin/env python3
"""
v0.2.1 Enhanced Training Pipeline
BERT + Bi-LSTM Architecture for End Position Accuracy Enhancement

Architecture:
cl-tohoku/bert-base-japanese-v3 → Bi-LSTM → Enhanced Position Heads

Target: Improve End Position accuracy by 50%+ over v0.2
Approach: BERT embeddings + Bi-LSTM context + Position-aware heads
"""

import argparse
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys
import os

from transformers import (
    AutoTokenizer, AutoModel,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

# Add src to path  
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_v021_bilstm.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class V021Config:
    """v0.2.1 Configuration"""
    base_model: str = "cl-tohoku/bert-base-japanese-v3"
    max_seq_length: int = 512
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 2
    dropout_rate: float = 0.1
    end_position_weight: float = 3.0  # Higher weight for end positions
    learning_rate: float = 2e-5
    epochs: int = 10
    batch_size: int = 8
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # LoRA config
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

class EnhancedPositionHead(nn.Module):
    """Enhanced Position Prediction Head with End Position Focus"""
    
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Position-specific layers
        self.position_dense = nn.Linear(hidden_dim, hidden_dim // 2)
        self.position_activation = nn.GELU()
        self.position_output = nn.Linear(hidden_dim // 2, 1)
        
        # End position enhancement layer
        self.end_enhancement = nn.Linear(hidden_dim, hidden_dim // 4)
        self.end_gate = nn.Linear(hidden_dim // 4, 1)
        
    def forward(self, hidden_states, attention_mask=None, position_type="start"):
        # Apply layer normalization and dropout
        normalized_hidden = self.layer_norm(hidden_states)
        dropped_hidden = self.dropout(normalized_hidden)
        
        # Position prediction
        position_hidden = self.position_dense(dropped_hidden)
        position_hidden = self.position_activation(position_hidden)
        position_logits = self.position_output(position_hidden).squeeze(-1)
        
        # End position enhancement
        if position_type == "end":
            end_hidden = self.end_enhancement(dropped_hidden)
            end_gate = torch.sigmoid(self.end_gate(end_hidden)).squeeze(-1)
            position_logits = position_logits * (1 + end_gate)  # Boost end positions
        
        # Apply attention mask if provided
        if attention_mask is not None:
            position_logits = position_logits.masked_fill(~attention_mask.bool(), -1e9)
        
        return position_logits

class BertBiLSTMQA(nn.Module):
    """BERT + Bi-LSTM + Enhanced Position Heads for Japanese Disaster QA"""
    
    def __init__(self, config: V021Config):
        super().__init__()
        self.config = config
        
        # Load BERT base model
        self.bert = AutoModel.from_pretrained(
            config.base_model,
            add_pooling_layer=False,
            use_safetensors=True
        )
        
        # Bi-LSTM layer
        self.bi_lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # LSTM output dimension
        lstm_output_dim = config.lstm_hidden_dim * 2  # Bidirectional
        
        # Enhanced position heads
        self.start_head = EnhancedPositionHead(lstm_output_dim, config.dropout_rate)
        self.end_head = EnhancedPositionHead(lstm_output_dim, config.dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize custom layer weights"""
        for module in [self.bi_lstm, self.start_head, self.end_head]:
            for name, param in module.named_parameters():
                if 'weight' in name:
                    if 'lstm' in name:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.normal_(param, std=0.02)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Bi-LSTM processing
        lstm_output, _ = self.bi_lstm(sequence_output)  # [batch_size, seq_len, lstm_hidden_dim * 2]
        
        # Position predictions
        start_logits = self.start_head(lstm_output, attention_mask, "start")
        end_logits = self.end_head(lstm_output, attention_mask, "end")
        
        outputs = {
            "start_logits": start_logits,
            "end_logits": end_logits
        }
        
        # Calculate loss if training
        if start_positions is not None and end_positions is not None:
            loss = self._calculate_loss(start_logits, end_logits, start_positions, end_positions)
            outputs["loss"] = loss
        
        return outputs
    
    def _calculate_loss(self, start_logits, end_logits, start_positions, end_positions):
        """Calculate position prediction loss with end position emphasis"""
        
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
        # Start position loss
        start_loss = loss_fn(start_logits, start_positions)
        
        # End position loss with higher weight
        end_loss = loss_fn(end_logits, end_positions) * self.config.end_position_weight
        
        total_loss = start_loss + end_loss
        return total_loss

class DisasterQADataset(Dataset):
    """Dataset for Japanese Disaster QA with BERT tokenization"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load dataset
        self._load_data(data_path)
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self, data_path: str):
        """Load QA dataset"""
        data_file = Path(data_path)
        
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        for item in raw_data:
            # Calculate token positions for start and end
            question = item["question"]
            context = item["context"]
            answer = item["answer"]
            
            # Tokenize question and context separately to find answer positions
            question_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            context_tokens = self.tokenizer(context, add_special_tokens=False)["input_ids"]
            
            # Find answer in context tokens
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            
            # Build full input: [CLS] question [SEP] context [SEP]
            input_ids = ([self.tokenizer.cls_token_id] + 
                        question_tokens + 
                        [self.tokenizer.sep_token_id] + 
                        context_tokens + 
                        [self.tokenizer.sep_token_id])
            
            # Adjust positions for the offset (CLS + question + SEP)
            context_offset = len(question_tokens) + 2  # [CLS] + question + [SEP]
            
            # Find answer start position in context tokens
            start_pos = self._find_answer_position(context_tokens, answer_tokens)
            if start_pos != -1:
                # Adjust for full input sequence
                start_position = start_pos + context_offset
                end_position = start_position + len(answer_tokens) - 1
                
                # Ensure positions are within bounds
                if end_position < len(input_ids):
                    self.samples.append({
                        "input_ids": input_ids,
                        "start_position": start_position,
                        "end_position": end_position,
                        "question": question,
                        "context": context,
                        "answer": answer
                    })
    
    def _find_answer_position(self, context_tokens: List[int], answer_tokens: List[int]) -> int:
        """Find start position of answer in context tokens"""
        for i in range(len(context_tokens) - len(answer_tokens) + 1):
            if context_tokens[i:i+len(answer_tokens)] == answer_tokens:
                return i
        return -1  # Not found
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_ids = sample["input_ids"][:self.max_length]
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "start_positions": torch.tensor(sample["start_position"], dtype=torch.long),
            "end_positions": torch.tensor(sample["end_position"], dtype=torch.long)
        }

class V021Trainer:
    """v0.2.1 BERT + Bi-LSTM Trainer"""
    
    def __init__(self, config: V021Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # Initialize model
        self.model = BertBiLSTMQA(config).to(self.device)
        
        # Apply LoRA
        self._apply_lora()
        
        logger.info(f"✅ v0.2.1 BERT+Bi-LSTM model initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _apply_lora(self):
        """Apply LoRA to BERT layers"""
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # Changed from QUESTION_ANS
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["query", "key", "value", "dense"]
        )
        
        # Apply LoRA only to BERT part
        self.model.bert = get_peft_model(self.model.bert, lora_config)
        
        logger.info(f"✅ LoRA applied to BERT layers")
        logger.info(f"   LoRA rank: {self.config.lora_rank}")
        logger.info(f"   LoRA alpha: {self.config.lora_alpha}")
    
    def train(self, train_dataset_path: str, val_dataset_path: str = None):
        """Train the v0.2.1 model"""
        
        # Load datasets
        train_dataset = DisasterQADataset(train_dataset_path, self.tokenizer, self.config.max_seq_length)
        val_dataset = DisasterQADataset(val_dataset_path, self.tokenizer, self.config.max_seq_length) if val_dataset_path else None
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False
        ) if val_dataset else None
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        num_training_steps = len(train_loader) * self.config.epochs
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )
        
        # Training loop
        self.model.train()
        training_stats = []
        
        logger.info(f"🚀 Starting v0.2.1 training...")
        logger.info(f"   Training samples: {len(train_dataset)}")
        logger.info(f"   Validation samples: {len(val_dataset) if val_dataset else 0}")
        logger.info(f"   Epochs: {self.config.epochs}")
        logger.info(f"   Batch size: {self.config.batch_size}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"]
                )
                loss = outputs["loss"]
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if batch_idx < self.config.warmup_steps:
                    lr_scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start
            
            # Validation
            val_metrics = self._validate(val_loader) if val_loader else {}
            
            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "epoch_time": epoch_time,
                **val_metrics
            }
            training_stats.append(epoch_stats)
            
            logger.info(f"✅ Epoch {epoch+1} completed - Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
            if val_metrics:
                logger.info(f"   Validation - Start Acc: {val_metrics.get('start_accuracy', 0):.3f}, End Acc: {val_metrics.get('end_accuracy', 0):.3f}")
        
        total_time = time.time() - start_time
        
        # Save model
        self._save_model(training_stats)
        
        logger.info(f"🏆 v0.2.1 training completed!")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info(f"   Final training loss: {training_stats[-1]['train_loss']:.4f}")
        
        return training_stats
    
    def _validate(self, val_loader):
        """Validate the model"""
        if not val_loader:
            return {}
        
        self.model.eval()
        start_correct = 0
        end_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"]
                )
                
                start_pred = torch.argmax(outputs["start_logits"], dim=1)
                end_pred = torch.argmax(outputs["end_logits"], dim=1)
                
                start_correct += (start_pred == batch["start_positions"]).sum().item()
                end_correct += (end_pred == batch["end_positions"]).sum().item()
                total += batch["start_positions"].size(0)
        
        self.model.train()
        
        return {
            "start_accuracy": start_correct / total,
            "end_accuracy": end_correct / total,
            "total_samples": total
        }
    
    def _save_model(self, training_stats):
        """Save the trained model"""
        
        model_dir = Path("models/v021_bert_bilstm")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full model state (including LoRA)
        torch.save(self.model.state_dict(), model_dir / "model_state.pt")
        
        # Save BERT model separately for proper loading
        self.model.bert.save_pretrained(model_dir / "bert")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(model_dir)
        
        # Save config
        with open(model_dir / "config.json", 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
        
        # Save training stats
        with open(model_dir / "training_stats.json", 'w', encoding='utf-8') as f:
            json.dump(training_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Model saved to {model_dir}")
        logger.info(f"✅ BERT with LoRA saved to {model_dir / 'bert'}")
        logger.info(f"✅ Full model state saved to {model_dir / 'model_state.pt'}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="v0.2.1 BERT + Bi-LSTM Training")
    parser.add_argument("--dataset-size", type=int, default=500, choices=[100, 500, 1000], 
                        help="Dataset size to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--end-position-weight", type=float, default=3.0, 
                        help="Weight for end position loss")
    
    args = parser.parse_args()
    
    # Ensure required directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Configuration
    config = V021Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        end_position_weight=args.end_position_weight
    )
    
    # Dataset paths
    train_path = f"data/processed/qa_dataset_v2/qa_samples_{args.dataset_size}.json"
    val_path = "data/processed/test_dataset/test_samples_300.json"  # Independent test set
    
    # Check if datasets exist
    if not Path(train_path).exists():
        logger.error(f"Training dataset not found: {train_path}")
        logger.info("Please run: python create_qa_dataset.py --size {args.dataset_size} --output-dir data/processed/qa_dataset_v2")
        return
    
    # Initialize trainer
    trainer = V021Trainer(config)
    
    logger.info("🎯 v0.2.1 BERT + Bi-LSTM Training Starting...")
    logger.info(f"   Training dataset: {train_path}")
    logger.info(f"   Validation dataset: {val_path}")
    logger.info(f"   End position weight: {config.end_position_weight}x")
    
    # Train model
    try:
        training_stats = trainer.train(train_path, val_path if Path(val_path).exists() else None)
        
        logger.info("🏆 v0.2.1 Training completed successfully!")
        logger.info("📊 Final Results:")
        final_stats = training_stats[-1]
        logger.info(f"   Training Loss: {final_stats['train_loss']:.4f}")
        if 'end_accuracy' in final_stats:
            logger.info(f"   End Position Accuracy: {final_stats['end_accuracy']:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()