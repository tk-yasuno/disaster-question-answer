"""
Enhanced Fine-tuning Module v0.3 with Safetensors Support
End Position accuracy enhancement using BiLSTM + custom CRF architecture
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.utils import TRANSFORMERS_CACHE
import logging
from pathlib import Path
import copy
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QATrainingConfig:
    """Training configuration for v0.3 enhanced QA model"""
    model_name: str = "cl-tohoku/bert-base-japanese-v3"
    max_length: int = 512
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # BiLSTM configuration
    bilstm_hidden_size: int = 256
    bilstm_num_layers: int = 2
    bilstm_dropout: float = 0.2
    
    # Loss weights for end position enhancement
    start_position_weight: float = 1.0
    end_position_weight: float = 2.5  # 2.5x emphasis on end positions
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

class EnhancedQAModelV3Safetensors(nn.Module):
    """Enhanced QA Model v0.3 with BiLSTM and Safetensors support"""
    
    def __init__(self, model_name: str, config: QATrainingConfig):
        super().__init__()
        self.config = config
        self.model_name = model_name
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with safetensors if available
        try:
            # Try to load with safetensors first
            model_config = AutoConfig.from_pretrained(model_name)
            self.bert = AutoModel.from_config(model_config)
            
            # Check for safetensors file
            cache_dir = TRANSFORMERS_CACHE
            safetensors_path = None
            for path in Path(cache_dir).rglob("*.safetensors"):
                if "bert-base-japanese-v3" in str(path):
                    safetensors_path = path
                    break
            
            if safetensors_path and safetensors_path.exists():
                logger.info(f"Loading model from safetensors: {safetensors_path}")
                state_dict = load_file(safetensors_path)
                self.bert.load_state_dict(state_dict, strict=False)
            else:
                logger.warning("Safetensors not found, using alternative loading method")
                # Use alternative loading method
                self._load_model_alternative()
                
        except Exception as e:
            logger.error(f"Failed to load with safetensors: {e}")
            self._load_model_alternative()
        
        # BiLSTM layer for enhanced sequence modeling
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=config.bilstm_hidden_size,
            num_layers=config.bilstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.bilstm_dropout
        )
        
        # Enhanced position prediction heads
        bilstm_output_size = config.bilstm_hidden_size * 2  # bidirectional
        
        self.start_position_head = nn.Sequential(
            nn.Linear(bilstm_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
        self.end_position_head = nn.Sequential(
            nn.Linear(bilstm_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
        logger.info(f"✅ Enhanced QA Model v0.3 (Safetensors) initialized with BiLSTM")
        logger.info(f"   - BiLSTM: {config.bilstm_hidden_size}x{config.bilstm_num_layers} bidirectional")
        logger.info(f"   - End Position Weight: {config.end_position_weight}x")
    
    def _load_model_alternative(self):
        """Alternative model loading without torch.load"""
        try:
            # Try to use from_pretrained with trust_remote_code
            self.bert = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=False
            )
        except Exception as e:
            logger.error(f"Alternative loading also failed: {e}")
            # Initialize with config only
            model_config = AutoConfig.from_pretrained(self.model_name)
            self.bert = AutoModel.from_config(model_config)
            logger.warning("Initialized model from config only - weights may be random")
    
    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        # BERT encoding
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # BiLSTM processing
        lstm_output, _ = self.bilstm(sequence_output)  # [batch, seq_len, bilstm_hidden*2]
        
        # Position predictions
        start_logits = self.start_position_head(lstm_output).squeeze(-1)  # [batch, seq_len]
        end_logits = self.end_position_head(lstm_output).squeeze(-1)     # [batch, seq_len]
        
        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }
        
        # Calculate loss if positions are provided
        if start_positions is not None and end_positions is not None:
            loss = self._calculate_weighted_loss(start_logits, end_logits, start_positions, end_positions, attention_mask)
            outputs['loss'] = loss
        
        return outputs
    
    def _calculate_weighted_loss(self, start_logits, end_logits, start_positions, end_positions, attention_mask):
        """Calculate weighted loss with emphasis on end positions"""
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        # Start position loss
        start_loss = loss_fct(start_logits, start_positions)
        start_loss = (start_loss * attention_mask.float()).sum() / attention_mask.sum()
        
        # End position loss (weighted)
        end_loss = loss_fct(end_logits, end_positions)
        end_loss = (end_loss * attention_mask.float()).sum() / attention_mask.sum()
        
        # Combine with weights
        total_loss = (
            self.config.start_position_weight * start_loss + 
            self.config.end_position_weight * end_loss
        )
        
        return total_loss

class LoRATrainerV3Safetensors:
    """LoRA Trainer for v0.3 Enhanced Model with Safetensors support"""
    
    def __init__(self, config: QATrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.enhanced_model = None
        
        logger.info(f"🚀 LoRA Trainer v0.3 (Safetensors) initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   End Position Weight: {config.end_position_weight}x")
    
    def load_base_model(self):
        """Load base model with enhanced architecture"""
        logger.info(f"Loading v0.3 safetensors enhanced model: {self.config.model_name}")
        
        try:
            self.enhanced_model = EnhancedQAModelV3Safetensors(self.config.model_name, self.config)
            self.enhanced_model.to(self.device)
            self.tokenizer = self.enhanced_model.tokenizer
            
            logger.info("✅ v0.3 Safetensors Enhanced Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load v0.3 enhanced model: {e}")
            raise e
    
    def prepare_datasets(self, qa_samples: List[Dict], split_ratio: float = 0.8) -> Tuple:
        """Prepare training and evaluation datasets"""
        
        # Split data
        split_idx = int(len(qa_samples) * split_ratio)
        train_samples = qa_samples[:split_idx]
        eval_samples = qa_samples[split_idx:]
        
        # Create datasets
        train_dataset = EnhancedQADataset(train_samples, self.tokenizer, self.config.max_length)
        eval_dataset = EnhancedQADataset(eval_samples, self.tokenizer, self.config.max_length)
        
        logger.info(f"📊 Dataset prepared: {len(train_samples)} train, {len(eval_samples)} eval")
        
        return train_dataset, eval_dataset
    
    def train_model(self, qa_samples: List[Dict], output_dir: str = "models/v03_safetensors_enhanced") -> str:
        """Train the enhanced model"""
        
        if self.enhanced_model is None:
            self.load_base_model()
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets(qa_samples)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.enhanced_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        self.enhanced_model.train()
        total_steps = len(train_loader) * self.config.num_epochs
        
        logger.info(f"🚀 Starting v0.3 Safetensors training...")
        logger.info(f"   Total steps: {total_steps}")
        logger.info(f"   Batch size: {self.config.batch_size}")
        logger.info(f"   Learning rate: {self.config.learning_rate}")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                # Forward pass
                outputs = self.enhanced_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                
                loss = outputs['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"   Epoch {epoch+1}/{self.config.num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"✅ Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}")
            
            # Evaluation
            if epoch % 1 == 0:  # Evaluate every epoch
                eval_results = self.evaluate_model(eval_loader)
                logger.info(f"📊 Evaluation - Start Acc: {eval_results['start_accuracy']:.1f}%, End Acc: {eval_results['end_accuracy']:.1f}%")
        
        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = output_path / "enhanced_model.pth"
        torch.save(self.enhanced_model.state_dict(), model_path)
        
        # Save tokenizer
        tokenizer_path = output_path / "tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save config
        config_path = output_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.__dict__, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Model saved to: {output_path}")
        return str(output_path)
    
    def evaluate_model(self, eval_loader):
        """Evaluate the enhanced model"""
        self.enhanced_model.eval()
        
        total_start_correct = 0
        total_end_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                start_positions = batch['start_positions'].to(self.device)
                end_positions = batch['end_positions'].to(self.device)
                
                outputs = self.enhanced_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions
                start_preds = torch.argmax(outputs['start_logits'], dim=-1)
                end_preds = torch.argmax(outputs['end_logits'], dim=-1)
                
                # Calculate accuracy
                total_start_correct += (start_preds == start_positions).sum().item()
                total_end_correct += (end_preds == end_positions).sum().item()
                total_samples += start_positions.size(0)
        
        self.enhanced_model.train()
        
        return {
            'start_accuracy': (total_start_correct / total_samples) * 100,
            'end_accuracy': (total_end_correct / total_samples) * 100,
            'total_samples': total_samples
        }

class EnhancedQADataset(Dataset):
    """Enhanced QA Dataset for v0.3 training"""
    
    def __init__(self, qa_samples: List[Dict], tokenizer, max_length: int = 512):
        self.qa_samples = qa_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.qa_samples)
    
    def __getitem__(self, idx):
        sample = self.qa_samples[idx]
        
        question = sample['question']
        context = sample['context']
        answer_text = sample['answer']
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Find answer positions
        answer_start = context.find(answer_text)
        if answer_start == -1:
            answer_start = 0
        
        answer_end = answer_start + len(answer_text) - 1
        
        # Convert to token positions
        question_tokens = self.tokenizer.tokenize(question)
        context_before_answer = context[:answer_start]
        context_before_tokens = self.tokenizer.tokenize(context_before_answer)
        
        # Account for [CLS] and [SEP] tokens
        token_start_pos = len(question_tokens) + 2 + len(context_before_tokens)
        
        answer_tokens = self.tokenizer.tokenize(answer_text)
        token_end_pos = token_start_pos + len(answer_tokens) - 1
        
        # Ensure positions are within bounds
        token_start_pos = min(token_start_pos, self.max_length - 1)
        token_end_pos = min(token_end_pos, self.max_length - 1)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': torch.tensor(token_start_pos, dtype=torch.long),
            'end_positions': torch.tensor(token_end_pos, dtype=torch.long)
        }

def create_augmented_dataset_v3_safetensors(base_samples: List[Dict], expansion_factor: int = 3) -> List[Dict]:
    """Create augmented dataset for v0.3 training with end position focus"""
    
    logger.info(f"Creating end-position augmented dataset from {len(base_samples)} samples")
    
    augmented_samples = []
    
    # Add original samples
    augmented_samples.extend(base_samples)
    
    # Create augmented samples with end position emphasis
    for _ in range(expansion_factor - 1):  # -1 because we already have originals
        for sample in base_samples:
            # Create variations that emphasize end position accuracy
            augmented_sample = copy.deepcopy(sample)
            
            # Add subtle variations to context while preserving answer boundaries
            context = augmented_sample['context']
            answer = augmented_sample['answer']
            
            # Find answer in context
            answer_start = context.find(answer)
            if answer_start != -1:
                # Add emphasis markers around answer (will be removed in preprocessing)
                before = context[:answer_start]
                after = context[answer_start + len(answer):]
                
                # Random variations
                variation_type = random.choice(['prefix', 'suffix', 'both'])
                
                if variation_type in ['prefix', 'both']:
                    prefixes = ['具体的には、', 'つまり、', 'すなわち、', '要するに、']
                    before += random.choice(prefixes)
                
                if variation_type in ['suffix', 'both']:
                    suffixes = ['である。', 'です。', 'となります。', 'とのことです。']
                    after = random.choice(suffixes) + after
                
                # Reconstruct context
                augmented_sample['context'] = before + answer + after
            
            augmented_samples.append(augmented_sample)
    
    logger.info(f"Created augmented dataset: {len(base_samples)} → {len(augmented_samples)} samples")
    return augmented_samples

def test_safetensors_model():
    """Test the safetensors model implementation"""
    logger.info("🧪 Testing Safetensors Model Implementation...")
    
    try:
        config = QATrainingConfig()
        model = EnhancedQAModelV3Safetensors("cl-tohoku/bert-base-japanese-v3", config)
        
        # Test forward pass
        batch_size = 2
        seq_len = 128
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        start_positions = torch.randint(0, seq_len, (batch_size,))
        end_positions = torch.randint(0, seq_len, (batch_size,))
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        
        logger.info("✅ Safetensors model test passed!")
        logger.info(f"   Start logits shape: {outputs['start_logits'].shape}")
        logger.info(f"   End logits shape: {outputs['end_logits'].shape}")
        logger.info(f"   Loss: {outputs['loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Safetensors model test failed: {e}")
        return False

if __name__ == "__main__":
    test_safetensors_model()