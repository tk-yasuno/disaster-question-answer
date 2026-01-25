#!/usr/bin/env python3
"""
v0.3 Lightweight End Position Enhanced Training Pipeline
PyTorchセキュリティ制限を回避した軽量実装

Target: Improve End Position accuracy from 8.7% to 25%+
Approach: Pattern-based + Simple ML + Enhanced tokenization
"""

import argparse
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os
import re
from collections import defaultdict, Counter
import pickle
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_v03_lightweight.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LightweightConfig:
    """Lightweight model configuration"""
    max_seq_length: int = 512
    context_window: int = 64
    end_position_weight: float = 2.5
    feature_dim: int = 128
    learning_rate: float = 0.01
    epochs: int = 5
    batch_size: int = 16

class JapaneseTokenizer:
    """Lightweight Japanese tokenizer for disaster QA"""
    
    def __init__(self):
        # Japanese text patterns
        self.hiragana = re.compile(r'[ひ-ゟ]')
        self.katakana = re.compile(r'[ア-ヿ]')
        self.kanji = re.compile(r'[一-龯]')
        self.number = re.compile(r'[0-9０-９]')
        self.punctuation = re.compile(r'[。、！？．，]')
        
        # Disaster-specific terms (high importance for end positions)
        self.disaster_terms = {
            '地震', '津波', '台風', '洪水', '火災', '避難', '緊急', '災害',
            '安全', '危険', '警報', '注意', '対策', '準備', '救助', '支援'
        }
        
        # Position indicators (important for end detection)
        self.position_indicators = {
            'まで', 'から', 'に', 'で', 'を', 'が', 'は', 'の', 'と', 'も',
            'である', 'です', 'だ', 'ます', 'してください', 'できます'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Simple Japanese tokenization"""
        # Basic character-level tokenization with word boundary detection
        tokens = []
        current_token = ""
        
        for i, char in enumerate(text):
            if self.punctuation.match(char):
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            elif char in ' 　\n\t':  # Whitespace
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def extract_features(self, text: str, position: int) -> np.ndarray:
        """Extract position-aware features"""
        tokens = self.tokenize(text)
        
        if position >= len(tokens):
            position = len(tokens) - 1
        
        features = np.zeros(64)  # Feature vector
        
        # Position features
        features[0] = position / len(tokens)  # Relative position
        features[1] = position  # Absolute position
        features[2] = len(tokens) - position  # Distance from end
        
        # Context features
        context_start = max(0, position - 5)
        context_end = min(len(tokens), position + 5)
        context = tokens[context_start:context_end]
        
        # Character type features
        current_token = tokens[position] if position < len(tokens) else ""
        features[3] = 1 if self.hiragana.search(current_token) else 0
        features[4] = 1 if self.katakana.search(current_token) else 0
        features[5] = 1 if self.kanji.search(current_token) else 0
        features[6] = 1 if self.number.search(current_token) else 0
        features[7] = 1 if self.punctuation.search(current_token) else 0
        
        # Disaster term features
        features[8] = 1 if current_token in self.disaster_terms else 0
        features[9] = sum(1 for t in context if t in self.disaster_terms) / len(context)
        
        # Position indicator features (crucial for end position)
        features[10] = 1 if current_token in self.position_indicators else 0
        features[11] = sum(1 for t in context if t in self.position_indicators) / len(context)
        
        # N-gram features
        if position > 0:
            prev_token = tokens[position - 1]
            features[12] = 1 if prev_token in self.position_indicators else 0
            features[13] = 1 if prev_token in self.disaster_terms else 0
        
        if position < len(tokens) - 1:
            next_token = tokens[position + 1]
            features[14] = 1 if next_token in self.position_indicators else 0
            features[15] = 1 if next_token in self.disaster_terms else 0
        
        # Token length features
        features[16] = len(current_token)
        features[17] = np.mean([len(t) for t in context])
        
        # Sentence boundary features
        features[18] = 1 if any(p in current_token for p in '。！？') else 0
        features[19] = 1 if any(p in ''.join(context) for p in '。！？') else 0
        
        return features

class SimpleMLP:
    """Simple Multi-Layer Perceptron for position prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((1, hidden_dim))
        self.w3 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b3 = np.zeros((1, output_dim))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        
        return self.a3
    
    def predict(self, x):
        return self.forward(x)

class LightweightQAModel:
    """Lightweight QA model with end position enhancement"""
    
    def __init__(self, config: LightweightConfig):
        self.config = config
        self.tokenizer = JapaneseTokenizer()
        
        # Position prediction models
        self.start_model = SimpleMLP(64, 128, 1)
        self.end_model = SimpleMLP(64, 128, 1)
        
        # Pattern-based enhancer for end positions
        self.end_patterns = [
            r'です$', r'である$', r'だ$', r'ます$', r'ました$',
            r'してください$', r'できます$', r'になります$',
            r'とのことです$', r'とされています$'
        ]
        
        logger.info(f"✅ Lightweight QA Model initialized")
        logger.info(f"   End Position Weight: {config.end_position_weight}x")
        logger.info(f"   Feature Dimension: 64")
        logger.info(f"   Pattern Enhancer: {len(self.end_patterns)} patterns")
    
    def extract_answer_positions(self, question: str, context: str, answer: str) -> Tuple[int, int]:
        """Extract answer start and end positions with enhanced end detection"""
        
        # Tokenize context
        context_tokens = self.tokenizer.tokenize(context)
        answer_tokens = self.tokenizer.tokenize(answer)
        
        # Find answer in context
        start_pos = -1
        end_pos = -1
        
        for i in range(len(context_tokens) - len(answer_tokens) + 1):
            if context_tokens[i:i+len(answer_tokens)] == answer_tokens:
                start_pos = i
                end_pos = i + len(answer_tokens) - 1
                break
        
        # If exact match not found, use fuzzy matching
        if start_pos == -1:
            answer_text = ''.join(answer_tokens)
            context_text = ''.join(context_tokens)
            char_start = context_text.find(answer_text)
            
            if char_start != -1:
                # Convert character position to token position
                char_count = 0
                for i, token in enumerate(context_tokens):
                    if char_count >= char_start:
                        start_pos = i
                        break
                    char_count += len(token)
                
                # Find end position
                char_end = char_start + len(answer_text)
                char_count = 0
                for i, token in enumerate(context_tokens):
                    char_count += len(token)
                    if char_count >= char_end:
                        end_pos = i
                        break
        
        # Ensure valid positions
        if start_pos == -1:
            start_pos = 0
        if end_pos == -1:
            end_pos = min(start_pos + len(answer_tokens) - 1, len(context_tokens) - 1)
        
        return start_pos, end_pos
    
    def predict_positions(self, question: str, context: str) -> Tuple[int, int]:
        """Predict answer start and end positions"""
        
        tokens = self.tokenizer.tokenize(context)
        start_scores = []
        end_scores = []
        
        # Score each position
        for i in range(len(tokens)):
            features = self.tokenizer.extract_features(context, i)
            
            start_score = self.start_model.predict(features.reshape(1, -1))[0, 0]
            end_score = self.end_model.predict(features.reshape(1, -1))[0, 0]
            
            # Pattern-based end position enhancement
            if any(re.search(pattern, tokens[i]) for pattern in self.end_patterns):
                end_score *= self.config.end_position_weight
            
            start_scores.append(start_score)
            end_scores.append(end_score)
        
        # Find best positions
        start_pos = np.argmax(start_scores)
        end_pos = np.argmax(end_scores)
        
        # Ensure end >= start
        if end_pos < start_pos:
            end_pos = start_pos
        
        return start_pos, end_pos
    
    def train(self, qa_samples: List[Dict]) -> Dict[str, float]:
        """Train the lightweight model"""
        
        logger.info(f"🚀 Starting Lightweight Training...")
        logger.info(f"   Samples: {len(qa_samples)}")
        logger.info(f"   Epochs: {self.config.epochs}")
        logger.info(f"   End Weight: {self.config.end_position_weight}x")
        
        # Prepare training data
        X_train = []
        y_start_train = []
        y_end_train = []
        
        for sample in qa_samples:
            question = sample['question']
            context = sample['context']
            answer = sample['answer']
            
            start_pos, end_pos = self.extract_answer_positions(question, context, answer)
            tokens = self.tokenizer.tokenize(context)
            
            # Create training examples for each position
            for i in range(len(tokens)):
                features = self.tokenizer.extract_features(context, i)
                X_train.append(features)
                y_start_train.append(1.0 if i == start_pos else 0.0)
                y_end_train.append(1.0 if i == end_pos else 0.0)
        
        X_train = np.array(X_train)
        y_start_train = np.array(y_start_train).reshape(-1, 1)
        y_end_train = np.array(y_end_train).reshape(-1, 1)
        
        logger.info(f"📊 Training Data: {len(X_train)} examples")
        
        # Simple gradient descent training
        batch_size = self.config.batch_size
        learning_rate = self.config.learning_rate
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_start_shuffled = y_start_train[indices]
            y_end_shuffled = y_end_train[indices]
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y_start = y_start_shuffled[i:i+batch_size]
                batch_y_end = y_end_shuffled[i:i+batch_size]
                
                # Forward pass
                start_pred = self.start_model.forward(batch_X)
                end_pred = self.end_model.forward(batch_X)
                
                # Loss calculation with end position weighting
                start_loss = np.mean((start_pred - batch_y_start) ** 2)
                end_loss = np.mean((end_pred - batch_y_end) ** 2) * self.config.end_position_weight
                
                total_loss = start_loss + end_loss
                epoch_loss += total_loss
                num_batches += 1
                
                # Simple gradient updates (simplified backprop)
                start_grad = 2 * (start_pred - batch_y_start) / batch_size
                end_grad = 2 * (end_pred - batch_y_end) * self.config.end_position_weight / batch_size
                
                # Update start model
                self.start_model.w3 -= learning_rate * np.dot(self.start_model.a2.T, start_grad)
                self.start_model.b3 -= learning_rate * np.sum(start_grad, axis=0, keepdims=True)
                
                # Update end model  
                self.end_model.w3 -= learning_rate * np.dot(self.end_model.a2.T, end_grad)
                self.end_model.b3 -= learning_rate * np.sum(end_grad, axis=0, keepdims=True)
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"✅ Epoch {epoch+1}/{self.config.epochs} - Loss: {avg_loss:.4f}")
        
        logger.info(f"🎉 Lightweight Training completed!")
        
        return {'training_loss': avg_loss}
    
    def evaluate(self, qa_samples: List[Dict]) -> Dict[str, float]:
        """Evaluate the model"""
        
        logger.info(f"🔍 Evaluating Lightweight Model...")
        
        start_correct = 0
        end_correct = 0
        total = len(qa_samples)
        
        for sample in qa_samples:
            question = sample['question']
            context = sample['context'] 
            answer = sample['answer']
            
            # True positions
            true_start, true_end = self.extract_answer_positions(question, context, answer)
            
            # Predicted positions
            pred_start, pred_end = self.predict_positions(question, context)
            
            # Check accuracy
            if pred_start == true_start:
                start_correct += 1
            if pred_end == true_end:
                end_correct += 1
        
        start_accuracy = (start_correct / total) * 100
        end_accuracy = (end_correct / total) * 100
        
        logger.info(f"📊 Evaluation Results:")
        logger.info(f"   Start Position Accuracy: {start_accuracy:.1f}%")
        logger.info(f"   End Position Accuracy: {end_accuracy:.1f}%")
        logger.info(f"   Total Samples: {total}")
        
        return {
            'start_accuracy': start_accuracy,
            'end_accuracy': end_accuracy,
            'total_samples': total
        }
    
    def save(self, model_path: str):
        """Save the lightweight model"""
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'config': self.config,
            'start_model_weights': {
                'w1': self.start_model.w1,
                'b1': self.start_model.b1,
                'w2': self.start_model.w2,
                'b2': self.start_model.b2,
                'w3': self.start_model.w3,
                'b3': self.start_model.b3,
            },
            'end_model_weights': {
                'w1': self.end_model.w1,
                'b1': self.end_model.b1,
                'w2': self.end_model.w2,
                'b2': self.end_model.b2,
                'w3': self.end_model.w3,
                'b3': self.end_model.b3,
            }
        }
        
        with open(Path(model_path) / 'lightweight_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 Model saved to: {model_path}")

def load_qa_dataset(dataset_path: str, max_samples: int = None) -> List[Dict]:
    """Load QA dataset from JSON file"""
    logger.info(f"📂 Loading dataset: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    logger.info(f"✅ Loaded {len(data)} QA samples")
    return data

def create_augmented_dataset(base_samples: List[Dict], expansion_factor: int = 3) -> List[Dict]:
    """Create augmented dataset with end position focus"""
    
    logger.info(f"🔄 Creating augmented dataset...")
    
    import copy
    import random
    
    augmented_samples = []
    augmented_samples.extend(base_samples)
    
    # Create augmented samples
    for _ in range(expansion_factor - 1):
        for sample in base_samples:
            augmented_sample = copy.deepcopy(sample)
            
            context = augmented_sample['context']
            answer = augmented_sample['answer']
            
            # Find answer in context
            answer_start = context.find(answer)
            if answer_start != -1:
                before = context[:answer_start]
                after = context[answer_start + len(answer):]
                
                # Add variations that emphasize end positions
                variations = random.choice([
                    ('', 'である。'),
                    ('', 'です。'),
                    ('つまり、', ''),
                    ('', 'となります。'),
                    ('具体的には、', 'とのことです。')
                ])
                
                prefix, suffix = variations
                augmented_sample['context'] = before + prefix + answer + suffix + after
            
            augmented_samples.append(augmented_sample)
    
    logger.info(f"✅ Augmented dataset: {len(base_samples)} → {len(augmented_samples)} samples")
    return augmented_samples

def ensure_directories():
    """Ensure required directories exist"""
    dirs = ['logs', 'data/processed/qa_dataset_v3', 'models', 'temp_eval_lightweight']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='v0.3 Lightweight Enhanced QA Training Pipeline')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'], default='both', help='Operation mode')
    parser.add_argument('--dataset-size', type=int, default=500, help='Dataset size')
    parser.add_argument('--use-augmented', action='store_true', help='Use augmented dataset')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--end-weight', type=float, default=2.5, help='End position weight')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting v0.3 Lightweight Enhanced Pipeline")
    
    ensure_directories()
    
    try:
        # Configuration
        config = LightweightConfig(
            epochs=args.epochs,
            end_position_weight=args.end_weight
        )
        
        # Load dataset
        base_dataset_path = f"data/processed/qa_dataset_v2/qa_samples_{args.dataset_size}.json"
        
        if args.use_augmented:
            logger.info(f"📊 Using augmented dataset (3x expansion)")
            base_samples = load_qa_dataset(base_dataset_path, args.dataset_size)
            qa_samples = create_augmented_dataset(base_samples)
        else:
            qa_samples = load_qa_dataset(base_dataset_path, args.dataset_size)
        
        # Split data
        split_idx = int(len(qa_samples) * 0.8)
        train_samples = qa_samples[:split_idx]
        eval_samples = qa_samples[split_idx:]
        
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"   Total: {len(qa_samples)} samples")
        logger.info(f"   Training: {len(train_samples)} samples") 
        logger.info(f"   Evaluation: {len(eval_samples)} samples")
        
        # Initialize model
        model = LightweightQAModel(config)
        
        if args.mode in ['train', 'both']:
            # Train model
            logger.info(f"🏋️ Training Mode: Lightweight End Position Enhancement")
            train_results = model.train(train_samples)
            
            # Save model
            model_path = "models/v03_lightweight_enhanced"
            model.save(model_path)
            
            logger.info(f"✅ Training completed!")
        
        if args.mode in ['evaluate', 'both']:
            # Evaluate model
            logger.info(f"📊 Evaluation Mode")
            eval_results = model.evaluate(eval_samples)
            
            logger.info(f"🎯 Final v0.3 Lightweight Results:")
            logger.info(f"   End Position Accuracy: {eval_results['end_accuracy']:.1f}% (Target: 25%+)")
            logger.info(f"   Start Position Accuracy: {eval_results['start_accuracy']:.1f}%")
            
            # Success check
            if eval_results['end_accuracy'] >= 25.0:
                logger.info("🎉 SUCCESS: End Position accuracy target achieved!")
            elif eval_results['end_accuracy'] >= 20.0:
                logger.info("✅ GOOD: Significant improvement achieved!")
            elif eval_results['end_accuracy'] >= 15.0:
                logger.info("📈 PROGRESS: Noticeable improvement made!")
            else:
                logger.info("🔄 BASELINE: Foundation established for further optimization")
        
        logger.info("✅ v0.3 Lightweight Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ v0.3 Lightweight Pipeline Failed: {e}")
        raise

if __name__ == "__main__":
    main()