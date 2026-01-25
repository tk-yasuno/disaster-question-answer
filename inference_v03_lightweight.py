#!/usr/bin/env python3
"""
v0.3 Lightweight Model Inference Script
軽量版v0.3モデルによる質問応答推論

Usage:
python inference_v03_lightweight.py --question "地震が発生したときに何をすべきですか？" --context "地震が発生した場合、まず自分の安全を確保することが重要です。机の下に隠れるか、頭を保護してください。その後、火の元を確認し、ガスの元栓を閉めてください。"
"""

import argparse
import json
import logging
import pickle
import numpy as np
from pathlib import Path
import sys
import os
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add src to path  
sys.path.append(str(Path(__file__).parent / "src"))

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
        features[0] = position / len(tokens) if len(tokens) > 0 else 0  # Relative position
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
        features[9] = sum(1 for t in context if t in self.disaster_terms) / len(context) if context else 0
        
        # Position indicator features (crucial for end position)
        features[10] = 1 if current_token in self.position_indicators else 0
        features[11] = sum(1 for t in context if t in self.position_indicators) / len(context) if context else 0
        
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
        features[17] = np.mean([len(t) for t in context]) if context else 0
        
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
        
        # Initialize weights (will be loaded from saved model)
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.w3 = None
        self.b3 = None
    
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
    
    def load_weights(self, weights_dict):
        """Load weights from dictionary"""
        self.w1 = weights_dict['w1']
        self.b1 = weights_dict['b1']
        self.w2 = weights_dict['w2']
        self.b2 = weights_dict['b2']
        self.w3 = weights_dict['w3']
        self.b3 = weights_dict['b3']

class LightweightQAInference:
    """Lightweight QA model for inference"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = JapaneseTokenizer()
        
        # Load model
        self.load_model()
        
        # Pattern-based enhancer for end positions
        self.end_patterns = [
            r'です$', r'である$', r'だ$', r'ます$', r'ました$',
            r'してください$', r'できます$', r'になります$',
            r'とのことです$', r'とされています$'
        ]
        
        logger.info(f"✅ v0.3 Lightweight QA Inference initialized")
        logger.info(f"   Model loaded from: {model_path}")
        logger.info(f"   Pattern enhancer: {len(self.end_patterns)} patterns")
    
    def load_model(self):
        """Load the trained lightweight model"""
        model_file = Path(self.model_path) / 'lightweight_model.pkl'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        # Initialize models
        self.start_model = SimpleMLP(64, 128, 1)
        self.end_model = SimpleMLP(64, 128, 1)
        
        # Load weights
        self.start_model.load_weights(model_data['start_model_weights'])
        self.end_model.load_weights(model_data['end_model_weights'])
        
        # Load config
        self.config = model_data['config']
        
        logger.info("✅ Model weights loaded successfully")
    
    def predict_positions(self, question: str, context: str) -> Tuple[int, int, List[float], List[float]]:
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
        
        return start_pos, end_pos, start_scores, end_scores
    
    def extract_answer(self, context: str, start_pos: int, end_pos: int) -> str:
        """Extract answer text from context using predicted positions"""
        tokens = self.tokenizer.tokenize(context)
        
        if start_pos >= len(tokens):
            return ""
        
        if end_pos >= len(tokens):
            end_pos = len(tokens) - 1
        
        answer_tokens = tokens[start_pos:end_pos + 1]
        return ''.join(answer_tokens)
    
    def answer_question(self, question: str, context: str) -> Dict:
        """Answer a question given context"""
        
        logger.info(f"🤔 Question: {question}")
        logger.info(f"📄 Context: {context[:100]}...")
        
        # Predict positions
        start_pos, end_pos, start_scores, end_scores = self.predict_positions(question, context)
        
        # Extract answer
        answer = self.extract_answer(context, start_pos, end_pos)
        
        # Calculate confidence
        tokens = self.tokenizer.tokenize(context)
        start_confidence = start_scores[start_pos] if start_pos < len(start_scores) else 0
        end_confidence = end_scores[end_pos] if end_pos < len(end_scores) else 0
        overall_confidence = (start_confidence + end_confidence) / 2
        
        result = {
            'question': question,
            'context': context,
            'answer': answer,
            'start_position': int(start_pos),
            'end_position': int(end_pos),
            'start_confidence': float(start_confidence),
            'end_confidence': float(end_confidence),
            'overall_confidence': float(overall_confidence),
            'tokens': tokens,
            'start_token': tokens[start_pos] if start_pos < len(tokens) else '',
            'end_token': tokens[end_pos] if end_pos < len(tokens) else ''
        }
        
        logger.info(f"✅ Answer: {answer}")
        logger.info(f"📊 Position: {start_pos} → {end_pos} (confidence: {overall_confidence:.3f})")
        
        return result

def run_interactive_mode(inference_model):
    """Interactive question-answering mode"""
    logger.info("\n🎯 v0.3 Lightweight QA - Interactive Mode")
    logger.info("Enter 'quit' to exit\n")
    
    while True:
        try:
            question = input("❓ Question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            context = input("📄 Context: ").strip()
            if not context:
                print("Context is required!")
                continue
            
            # Get answer
            result = inference_model.answer_question(question, context)
            
            print(f"\n🎯 Result:")
            print(f"   Answer: '{result['answer']}'")
            print(f"   Position: {result['start_position']} → {result['end_position']}")
            print(f"   Tokens: '{result['start_token']}' → '{result['end_token']}'")
            print(f"   Confidence: {result['overall_confidence']:.3f}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    logger.info("👋 Goodbye!")

def run_demo_examples(inference_model):
    """Run demo examples"""
    logger.info("🎬 Running v0.3 Lightweight Demo Examples...")
    
    demo_cases = [
        {
            'question': '地震が発生したときに何をすべきですか？',
            'context': '地震が発生した場合、まず自分の安全を確保することが重要です。机の下に隠れるか、頭を保護してください。その後、火の元を確認し、ガスの元栓を閉めてください。'
        },
        {
            'question': '津波警報が出たらどうすればいいですか？',
            'context': '津波警報が発表されたら、直ちに高台や頑丈な建物の3階以上に避難してください。海や川から離れ、津波の到達予想時間を確認してください。'
        },
        {
            'question': '台風に備えて何を準備すべきですか？',
            'context': '台風の接近に備えて、非常用品を準備してください。懐中電灯、ラジオ、飲料水、非常食、救急用品を用意し、窓ガラスには飛散防止フィルムを貼ってください。'
        },
        {
            'question': '火災が発生した時の対応は？',
            'context': '火災が発生した場合は、まず119番通報をしてください。小さな火であれば消火器で消火を試みますが、火が大きくなったらすぐに避難してください。煙を吸わないよう低い姿勢で避難します。'
        }
    ]
    
    for i, case in enumerate(demo_cases, 1):
        logger.info(f"\n🎯 Demo Case {i}:")
        result = inference_model.answer_question(case['question'], case['context'])
        
        print(f"📝 Result Summary:")
        print(f"   Q: {case['question']}")
        print(f"   A: '{result['answer']}'")
        print(f"   Position: {result['start_position']} → {result['end_position']}")
        print(f"   Confidence: {result['overall_confidence']:.3f}")
        print(f"   Tokens: '{result['start_token']}' → '{result['end_token']}'")
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description='v0.3 Lightweight QA Inference')
    parser.add_argument('--model-path', default='models/v03_lightweight_enhanced', help='Path to trained model')
    parser.add_argument('--mode', choices=['demo', 'interactive', 'single'], default='demo', help='Inference mode')
    parser.add_argument('--question', help='Question for single mode')
    parser.add_argument('--context', help='Context for single mode')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting v0.3 Lightweight QA Inference")
    logger.info(f"   Model Path: {args.model_path}")
    logger.info(f"   Mode: {args.mode}")
    
    try:
        # Initialize inference model
        inference_model = LightweightQAInference(args.model_path)
        
        if args.mode == 'demo':
            run_demo_examples(inference_model)
        elif args.mode == 'interactive':
            run_interactive_mode(inference_model)
        elif args.mode == 'single':
            if not args.question or not args.context:
                logger.error("Question and context are required for single mode")
                return
            
            result = inference_model.answer_question(args.question, args.context)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        logger.info("✅ v0.3 Lightweight QA Inference completed!")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()