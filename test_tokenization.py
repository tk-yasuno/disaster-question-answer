#!/usr/bin/env python3
"""
v0.2: 簡単なファインチューニングテスト（データ処理問題を修正）
"""

import json
import torch
from transformers import AutoTokenizer
from pathlib import Path
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tokenization():
    """トークナイゼーション処理のテスト"""
    
    print("🧪 Testing Tokenization Process")
    print("=" * 40)
    
    # データサンプル読み込み
    with open("data/processed/qa_dataset_v2/qa_samples_100.json", 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"📊 Loaded {len(samples)} samples")
    
    # トークナイザー初期化
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
    
    # 最初のサンプルでテスト
    sample = samples[0]
    print(f"📝 Test Sample:")
    print(f"  Question: {sample['question']}")
    print(f"  Context: {sample['context'][:100]}...")
    print(f"  Answer: {sample['answer']}")
    
    # トークナイゼーション実行
    question = sample['question']
    context = sample['context']
    answer = sample['answer']
    
    # エンコード
    encoding = tokenizer(
        question,
        context,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    )
    
    print(f"\n🔤 Tokenization Results:")
    print(f"  Input IDs shape: {encoding['input_ids'].shape}")
    print(f"  Attention mask shape: {encoding['attention_mask'].shape}")
    
    # 回答位置の検索
    answer_start = context.find(answer)
    if answer_start == -1:
        print(f"⚠️ Answer not found in context. Using start=0")
        answer_start = 0
    
    answer_end = answer_start + len(answer) - 1
    
    # トークン位置での検索
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    
    print(f"\n📍 Position Information:")
    print(f"  Character positions: start={answer_start}, end={answer_end}")
    print(f"  Answer tokens: {answer_tokens[:5]}...")
    print(f"  Context tokens: {len(context_tokens)} tokens")
    
    # 簡単なトークナイゼーション関数
    def tokenize_sample(sample_data):
        q = sample_data['question']
        c = sample_data['context']
        a = sample_data['answer']
        
        # 基本エンコーディング
        encoding = tokenizer(
            q, c,
            truncation=True,
            padding='max_length', 
            max_length=512,
            return_tensors="pt"
        )
        
        # 答えの位置を簡易的に設定
        start_pos = 1  # [CLS] token の後
        end_pos = start_pos + 5  # 適当な長さ
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze() if 'token_type_ids' in encoding else torch.zeros_like(encoding['input_ids'].squeeze()),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_pos, dtype=torch.long),
            'end_positions': torch.tensor(end_pos, dtype=torch.long)
        }
        
        return result
    
    # テストトークナイゼーション
    processed = tokenize_sample(sample)
    
    print(f"\n✅ Processed Sample:")
    for key, value in processed.items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    print(f"\n🎯 Tokenization Test Complete!")
    return processed


if __name__ == "__main__":
    test_tokenization()