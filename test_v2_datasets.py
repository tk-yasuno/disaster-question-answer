#!/usr/bin/env python3
"""
v0.2: 段階的評価・ベンチマークスクリプト
100→500→1000サンプルでのファインチューニング比較
"""

import json
import time
import torch
from pathlib import Path
from src.modules.fine_tuning import DisaQuADDataset, LoRATrainer
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_progressive_evaluation():
    """段階的評価の実行"""
    
    dataset_sizes = [100, 500, 1000]
    results = {}
    
    print("🎯 Progressive Fine-tuning Evaluation")
    print("=" * 50)
    
    for size in dataset_sizes:
        print(f"\n📊 Step: Training with {size} samples")
        print("-" * 30)
        
        try:
            # データセット読み込みテスト
            dataset = DisaQuADDataset(
                data_dir=Path("data/processed/qa_dataset_v2"),
                dataset_size=size
            )
            
            print(f"✅ Dataset loaded: {len(dataset.samples)} samples")
            
            # サンプルデータの確認
            if len(dataset.samples) > 0:
                sample = dataset.samples[0]
                print(f"📝 Sample data structure:")
                for key, value in sample.items():
                    print(f"  - {key}: {str(value)[:50]}...")
                
                # データセット分割テスト
                train_dataset, eval_dataset = dataset.split(0.8)
                print(f"📈 Train samples: {len(train_dataset.samples)}")
                print(f"📉 Eval samples: {len(eval_dataset.samples)}")
                
                # 簡単なメトリクス収集
                results[size] = {
                    "total_samples": len(dataset.samples),
                    "train_samples": len(train_dataset.samples), 
                    "eval_samples": len(eval_dataset.samples),
                    "load_time": time.time(),
                    "status": "success"
                }
            else:
                print("❌ No samples loaded")
                results[size] = {"status": "failed", "error": "No samples"}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results[size] = {"status": "failed", "error": str(e)}
    
    # 結果サマリー
    print("\n📊 Progressive Evaluation Summary")
    print("=" * 50)
    
    for size, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            print(f"✅ {size} samples: {result['train_samples']} train, {result['eval_samples']} eval")
        else:
            print(f"❌ {size} samples: {result.get('error', 'Unknown error')}")
    
    return results


def test_single_fine_tuning(dataset_size=100):
    """単一サイズでのファインチューニングテスト"""
    
    print(f"\n🧪 Testing Fine-tuning with {dataset_size} samples")
    print("-" * 40)
    
    try:
        # データセット準備
        dataset = DisaQuADDataset(
            data_dir=Path("data/processed/qa_dataset_v2"),
            dataset_size=dataset_size
        )
        
        if len(dataset.samples) == 0:
            print("❌ No samples loaded - aborting")
            return None
        
        train_dataset, eval_dataset = dataset.split(0.8)
        print(f"📊 Dataset split: {len(train_dataset.samples)} train, {len(eval_dataset.samples)} eval")
        
        # 1サンプルでの動作確認
        if len(train_dataset.samples) > 0:
            sample = train_dataset.samples[0]
            print(f"📝 Processing test sample:")
            print(f"  Question: {sample['question'][:50]}...")
            print(f"  Context: {sample['context'][:50]}...")
            print(f"  Answer: {sample['answer'][:50]}...")
            
        print("✅ Dataset preparation successful")
        return {"status": "success", "samples": len(dataset.samples)}
        
    except Exception as e:
        print(f"❌ Error in fine-tuning test: {e}")
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    # 段階評価実行
    progressive_results = run_progressive_evaluation()
    
    # 単一テスト実行
    single_test_result = test_single_fine_tuning(100)
    
    print(f"\n🎯 v0.2 Progressive Dataset Evaluation Complete")