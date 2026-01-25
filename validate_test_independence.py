#!/usr/bin/env python3
"""
テストデータと訓練データの独立性検証
"""

import json
from pathlib import Path
from typing import List, Dict, Set


def load_datasets():
    """各種データセットを読み込み"""
    
    datasets = {}
    
    # 訓練データセット読み込み
    train_files = [
        "data/processed/qa_dataset_v2/qa_samples_100.json",
        "data/processed/qa_dataset_v2/qa_samples_500.json", 
        "data/processed/qa_dataset_v2/qa_samples_1000.json"
    ]
    
    for file_path in train_files:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                size = file_path.split('_')[-1].split('.')[0]
                datasets[f"train_{size}"] = json.load(f)
    
    # テストデータセット読み込み
    test_file = "data/processed/test_dataset/test_samples_300.json"
    if Path(test_file).exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            datasets["test_300"] = json.load(f)
    
    return datasets


def extract_questions(samples: List[Dict]) -> Set[str]:
    """サンプルから質問テキストを抽出"""
    return {sample["question"] for sample in samples}


def extract_contexts(samples: List[Dict]) -> Set[str]:
    """サンプルからコンテキストを抽出"""
    return {sample["context"] for sample in samples}


def check_overlap(set1: Set[str], set2: Set[str], label1: str, label2: str):
    """2つのセット間の重複をチェック"""
    
    overlap = set1.intersection(set2)
    
    print(f"\n🔍 Overlap Analysis: {label1} vs {label2}")
    print(f"   {label1}: {len(set1)} unique items")
    print(f"   {label2}: {len(set2)} unique items")
    print(f"   Overlap: {len(overlap)} items ({len(overlap)/max(len(set1), len(set2))*100:.1f}%)")
    
    if overlap:
        print(f"   ⚠️ Found overlapping content:")
        for item in list(overlap)[:3]:  # 最初の3件だけ表示
            print(f"     - {item[:60]}...")
    else:
        print(f"   ✅ No overlap detected - datasets are independent!")
    
    return len(overlap)


def analyze_content_patterns(datasets: Dict):
    """コンテンツパターンの分析"""
    
    print("\n📊 Content Pattern Analysis")
    print("=" * 50)
    
    for name, samples in datasets.items():
        if not samples:
            continue
            
        print(f"\n🏷️ Dataset: {name} ({len(samples)} samples)")
        
        # 災害タイプ分布
        disaster_types = {}
        question_types = {}
        
        for sample in samples:
            # 災害タイプ
            disaster_type = sample.get("disaster_type", "unknown")
            disaster_types[disaster_type] = disaster_types.get(disaster_type, 0) + 1
            
            # 質問タイプ
            question_type = sample.get("question_type", "unknown")
            question_types[question_type] = question_types.get(question_type, 0) + 1
        
        print(f"   Disaster types: {dict(list(disaster_types.items())[:3])}...")
        print(f"   Question types: {question_types}")
        
        # サンプル例
        if samples:
            sample = samples[0]
            print(f"   Sample Q: {sample.get('question', '')[:50]}...")
            print(f"   Sample A: {sample.get('answer', '')[:50]}...")


def main():
    """メイン検証処理"""
    
    print("🔬 Test Data Independence Validation")
    print("=" * 50)
    
    # データセット読み込み
    datasets = load_datasets()
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    print(f"📊 Loaded datasets: {list(datasets.keys())}")
    
    # コンテンツパターン分析
    analyze_content_patterns(datasets)
    
    # 独立性検証
    print("\n🔍 Independence Validation")
    print("=" * 50)
    
    if "test_300" not in datasets:
        print("❌ Test dataset not found!")
        return
    
    test_questions = extract_questions(datasets["test_300"])
    test_contexts = extract_contexts(datasets["test_300"])
    
    total_overlaps = 0
    
    # 各訓練データセットとの比較
    for train_name, train_samples in datasets.items():
        if not train_name.startswith("train_"):
            continue
            
        train_questions = extract_questions(train_samples)
        train_contexts = extract_contexts(train_samples)
        
        # 質問の重複チェック
        q_overlap = check_overlap(test_questions, train_questions, "Test Questions", f"Train Questions ({train_name})")
        
        # コンテキストの重複チェック
        c_overlap = check_overlap(test_contexts, train_contexts, "Test Contexts", f"Train Contexts ({train_name})")
        
        total_overlaps += q_overlap + c_overlap
    
    # 最終結果
    print(f"\n🎯 Final Independence Assessment")
    print("=" * 50)
    
    if total_overlaps == 0:
        print("✅ Test dataset is COMPLETELY INDEPENDENT from training data!")
        print("🎉 Safe to use for unbiased evaluation")
    else:
        print(f"⚠️ Found {total_overlaps} overlapping items")
        print("🔄 Consider regenerating test data or removing overlaps")
    
    return total_overlaps


if __name__ == "__main__":
    main()