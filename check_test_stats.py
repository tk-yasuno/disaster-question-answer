#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
テストデータの統計情報を確認するスクリプト
"""

import json
import pandas as pd

def check_test_data_stats():
    # テストデータを読み込み
    with open('data/processed/test_dataset/test_samples_300.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print('=== テストデータ統計情報 ===')
    print(f'総サンプル数: {len(test_data)}')

    # 災害タイプ別
    disaster_counts = {}
    for sample in test_data:
        dtype = sample['disaster_type']
        disaster_counts[dtype] = disaster_counts.get(dtype, 0) + 1

    print('\n災害タイプ別分布:')
    for dtype, count in disaster_counts.items():
        print(f'  {dtype}: {count}')

    # 難易度別
    difficulty_counts = {}
    for sample in test_data:
        diff = sample['difficulty_level']
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    print('\n難易度別分布:')
    for diff, count in difficulty_counts.items():
        print(f'  {diff}: {count}')

    # 質問タイプ別
    qtype_counts = {}
    for sample in test_data:
        qtype = sample['question_type']
        qtype_counts[qtype] = qtype_counts.get(qtype, 0) + 1

    print('\n質問タイプ別分布:')
    for qtype, count in qtype_counts.items():
        print(f'  {qtype}: {count}')

    # ユニークな質問数
    unique_questions = set()
    for sample in test_data:
        unique_questions.add(sample['question'])

    print(f'\nユニークな質問数: {len(unique_questions)}')

    # サンプル質問の例
    print('\n=== テスト質問の例 ===')
    for i, sample in enumerate(test_data[:5]):
        print(f'{i+1}. 災害タイプ: {sample["disaster_type"]}')
        print(f'   質問: {sample["question"]}')
        print(f'   難易度: {sample["difficulty_level"]}')
        print()

    # 各災害タイプからランダムに1つずつ表示
    print('\n=== 各災害タイプの質問例 ===')
    disaster_samples = {}
    for sample in test_data:
        dtype = sample['disaster_type']
        if dtype not in disaster_samples:
            disaster_samples[dtype] = sample

    for dtype, sample in disaster_samples.items():
        print(f'【{dtype.upper()}】')
        print(f'  質問: {sample["question"]}')
        print(f'  コンテキスト: {sample["context"][:100]}...')
        print(f'  回答: {sample["answer"][:60]}...')
        print(f'  難易度: {sample["difficulty_level"]}')
        print()

if __name__ == "__main__":
    check_test_data_stats()