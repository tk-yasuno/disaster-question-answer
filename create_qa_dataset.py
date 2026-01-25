"""
災害QAデータセット作成スクリプト
既存のPDFデータから疑似的なQAペアを生成
"""
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.modules.data_processor import DisasterDataProcessor
import pickle
from pathlib import Path

def create_sample_qa_dataset():
    """
    災害文書データから疑似QAデータセットを作成
    """
    print("=== 災害QAデータセット作成開始 ===")
    
    # データ処理器の初期化
    processor = DisasterDataProcessor()
    
    # 既存の抽出済みドキュメントを読み込み
    processed_dir = Path("data/processed")
    documents_file = processed_dir / "extracted_documents.pkl"
    
    if not documents_file.exists():
        print("❌ 抽出済みドキュメントが見つかりません。先にPDF処理を実行してください。")
        return
        
    with open(documents_file, 'rb') as f:
        documents = pickle.load(f)
    
    print(f"📄 読み込み済みドキュメント数: {len(documents)}")
    
    # サンプルQAペアを作成
    sample_qa_pairs = [
        {
            "id": "disaster_qa_001",
            "question": "津波警報が発令されたらどうすべきですか？",
            "context": "津波警報が発令された場合、直ちに海岸や河川から離れ、高台や津波避難ビルなど安全な場所に避難してください。避難の際は徒歩で移動し、自動車の使用は避けてください。",
            "answer": "直ちに高台や津波避難ビルに避難する",
            "start_pos": 15,
            "end_pos": 33,
            "category": "tsunami",
            "question_type": "WhatAct"
        },
        {
            "id": "disaster_qa_002", 
            "question": "地震が発生した時の初期対応は何ですか？",
            "context": "地震が発生した場合、まず身の安全を確保してください。机の下に隠れる、頭を保護するなどして身を守り、揺れが収まってから火の始末や出口の確保を行います。",
            "answer": "身の安全を確保し、机の下に隠れるなどして身を守る",
            "start_pos": 18,
            "end_pos": 44,
            "category": "earthquake", 
            "question_type": "WhatAct"
        },
        {
            "id": "disaster_qa_003",
            "question": "台風接近時の事前準備として重要なことは？",
            "context": "台風接近前には、飛散しやすい物の固定、食料・水・懐中電灯・ラジオなどの非常用品の準備、避難場所や避難経路の確認を行うことが重要です。",
            "answer": "飛散物の固定、非常用品の準備、避難経路の確認",
            "start_pos": 10,
            "end_pos": 35,
            "category": "typhoon",
            "question_type": "WhatAct"
        },
        {
            "id": "disaster_qa_004",
            "question": "避難所に到着したら最初に何をすべきですか？",
            "context": "避難所に到着したら、まず受付で避難者名簿に記入し、避難所のルールや設備の説明を受けてください。その後、指定された場所に荷物を置き、必要に応じて健康状態の報告を行います。",
            "answer": "受付で避難者名簿に記入し、避難所のルールを確認する",
            "start_pos": 12,
            "end_pos": 40,
            "category": "evacuation",
            "question_type": "WhatAct"
        },
        {
            "id": "disaster_qa_005",
            "question": "緊急地震速報を受信したらどう行動すべきですか？",
            "context": "緊急地震速報を受信した場合、すぐに行動を開始してください。屋内では机の下に隠れ頭を保護し、屋外では建物から離れて安全な場所に移動します。運転中の場合は安全な場所に停車してください。",
            "answer": "すぐに安全な場所に避難し、頭を保護する",
            "start_pos": 20,
            "end_pos": 35,
            "category": "earthquake_warning",
            "question_type": "WhatAct"
        }
    ]
    
    # QAデータセットを保存
    qa_data = {
        "version": "1.0",
        "dataset_name": "DisasterQA_Sample",
        "description": "災害対応に関する質問応答データセット（サンプル版）",
        "data": sample_qa_pairs
    }
    
    # データセット保存ディレクトリの作成
    dataset_dir = processed_dir / "qa_dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    # JSONファイルとして保存
    qa_file = dataset_dir / "disaster_qa_sample.json"
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ QAデータセット作成完了: {qa_file}")
    print(f"📊 作成されたQAペア数: {len(sample_qa_pairs)}")
    
    return qa_file

if __name__ == "__main__":
    create_sample_qa_dataset()