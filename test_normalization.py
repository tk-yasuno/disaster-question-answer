"""
用語正規化テストスクリプト
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.modules.glossary import GlossaryManager
import logging

logging.basicConfig(level=logging.INFO)

def test_normalization():
    print("=== 用語正規化詳細テスト ===")
    
    # GlossaryManagerの初期化
    glossary = GlossaryManager()
    
    # テスト用語
    test_queries = [
        "津波警報が出たら何をすべき？",
        "地震が起きたらどうすればいいですか？", 
        "台風に備えるにはどうすればよいですか？",
        "避難所はどこですか？",
        "防災グッズは何が必要ですか？",
        "土砂災害が発生した場合の対応は？",
        "緊急地震速報が鳴ったらどうする？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 テスト {i}: {query}")
        
        # 正規化実行
        normalized = glossary.normalize_query(query)
        print(f"📝 正規化結果: {normalized}")
        
        # 類似用語検索
        similar_terms = glossary._find_similar_terms(query)
        print(f"🔗 類似用語:")
        for j, term_info in enumerate(similar_terms[:5], 1):  # Top 5のみ表示
            print(f"  {j}. {term_info['japanese']} -> {term_info['english']} (類似度: {term_info['similarity']:.4f})")
        
        print("-" * 60)

if __name__ == "__main__":
    test_normalization()