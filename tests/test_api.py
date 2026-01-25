"""
Disaster QA API テストケース

FastAPI エンドポイントの動作確認用テスト
"""

import pytest
import requests
import json
from typing import Dict, Any
import asyncio
import time

# テスト設定
BASE_URL = "http://localhost:8000"
TIMEOUT = 30  # 30秒タイムアウト

class DisasterQAAPITester:
    """API テスタークラス"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def wait_for_api(self, max_wait: int = 60) -> bool:
        """APIサーバーの起動を待つ"""
        print(f"Waiting for API server at {self.base_url}...")
        
        for i in range(max_wait):
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get("status") == "healthy":
                        print("✅ API server is ready!")
                        return True
                    else:
                        print(f"API server starting... ({health_data.get('status')})")
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        print("❌ API server did not start within timeout")
        return False
    
    def test_root_endpoint(self) -> Dict[str, Any]:
        """ルートエンドポイントテスト"""
        print("\\n🧪 Testing root endpoint...")
        
        response = self.session.get(f"{self.base_url}/")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        
        print(f"✅ Root endpoint: {data}")
        return data
    
    def test_health_check(self) -> Dict[str, Any]:
        """ヘルスチェックテスト"""
        print("\\n🧪 Testing health check...")
        
        response = self.session.get(f"{self.base_url}/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "pipeline_info" in data
        
        print(f"✅ Health status: {data['status']}")
        print(f"📊 Pipeline info: {data['pipeline_info']}")
        
        return data
    
    def test_single_qa(self, question: str = "地震が発生したら何をすべきですか？") -> Dict[str, Any]:
        """単一QAエンドポイントテスト"""
        print(f"\\n🧪 Testing single QA: '{question}'")
        
        payload = {
            "question": question,
            "language": "ja",
            "top_k": 3
        }
        
        response = self.session.post(
            f"{self.base_url}/qa",
            json=payload,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        
        # レスポンス構造確認
        required_fields = ["answer", "confidence", "question_type", 
                          "disaster_category", "sources", "processing_time_ms"]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        print(f"✅ Answer: {data['answer'][:100]}...")
        print(f"🎯 Question type: {data['question_type']}")
        print(f"🏷️ Category: {data['disaster_category']}")
        print(f"📊 Confidence: {data['confidence']:.3f}")
        print(f"⏱️ Processing time: {data['processing_time_ms']:.1f}ms")
        print(f"📚 Sources: {len(data['sources'])} documents")
        
        return data
    
    def test_batch_qa(self, questions: list = None) -> Dict[str, Any]:
        """バッチQAエンドポイントテスト"""
        if questions is None:
            questions = [
                "津波警報が出たら何をすべき？",
                "台風接近時の準備は？",
                "避難所とは何ですか？"
            ]
        
        print(f"\\n🧪 Testing batch QA with {len(questions)} questions...")
        
        payload = {
            "questions": questions,
            "language": "ja",
            "top_k": 3
        }
        
        response = self.session.post(
            f"{self.base_url}/qa/batch",
            json=payload,
            timeout=TIMEOUT * 2  # バッチは時間がかかる
        )
        
        assert response.status_code == 200
        
        data = response.json()
        
        assert "results" in data
        assert "total_questions" in data
        assert len(data["results"]) == len(questions)
        
        print(f"✅ Processed {data['total_questions']} questions")
        print(f"⏱️ Total time: {data['total_processing_time_ms']:.1f}ms")
        
        # 各結果の確認
        for i, result in enumerate(data["results"]):
            print(f"  Q{i+1}: {questions[i][:30]}...")
            print(f"  A{i+1}: {result['answer'][:50]}...")
            print(f"  Type: {result['question_type']}, Confidence: {result['confidence']:.3f}")
        
        return data
    
    def test_glossary_normalize(self, query: str = "津波警報") -> Dict[str, Any]:
        """用語正規化エンドポイントテスト"""
        print(f"\\n🧪 Testing glossary normalization: '{query}'")
        
        response = self.session.post(
            f"{self.base_url}/glossary/normalize",
            params={"query": query},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        
        print(f"✅ Normalized query: {data.get('normalized_query', 'N/A')}")
        print(f"🔍 Found terms: {len(data.get('found_terms', []))}")
        print(f"🏷️ Categories: {data.get('categories', [])}")
        
        return data
    
    def test_categories_endpoint(self) -> list:
        """カテゴリ一覧エンドポイントテスト"""
        print("\\n🧪 Testing categories endpoint...")
        
        response = self.session.get(f"{self.base_url}/categories")
        
        assert response.status_code == 200
        
        categories = response.json()
        assert isinstance(categories, list)
        assert len(categories) > 0
        
        print(f"✅ Available categories: {categories}")
        return categories
    
    def test_question_types_endpoint(self) -> list:
        """質問タイプ一覧エンドポイントテスト"""
        print("\\n🧪 Testing question types endpoint...")
        
        response = self.session.get(f"{self.base_url}/question-types")
        
        assert response.status_code == 200
        
        question_types = response.json()
        assert isinstance(question_types, list)
        assert len(question_types) > 0
        
        print(f"✅ Available question types: {question_types}")
        return question_types
    
    def test_error_handling(self):
        """エラーハンドリングテスト"""
        print("\\n🧪 Testing error handling...")
        
        # 空の質問
        response = self.session.post(
            f"{self.base_url}/qa",
            json={"question": "", "language": "ja"}
        )
        assert response.status_code == 422  # Validation error
        
        # 無効なパラメータ
        response = self.session.post(
            f"{self.base_url}/qa",
            json={"question": "test", "top_k": -1}
        )
        assert response.status_code == 422
        
        print("✅ Error handling works correctly")
    
    def run_all_tests(self):
        """全テストを実行"""
        print("🚀 Starting Disaster QA API Tests")
        print("=" * 50)
        
        # APIサーバー待機
        if not self.wait_for_api():
            print("❌ Cannot connect to API server")
            return False
        
        try:
            # 各テストを実行
            self.test_root_endpoint()
            self.test_health_check()
            self.test_categories_endpoint()
            self.test_question_types_endpoint()
            self.test_glossary_normalize()
            self.test_single_qa()
            self.test_batch_qa()
            self.test_error_handling()
            
            print("\\n" + "=" * 50)
            print("🎉 All tests passed successfully!")
            return True
            
        except Exception as e:
            print(f"\\n❌ Test failed: {e}")
            return False


def main():
    """テスト実行メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Disaster QA API Tester')
    parser.add_argument('--url', type=str, default=BASE_URL,
                       help='API server URL')
    parser.add_argument('--test', type=str, choices=[
        'all', 'health', 'qa', 'batch', 'glossary'
    ], default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    tester = DisasterQAAPITester(args.url)
    
    if args.test == 'all':
        success = tester.run_all_tests()
    elif args.test == 'health':
        tester.test_health_check()
        success = True
    elif args.test == 'qa':
        tester.test_single_qa()
        success = True
    elif args.test == 'batch':
        tester.test_batch_qa()
        success = True
    elif args.test == 'glossary':
        tester.test_glossary_normalize()
        success = True
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()