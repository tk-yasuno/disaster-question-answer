"""
用語理解・正規化モジュール

災害用語、地名、行政用語の理解と正規化を行う。
sentence-transformersを使用してセマンティック検索を実現。
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Optional
import faiss
import logging
from pathlib import Path
import pickle

from ..utils import load_config, get_project_root

logger = logging.getLogger(__name__)

class GlossaryManager:
    """
    用語辞書管理クラス
    
    機能:
    - 災害用語辞書の読み込み・管理
    - クエリの用語正規化
    - セマンティック類似度検索
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.model_name = self.config['embedding_model']
        self.min_similarity = self.config['glossary']['min_similarity']
        self.max_candidates = self.config['glossary']['max_candidates']
        
        # Embedding モデル初期化
        logger.info(f"Loading embedding model: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)
        
        # 用語辞書とインデックス
        self.terms_df = None
        self.term_embeddings = None
        self.faiss_index = None
        
        # 初期化
        self._load_glossary()
        self._build_embeddings()
    
    def _load_glossary(self):
        """用語辞書CSVファイルを読み込む"""
        try:
            glossary_path = get_project_root() / self.config['data_paths']['glossary'] / 'disaster_terms.csv'
            
            # CSVファイル読み込み（ヘッダー付き）
            self.terms_df = pd.read_csv(glossary_path, 
                                      names=['japanese', 'english', 'category'],
                                      comment='#',  # コメント行をスキップ
                                      skipinitialspace=True)
            
            # 空行・無効行を除去
            self.terms_df = self.terms_df.dropna()
            self.terms_df = self.terms_df[self.terms_df['japanese'].str.strip() != '']
            
            logger.info(f"Loaded {len(self.terms_df)} terms from glossary")
            
        except Exception as e:
            logger.error(f"Failed to load glossary: {e}")
            # フォールバック: 空のデータフレーム
            self.terms_df = pd.DataFrame(columns=['japanese', 'english', 'category'])
    
    def _build_embeddings(self):
        """用語のembeddingを構築してFAISSインデックスを作成"""
        if len(self.terms_df) == 0:
            logger.warning("No terms to build embeddings")
            return
        
        try:
            # 日本語用語をembedding
            terms_list = self.terms_df['japanese'].tolist()
            logger.info("Building term embeddings...")
            
            self.term_embeddings = self.encoder.encode(terms_list, 
                                                     show_progress_bar=True,
                                                     convert_to_numpy=True)
            
            # FAISSインデックス構築
            dimension = self.term_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # L2正規化してcosine similarityに
            normalized_embeddings = self.term_embeddings / np.linalg.norm(
                self.term_embeddings, axis=1, keepdims=True)
            
            self.faiss_index.add(normalized_embeddings.astype(np.float32))
            
            logger.info(f"Built FAISS index with {self.faiss_index.ntotal} terms")
            
        except Exception as e:
            logger.error(f"Failed to build embeddings: {e}")
            raise
    
    def normalize_query(self, query: str, language: str = 'auto') -> Dict:
        """
        クエリを正規化し、災害用語を標準形に変換
        
        Args:
            query: 入力クエリ
            language: 言語 ('ja', 'en', 'auto')
        
        Returns:
            正規化結果辞書
        """
        if not query.strip():
            return {
                'normalized_query': query,
                'found_terms': [],
                'categories': [],
                'language': language
            }
        
        # 用語検索
        found_terms = self._find_similar_terms(query)
        
        # 正規化クエリを構築
        normalized_query = query
        categories = set()
        
        for term_info in found_terms:
            japanese_term = term_info['japanese']
            english_term = term_info['english']
            category = term_info['category']
            
            categories.add(category)
            
            # 用語置換 (簡単な実装)
            if japanese_term in query:
                normalized_query = normalized_query.replace(
                    japanese_term, 
                    f"{japanese_term} ({english_term})"
                )
        
        return {
            'normalized_query': normalized_query,
            'found_terms': found_terms,
            'categories': list(categories),
            'language': self._detect_language(query) if language == 'auto' else language
        }
    
    def _find_similar_terms(self, query: str) -> List[Dict]:
        """クエリに類似する用語を検索"""
        if self.faiss_index is None or len(self.terms_df) == 0:
            return []
        
        try:
            # クエリをembedding
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            # FAISS検索
            similarities, indices = self.faiss_index.search(
                query_embedding.astype(np.float32),
                min(self.max_candidates, len(self.terms_df))
            )
            
            # 結果をフィルタリング
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity >= self.min_similarity:
                    term_info = {
                        'japanese': self.terms_df.iloc[idx]['japanese'],
                        'english': self.terms_df.iloc[idx]['english'],
                        'category': self.terms_df.iloc[idx]['category'],
                        'similarity': float(similarity)
                    }
                    results.append(term_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Term search failed: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """簡単な言語検出 (ひらがな・カタカナ・漢字の存在で判定)"""
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309F' or  # ひらがな
                            '\u30A0' <= char <= '\u30FF' or                           # カタカナ
                            '\u4E00' <= char <= '\u9FAF')                            # 漢字
        
        return 'ja' if japanese_chars > len(text) * 0.3 else 'en'
    
    def categorize_query(self, query: str) -> str:
        """クエリを災害カテゴリに分類"""
        normalization_result = self.normalize_query(query)
        categories = normalization_result['categories']
        
        if not categories:
            return 'general'
        
        # カテゴリ優先順位
        priority_map = {
            'natural_disaster': ['earthquake', 'tsunami', 'typhoon', 'flood', 'volcanic_eruption'],
            'official_alert': 'alert',
            'evacuation_action': 'evacuation', 
            'recovery_phase': 'recovery'
        }
        
        # 最初に見つかった重要カテゴリを返す
        for category in categories:
            if category in ['earthquake', 'tsunami', 'typhoon', 'flood', 'volcanic_eruption']:
                return category
        
        return categories[0] if categories else 'general'
    
    def get_related_terms(self, term: str, max_results: int = 5) -> List[Dict]:
        """関連用語を取得"""
        return self._find_similar_terms(term)[:max_results]
    
    def add_custom_term(self, japanese: str, english: str, category: str):
        """カスタム用語を追加 (将来の拡張用)"""
        new_term = pd.DataFrame({
            'japanese': [japanese],
            'english': [english], 
            'category': [category]
        })
        
        self.terms_df = pd.concat([self.terms_df, new_term], ignore_index=True)
        
        # embeddings再構築
        self._build_embeddings()
        
        logger.info(f"Added custom term: {japanese} -> {english}")
    
    def save_index(self, path: str):
        """FAISSインデックスを保存"""
        try:
            index_path = Path(path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(self.faiss_index, str(index_path / 'glossary.faiss'))
            
            # メタデータ保存
            metadata = {
                'terms_df': self.terms_df,
                'model_name': self.model_name,
                'config': self.config
            }
            
            with open(index_path / 'glossary_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"Saved glossary index to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self, path: str):
        """保存されたFAISSインデックスを読み込み"""
        try:
            index_path = Path(path)
            
            self.faiss_index = faiss.read_index(str(index_path / 'glossary.faiss'))
            
            # メタデータ読み込み
            with open(index_path / 'glossary_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            self.terms_df = metadata['terms_df']
            
            logger.info(f"Loaded glossary index from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise


# CLI用のメイン関数
def main():
    """コマンドライン実行用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Glossary Manager CLI')
    parser.add_argument('--build-glossary', action='store_true',
                      help='Build glossary embeddings')
    parser.add_argument('--test-query', type=str,
                      help='Test query normalization')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    glossary = GlossaryManager()
    
    if args.build_glossary:
        print("✅ Glossary embeddings built successfully")
    
    if args.test_query:
        result = glossary.normalize_query(args.test_query)
        print(f"Query: {args.test_query}")
        print(f"Normalized: {result['normalized_query']}")
        print(f"Found terms: {result['found_terms']}")
        print(f"Categories: {result['categories']}")


if __name__ == '__main__':
    main()