"""
災害PDFデータ処理・前処理モジュール

46個のPDFファイルからテキスト抽出、チャンク分割、ベクトル化を行う
災害文書特有の構造（図表、写真説明等）に対応
"""

import os
import re
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
from tqdm import tqdm
import hashlib

from ..utils import load_config, get_project_root, ensure_dir

logger = logging.getLogger(__name__)

class DisasterDataProcessor:
    """
    災害PDFデータ処理クラス
    
    機能:
    - PDFテキスト抽出（図表・写真キャプション含む）
    - 災害文書特有の構造解析
    - テキストチャンク分割
    - ベクトル化・インデックス構築
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.embedding_model = self.config['embedding_model']
        self.chunk_size = self.config['retrieval']['chunk_size']
        self.chunk_overlap = self.config['retrieval']['chunk_overlap']
        
        # Embedding モデル
        logger.info(f"Loading embedding model: {self.embedding_model}")
        self.encoder = SentenceTransformer(self.embedding_model)
        
        # データパス
        self.pdf_dir = get_project_root() / self.config['data_paths']['disaster_docs']
        self.processed_dir = get_project_root() / self.config['data_paths']['processed']
        ensure_dir(self.processed_dir)
        
        # 処理済みデータ
        self.documents = []
        self.document_embeddings = None
        self.faiss_index = None
    
    def extract_text_from_pdfs(self, force_reprocess: bool = False) -> List[Dict]:
        """
        全PDFファイルからテキストを抽出
        
        Args:
            force_reprocess: 強制再処理フラグ
        
        Returns:
            抽出文書のリスト
        """
        cache_file = self.processed_dir / 'extracted_documents.pkl'
        
        # キャッシュ確認
        if not force_reprocess and cache_file.exists():
            logger.info("Loading cached extracted documents")
            with open(cache_file, 'rb') as f:
                self.documents = pickle.load(f)
            return self.documents
        
        logger.info(f"Extracting text from PDF files in {self.pdf_dir}")
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        extracted_docs = []
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                doc_info = self._extract_single_pdf(pdf_path)
                if doc_info:
                    extracted_docs.append(doc_info)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue
        
        self.documents = extracted_docs
        
        # キャッシュ保存
        with open(cache_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Extracted text from {len(extracted_docs)} PDF files")
        return extracted_docs
    
    def _extract_single_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """単一PDFファイルからテキスト抽出"""
        try:
            doc = fitz.open(pdf_path)
            pages_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # テキスト抽出
                text = page.get_text()
                
                # 画像・図表情報も取得
                images = page.get_images()
                image_info = [f"[図表{i+1}]" for i in range(len(images))]
                
                # ページ情報
                page_info = {
                    'page_number': page_num + 1,
                    'text': text,
                    'images_count': len(images),
                    'images_info': image_info
                }
                
                pages_content.append(page_info)
            
            doc.close()
            
            # 文書メタデータ
            document_info = {
                'filename': pdf_path.name,
                'filepath': str(pdf_path),
                'title': self._extract_title_from_filename(pdf_path.name),
                'disaster_type': self._classify_disaster_type(pdf_path.name),
                'pages_count': len(pages_content),
                'pages': pages_content,
                'full_text': self._combine_pages_text(pages_content)
            }
            
            return document_info
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return None
    
    def _extract_title_from_filename(self, filename: str) -> str:
        """ファイル名から文書タイトルを推定"""
        # ファイル名パターン解析
        patterns = {
            r'disaster-H(\d+)_([^.]+)': r'平成\1年 \2災害',
            r'(\d{4})(\d{2})_([^.]+)': r'\1年\2月 \3',
            r'jirei(\d+)': r'災害事例集 第\1集',
        }
        
        for pattern, replacement in patterns.items():
            match = re.search(pattern, filename)
            if match:
                return re.sub(pattern, replacement, filename.replace('.pdf', ''))
        
        return filename.replace('.pdf', '').replace('_', ' ')
    
    def _classify_disaster_type(self, filename: str) -> str:
        """ファイル名から災害種別を分類"""
        disaster_keywords = {
            'earthquake': ['jishin', 'earthquake', 'chuetsu', 'hanshin', 'kumamoto', 'higashinihon'],
            'tsunami': ['tsunami', 'higashinihon'],
            'typhoon': ['typhoon', 'taifuu'],
            'flood': ['flood', 'ooame', '07ooame', 'kyushu'],
            'volcanic': ['volcano', 'mitake', 'hakone', 'kuchinoerabu'],
            'landslide': ['dosekiryu', 'landslide'],
            'snow': ['oyuki', 'snow']
        }
        
        filename_lower = filename.lower()
        
        for disaster_type, keywords in disaster_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                return disaster_type
        
        return 'general'
    
    def _combine_pages_text(self, pages: List[Dict]) -> str:
        """ページテキストを結合"""
        combined_text = ""
        for page in pages:
            page_text = page['text'].strip()
            if page_text:
                combined_text += f"\\n--- Page {page['page_number']} ---\\n{page_text}\\n"
        return combined_text
    
    def create_text_chunks(self, force_reprocess: bool = False) -> List[Dict]:
        """テキストをチャンクに分割"""
        
        cache_file = self.processed_dir / 'text_chunks.pkl'
        
        # キャッシュ確認
        if not force_reprocess and cache_file.exists():
            logger.info("Loading cached text chunks")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        if not self.documents:
            logger.warning("No documents loaded. Run extract_text_from_pdfs first.")
            return []
        
        logger.info("Creating text chunks...")
        
        chunks = []
        chunk_id = 0
        
        for doc in tqdm(self.documents, desc="Chunking documents"):
            doc_chunks = self._chunk_document(doc, chunk_id)
            chunks.extend(doc_chunks)
            chunk_id += len(doc_chunks)
        
        # キャッシュ保存
        with open(cache_file, 'wb') as f:
            pickle.dump(chunks, f)
        
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def _chunk_document(self, document: Dict, start_chunk_id: int) -> List[Dict]:
        """単一文書をチャンクに分割"""
        full_text = document['full_text']
        chunks = []
        
        # 簡単なチャンク分割（文単位）
        sentences = re.split(r'[。！？\\n]+', full_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        current_chunk_id = start_chunk_id
        
        for sentence in sentences:
            # チャンクサイズをチェック
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # 現在のチャンクを保存
                chunk_info = {
                    'chunk_id': current_chunk_id,
                    'text': current_chunk.strip(),
                    'document_filename': document['filename'],
                    'document_title': document['title'],
                    'disaster_type': document['disaster_type'],
                    'chunk_hash': self._generate_chunk_hash(current_chunk),
                    'word_count': len(current_chunk.split())
                }
                chunks.append(chunk_info)
                
                # 次のチャンクを開始（オーバーラップ処理）
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + sentence
                current_chunk_id += 1
            else:
                current_chunk += sentence
        
        # 残りのチャンクを処理
        if current_chunk.strip():
            chunk_info = {
                'chunk_id': current_chunk_id,
                'text': current_chunk.strip(),
                'document_filename': document['filename'],
                'document_title': document['title'],
                'disaster_type': document['disaster_type'],
                'chunk_hash': self._generate_chunk_hash(current_chunk),
                'word_count': len(current_chunk.split())
            }
            chunks.append(chunk_info)
        
        return chunks
    
    def _generate_chunk_hash(self, text: str) -> str:
        """チャンクのハッシュ値を生成"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def build_vector_index(self, chunks: List[Dict] = None, force_rebuild: bool = False) -> faiss.Index:
        """テキストチャンクのベクトルインデックスを構築"""
        
        index_file = self.processed_dir / 'faiss_index.bin'
        metadata_file = self.processed_dir / 'index_metadata.pkl'
        
        # キャッシュ確認
        if not force_rebuild and index_file.exists() and metadata_file.exists():
            logger.info("Loading cached FAISS index")
            self.faiss_index = faiss.read_index(str(index_file))
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self.document_embeddings = metadata['embeddings']
                self.chunks_metadata = metadata['chunks']
            return self.faiss_index
        
        if chunks is None:
            chunks = self.create_text_chunks()
        
        if not chunks:
            logger.warning("No chunks available for indexing")
            return None
        
        logger.info(f"Building FAISS index for {len(chunks)} chunks...")
        
        # テキストをembedding
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.encoder.encode(chunk_texts, 
                                       show_progress_bar=True,
                                       convert_to_numpy=True)
        
        # FAISSインデックス構築
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product index
        
        # L2正規化
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.faiss_index.add(normalized_embeddings.astype(np.float32))
        
        self.document_embeddings = normalized_embeddings
        self.chunks_metadata = chunks
        
        # インデックス保存
        faiss.write_index(self.faiss_index, str(index_file))
        
        # メタデータ保存
        metadata = {
            'embeddings': self.document_embeddings,
            'chunks': chunks,
            'model_name': self.embedding_model,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Built FAISS index with {self.faiss_index.ntotal} chunks")
        return self.faiss_index
    
    def search_similar_chunks(self, query: str, top_k: int = None) -> List[Dict]:
        """類似チャンクを検索"""
        if top_k is None:
            top_k = self.config['retrieval']['top_k']
        
        if self.faiss_index is None:
            logger.error("FAISS index not built. Run build_vector_index first.")
            return []
        
        # クエリをembedding
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # 検索実行
        similarities, indices = self.faiss_index.search(
            query_embedding.astype(np.float32), top_k)
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.chunks_metadata):
                chunk_info = self.chunks_metadata[idx].copy()
                chunk_info['similarity'] = float(similarity)
                results.append(chunk_info)
        
        return results
    
    def get_statistics(self) -> Dict:
        """データ処理統計を取得"""
        if not self.documents:
            return {"error": "No documents processed"}
        
        stats = {
            "total_documents": len(self.documents),
            "disaster_types": {},
            "total_pages": 0,
            "total_chunks": len(getattr(self, 'chunks_metadata', [])),
            "avg_chunk_length": 0
        }
        
        # 災害種別統計
        for doc in self.documents:
            disaster_type = doc.get('disaster_type', 'unknown')
            stats["disaster_types"][disaster_type] = stats["disaster_types"].get(disaster_type, 0) + 1
            stats["total_pages"] += doc.get('pages_count', 0)
        
        # チャンク統計
        if hasattr(self, 'chunks_metadata') and self.chunks_metadata:
            chunk_lengths = [chunk.get('word_count', 0) for chunk in self.chunks_metadata]
            stats["avg_chunk_length"] = np.mean(chunk_lengths)
        
        return stats


def main():
    """CLI実行用メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Disaster Data Processor CLI')
    parser.add_argument('--process-pdfs', action='store_true',
                      help='Extract text from PDF files')
    parser.add_argument('--create-chunks', action='store_true',
                      help='Create text chunks')
    parser.add_argument('--build-index', action='store_true',
                      help='Build FAISS vector index')
    parser.add_argument('--force', action='store_true',
                      help='Force reprocessing')
    parser.add_argument('--stats', action='store_true',
                      help='Show processing statistics')
    
    args = parser.parse_args()
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    processor = DisasterDataProcessor()
    
    if args.process_pdfs:
        processor.extract_text_from_pdfs(force_reprocess=args.force)
        print("✅ PDF text extraction completed")
    
    if args.create_chunks:
        chunks = processor.create_text_chunks(force_reprocess=args.force)
        print(f"✅ Created {len(chunks)} text chunks")
    
    if args.build_index:
        processor.build_vector_index(force_rebuild=args.force)
        print("✅ FAISS index built successfully")
    
    if args.stats:
        stats = processor.get_statistics()
        print("📊 Processing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()