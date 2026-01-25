"""
共通ユーティリティ関数
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """設定ファイルを読み込む"""
    if config_path is None:
        # src/utils.py から disaster-question-answer/config/config.yaml へのパス
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_logging(config: Dict[str, Any] = None):
    """ログ設定を初期化"""
    if config is None:
        config = load_config()
    
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('file', 'logs/disaster_qa.log')
    
    # ログディレクトリを作成
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得"""
    # src/utils.py から disaster-question-answer/ へのパス
    return Path(__file__).parent.parent

def ensure_dir(path: str):
    """ディレクトリが存在しない場合は作成"""
    os.makedirs(path, exist_ok=True)