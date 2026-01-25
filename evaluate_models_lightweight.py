"""
軽量版モデル評価スクリプト
詳細分析を省略し、基本的なメトリクス評価のみ実行
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import time

from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer
)
from peft import PeftModel

from src.utils import load_config, get_project_root, ensure_dir
from src.modules.enhanced_evaluator import create_enhanced_compute_metrics
from src.modules.fine_tuning import DisaQuADSample

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightModelEvaluator:
    """軽量版モデル評価クラス"""
    
    def __init__(self, base_model_name: str = "cl-tohoku/bert-base-japanese-v3"):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.models = {}
        self.results = {}
        
    def load_model(self, model_path: str, model_name: str):
        """LoRAファインチューニング済みモデルをロード"""
        import gc
        
        try:
            logger.info(f"Loading {model_name} from {model_path}")
            
            # 既存のモデルがある場合はクリーンアップ
            if model_name in self.models:
                logger.info(f"Cleaning up existing model: {model_name}")
                del self.models[model_name]
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            
            # ベースモデルのロード
            base_model = AutoModelForQuestionAnswering.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
            
            # デバイスに移動
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            base_model = base_model.to(device)
            
            # LoRAモデルのロード
            model = PeftModel.from_pretrained(base_model, model_path)
            model.eval()
            
            self.models[model_name] = model
            logger.info(f"✅ Successfully loaded {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {str(e)}")
            return False
    
    def load_test_dataset(self, test_data_path: str) -> List[DisaQuADSample]:
        """独立テストデータセットをロード"""
        logger.info(f"Loading test dataset from {test_data_path}")
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            answer_start = item.get('start_char', 0)
            answer_text = item['answer']
            
            sample = DisaQuADSample(
                question=item['question'],
                context=item['context'],
                answer=answer_text,
                start_position=answer_start,
                end_position=answer_start + len(answer_text),
                disaster_type=item.get('disaster_type', 'unknown'),
                question_type=item.get('question_type', 'unknown'),
                document_source=item.get('document_source', 'unknown')
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} test samples")
        return samples
    
    def evaluate_model_basic(self, model_name: str, test_samples: List[DisaQuADSample]) -> Dict:
        """基本的なモデル評価（軽量版）"""
        logger.info(f"Evaluating {model_name} (basic evaluation only)")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # データセットをHugging Face形式に変換
        from src.modules.fine_tuning import LoRATrainer, DisaQuADDataset
        temp_trainer = LoRATrainer()
        temp_trainer.tokenizer = self.tokenizer
        temp_trainer.max_seq_length = 384
        
        temp_dataset = DisaQuADDataset.__new__(DisaQuADDataset)
        temp_dataset.samples = test_samples
        
        # データセット変換
        hf_dataset = temp_trainer.prepare_dataset(temp_dataset)
        
        # 評価メトリクス関数の作成
        compute_metrics = create_enhanced_compute_metrics(self.tokenizer)
        
        # Trainer設定（最小構成）
        training_args = TrainingArguments(
            output_dir=f"./temp_eval_{model_name}",
            per_device_eval_batch_size=2,  # 最小バッチサイズ
            dataloader_drop_last=False,
            eval_accumulation_steps=4,
            disable_tqdm=False,
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
            logging_strategy="no",
            save_strategy="no",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
        
        # 評価実行
        logger.info(f"Starting basic evaluation for {model_name}...")
        start_time = time.time()
        eval_results = trainer.evaluate(eval_dataset=hf_dataset)
        eval_time = time.time() - start_time
        
        logger.info(f"Basic evaluation completed for {model_name} in {eval_time:.1f}s")
        
        # 結果を保存
        self.results[model_name] = eval_results
        self.results[model_name]['evaluation_time'] = eval_time
        
        # クリーンアップ
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()
        
        return eval_results
    
    def save_results(self, save_path: str = "evaluation_results/lightweight_results.json"):
        """結果をJSONファイルに保存"""
        save_file = Path(save_path)
        save_file.parent.mkdir(exist_ok=True)
        
        # NumPyタイプをPythonタイプに変換
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = self._convert_to_serializable(results)
        
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {save_file}")
    
    def _convert_to_serializable(self, obj):
        """NumPy型をJSONシリアライズ可能な型に変換"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def main():
    """メイン実行関数（軽量版）"""
    logger.info("🚀 Starting lightweight model evaluation")
    
    start_time = time.time()
    
    # プロジェクトのパス設定
    project_root = get_project_root()
    models_dir = project_root / "models" / "lora_finetuned_bert-base-japanese-v3"
    test_data_path = project_root / "data" / "processed" / "test_dataset" / "test_samples_300.json"
    
    # 評価器を初期化
    evaluator = LightweightModelEvaluator()
    
    # テストデータセットをロード
    test_samples = evaluator.load_test_dataset(test_data_path)
    logger.info(f"Loaded test dataset with {len(test_samples)} samples")
    
    # 各モデルをロードして評価
    model_configs = [
        ("100-samples", models_dir / "checkpoint-400"),
        ("500-samples", models_dir / "checkpoint-1100"),
        ("1000-samples", models_dir / "checkpoint-1200")
    ]
    
    evaluation_results = {}
    
    for idx, (model_name, model_path) in enumerate(model_configs):
        model_start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Starting evaluation {idx+1}/{len(model_configs)}: {model_name}")
        
        if model_path.exists():
            success = evaluator.load_model(str(model_path), model_name)
            if success:
                try:
                    results = evaluator.evaluate_model_basic(model_name, test_samples)
                    evaluation_results[model_name] = results
                    model_time = time.time() - model_start_time
                    logger.info(f"✅ {model_name} evaluation completed in {model_time:.1f}s")
                except Exception as e:
                    logger.error(f"❌ Failed to evaluate {model_name}: {e}")
                    continue
            else:
                logger.error(f"❌ Failed to load {model_name}")
        else:
            logger.warning(f"⚠️ Model path not found: {model_path}")
        
        # 進行状況表示
        completed = len(evaluation_results)
        total = len(model_configs)
        elapsed_time = time.time() - start_time
        logger.info(f"📊 Progress: {completed}/{total} models completed")
        logger.info(f"⏱️ Elapsed time: {elapsed_time:.1f}s")
    
    # 総実行時間
    total_time = time.time() - start_time
    
    # 結果の表示
    print("\n" + "="*80)
    print("📊 LIGHTWEIGHT EVALUATION RESULTS")
    print("="*80)
    print(f"⏱️ Total execution time: {total_time:.1f} seconds")
    print(f"✅ Successfully evaluated: {len(evaluation_results)}/{len(model_configs)} models")
    
    if evaluation_results:
        for model_name, results in evaluation_results.items():
            print(f"\n🔹 {model_name.upper()}:")
            print(f"   Start Position Accuracy: {results.get('eval_start_position_accuracy', 0):.3f}")
            print(f"   End Position Accuracy:   {results.get('eval_end_position_accuracy', 0):.3f}")
            print(f"   Span F1:                 {results.get('eval_span_f1', 0):.3f}")
            print(f"   Overall F1:              {results.get('eval_overall_f1', 0):.3f}")
            print(f"   Evaluation time:         {results.get('evaluation_time', 0):.1f}s")
        
        # 結果保存
        try:
            evaluator.save_results("evaluation_results/lightweight_results.json")
            print("\n📄 Results saved to: evaluation_results/lightweight_results.json")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        logger.info("Lightweight evaluation completed successfully! 🎉")
    else:
        logger.error("❌ No models were successfully evaluated")


if __name__ == "__main__":
    main()