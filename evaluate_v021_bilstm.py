#!/usr/bin/env python3
"""
v0.2.1 Model Evaluation Script
BERT + Bi-LSTM End Position Enhanced Model Evaluation

Usage:
python evaluate_v021_bilstm.py --dataset-size 500 --model-path models/v021_bert_bilstm
"""

import argparse
import json
import logging
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import sys

from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import the model class
from train_v021_bilstm import BertBiLSTMQA, V021Config, DisasterQADataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class V021Evaluator:
    """v0.2.1 Model Evaluator"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"✅ v0.2.1 Model loaded from {model_path}")
        logger.info(f"   Device: {self.device}")
    
    def _load_model(self):
        """Load the trained v0.2.1 model"""
        
        # Load config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        self.config = V021Config(**config_dict)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # Initialize model
        self.model = BertBiLSTMQA(self.config).to(self.device)
        
        # Apply LoRA to match training setup
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["query", "key", "value", "dense"]
        )
        self.model.bert = get_peft_model(self.model.bert, lora_config)
        
        # Load model state
        model_state_path = self.model_path / "model_state.pt"
        if not model_state_path.exists():
            raise FileNotFoundError(f"Model state file not found: {model_state_path}")
        
        # Load with strict=False to handle LoRA structure differences
        state_dict = torch.load(model_state_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        logger.info("✅ Model and tokenizer loaded successfully")
    
    def evaluate(self, test_dataset_path: str) -> Dict:
        """Evaluate the model on test dataset"""
        
        # Load test dataset
        test_dataset = DisasterQADataset(test_dataset_path, self.tokenizer, self.config.max_seq_length)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"📊 Evaluating on {len(test_dataset)} samples...")
        
        # Evaluation metrics
        start_predictions = []
        end_predictions = []
        start_labels = []
        end_labels = []
        
        total_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Extract only required inputs
                model_inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "start_positions": batch["start_positions"],
                    "end_positions": batch["end_positions"]
                }
                
                outputs = self.model(**model_inputs)
                
                # Loss
                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                    num_batches += 1
                
                # Predictions
                start_pred = torch.argmax(outputs["start_logits"], dim=1)
                end_pred = torch.argmax(outputs["end_logits"], dim=1)
                
                start_predictions.extend(start_pred.cpu().numpy())
                end_predictions.extend(end_pred.cpu().numpy())
                start_labels.extend(batch["start_positions"].cpu().numpy())
                end_labels.extend(batch["end_positions"].cpu().numpy())
        
        eval_time = time.time() - start_time
        
        # Calculate metrics
        start_accuracy = accuracy_score(start_labels, start_predictions)
        end_accuracy = accuracy_score(end_labels, end_predictions)
        
        # Exact match accuracy (both start and end correct)
        exact_matches = [(s_pred == s_true and e_pred == e_true) for s_pred, s_true, e_pred, e_true in 
                        zip(start_predictions, start_labels, end_predictions, end_labels)]
        exact_match_accuracy = np.mean(exact_matches)
        
        # F1 scores
        start_f1 = f1_score(start_labels, start_predictions, average='macro', zero_division=0)
        end_f1 = f1_score(end_labels, end_predictions, average='macro', zero_division=0)
        
        # Span F1 (considering both start and end)
        span_f1_scores = []
        for i in range(len(start_labels)):
            pred_span = set(range(start_predictions[i], end_predictions[i] + 1))
            true_span = set(range(start_labels[i], end_labels[i] + 1))
            
            if len(pred_span) == 0 and len(true_span) == 0:
                span_f1_scores.append(1.0)
            elif len(pred_span) == 0 or len(true_span) == 0:
                span_f1_scores.append(0.0)
            else:
                intersection = len(pred_span.intersection(true_span))
                precision = intersection / len(pred_span)
                recall = intersection / len(true_span)
                
                if precision + recall == 0:
                    span_f1_scores.append(0.0)
                else:
                    span_f1_scores.append(2 * precision * recall / (precision + recall))
        
        span_f1 = np.mean(span_f1_scores)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        metrics = {
            "test_samples": len(test_dataset),
            "evaluation_time": eval_time,
            "average_loss": avg_loss,
            "start_accuracy": start_accuracy,
            "end_accuracy": end_accuracy,
            "exact_match_accuracy": exact_match_accuracy,
            "start_f1": start_f1,
            "end_f1": end_f1,
            "span_f1": span_f1,
            "overall_f1": (start_f1 + end_f1) / 2
        }
        
        return metrics
    
    def inference(self, question: str, context: str) -> Dict:
        """Perform inference on a single question-context pair"""
        
        # Tokenize
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=self.config.max_seq_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            # Extract only the required inputs for the model
            model_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
            
            outputs = self.model(**model_inputs)
            
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            
            start_idx = torch.argmax(start_logits, dim=1).item()
            end_idx = torch.argmax(end_logits, dim=1).item()
            
            # Ensure end >= start
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Extract answer
            input_ids = inputs["input_ids"][0]
            if start_idx < len(input_ids) and end_idx < len(input_ids):
                answer_tokens = input_ids[start_idx:end_idx + 1]
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            else:
                answer = "[No answer found]"
            
            # Calculate confidence scores
            start_confidence = torch.softmax(start_logits, dim=1)[0, start_idx].item()
            end_confidence = torch.softmax(end_logits, dim=1)[0, end_idx].item()
            overall_confidence = (start_confidence + end_confidence) / 2
        
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "start_position": start_idx,
            "end_position": end_idx,
            "start_confidence": start_confidence,
            "end_confidence": end_confidence,
            "overall_confidence": overall_confidence
        }
    
    def compare_with_baseline(self, test_dataset_path: str, baseline_results: Dict = None) -> Dict:
        """Compare v0.2.1 results with baseline (v0.2)"""
        
        current_metrics = self.evaluate(test_dataset_path)
        
        if baseline_results:
            comparison = {
                "v021_results": current_metrics,
                "baseline_results": baseline_results,
                "improvements": {}
            }
            
            for metric in ["start_accuracy", "end_accuracy", "exact_match_accuracy", "span_f1"]:
                if metric in baseline_results:
                    baseline_val = baseline_results[metric]
                    current_val = current_metrics[metric]
                    improvement = ((current_val - baseline_val) / baseline_val) * 100 if baseline_val > 0 else 0
                    comparison["improvements"][metric] = {
                        "baseline": baseline_val,
                        "v021": current_val,
                        "improvement_percent": improvement
                    }
            
            return comparison
        else:
            return {"v021_results": current_metrics}

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="v0.2.1 BERT + Bi-LSTM Evaluation")
    parser.add_argument("--model-path", type=str, default="models/v021_bert_bilstm",
                        help="Path to trained v0.2.1 model")
    parser.add_argument("--dataset-size", type=int, default=500, choices=[100, 500, 1000],
                        help="Test dataset size")
    parser.add_argument("--test-dataset", type=str, default=None,
                        help="Custom test dataset path")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo inference examples")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare with baseline results")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        logger.info("Please train the model first: python train_v021_bilstm.py")
        return
    
    # Initialize evaluator
    evaluator = V021Evaluator(args.model_path)
    
    # Demo inference
    if args.demo:
        logger.info("🎯 Running demo inference...")
        
        demo_cases = [
            {
                "question": "地震が発生したときにまず何をすべきですか？",
                "context": "地震が発生した場合、まず自分の安全を確保することが重要です。机の下に隠れるか、頭を保護してください。その後、火の元を確認し、ガスの元栓を閉めてください。"
            },
            {
                "question": "津波警報が出たらどうすればいいですか？",
                "context": "津波警報が発表されたら、直ちに高台や頑丈な建物の3階以上に避難してください。海や川から離れ、津波の到達予想時間を確認してください。"
            },
            {
                "question": "緊急時の連絡先は何番ですか？",
                "context": "緊急時には、まず119番（消防救急）または110番（警察）に連絡してください。災害時は、災害用伝言ダイヤル171番も利用できます。"
            }
        ]
        
        for i, case in enumerate(demo_cases, 1):
            logger.info(f"\n📝 Demo {i}:")
            result = evaluator.inference(case["question"], case["context"])
            logger.info(f"Question: {result['question']}")
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Confidence: {result['overall_confidence']:.3f}")
            logger.info(f"Position: [{result['start_position']}, {result['end_position']}]")
    
    # Full evaluation
    test_dataset = args.test_dataset or f"data/processed/qa_dataset_v2/qa_samples_{args.dataset_size}.json"
    
    if Path(test_dataset).exists():
        logger.info(f"📊 Evaluating model on {test_dataset}...")
        
        if args.compare_baseline:
            # Load baseline results if available
            baseline_path = Path("results/v02_baseline_results.json")
            baseline_results = None
            if baseline_path.exists():
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline_results = json.load(f)
            
            results = evaluator.compare_with_baseline(test_dataset, baseline_results)
        else:
            results = {"v021_results": evaluator.evaluate(test_dataset)}
        
        # Display results
        logger.info("\n🏆 v0.2.1 Evaluation Results:")
        logger.info("=" * 50)
        
        v021_results = results.get("v021_results", results)
        logger.info(f"Test Samples: {v021_results['test_samples']}")
        logger.info(f"Evaluation Time: {v021_results['evaluation_time']:.1f}s")
        logger.info(f"Average Loss: {v021_results['average_loss']:.4f}")
        logger.info(f"Start Position Accuracy: {v021_results['start_accuracy']:.3f}")
        logger.info(f"End Position Accuracy: {v021_results['end_accuracy']:.3f}")
        logger.info(f"Exact Match Accuracy: {v021_results['exact_match_accuracy']:.3f}")
        logger.info(f"Span F1 Score: {v021_results['span_f1']:.3f}")
        logger.info(f"Overall F1 Score: {v021_results['overall_f1']:.3f}")
        
        # Show improvements if comparing with baseline
        if "improvements" in results:
            logger.info("\n📈 Improvements over Baseline:")
            logger.info("-" * 30)
            for metric, data in results["improvements"].items():
                improvement = data["improvement_percent"]
                logger.info(f"{metric.replace('_', ' ').title()}: {data['baseline']:.3f} → {data['v021']:.3f} ({improvement:+.1f}%)")
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        output_file = results_dir / "v021_evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Results saved to {output_file}")
        
    else:
        logger.error(f"Test dataset not found: {test_dataset}")
        logger.info("Available datasets:")
        dataset_dir = Path("data/processed/qa_dataset_v2")
        if dataset_dir.exists():
            for file in dataset_dir.glob("*.json"):
                logger.info(f"  - {file}")

if __name__ == "__main__":
    main()