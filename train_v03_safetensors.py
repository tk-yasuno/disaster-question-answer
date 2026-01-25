#!/usr/bin/env python3
"""
v0.3 Safetensors Enhanced Training Pipeline
Solves PyTorch security issues with safetensors support

Target: Improve End Position accuracy from 8.7% to 25%+
Architecture: BERT → BiLSTM → Enhanced Position Heads
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import modules
from src.modules.fine_tuning_v3_safetensors import (
    QATrainingConfig, 
    LoRATrainerV3Safetensors, 
    create_augmented_dataset_v3_safetensors
)
from src.modules.fine_tuning import load_qa_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_v03_safetensors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure all necessary directories exist"""
    dirs = [
        'logs',
        'data/processed/qa_dataset_v3',
        'models',
        'temp_eval_safetensors'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_or_create_augmented_dataset(base_dataset_path: str, dataset_size: int, use_augmented: bool = True) -> List[Dict]:
    """Load or create augmented dataset for v0.3 training"""
    
    # Define augmented dataset path
    base_name = Path(base_dataset_path).stem
    augmented_path = f"data/processed/qa_dataset_v3/{base_name}_v3_safetensors_augmented.json"
    
    if use_augmented and Path(augmented_path).exists():
        logger.info(f"📊 Loading existing augmented dataset: {augmented_path}")
        with open(augmented_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    if use_augmented:
        logger.info("🔄 Augmented dataset not found, creating...")
        logger.info(f"📊 Creating v0.3 Safetensors Augmented Dataset (base: {dataset_size} samples)")
        
        # Load base dataset
        base_samples = load_qa_dataset(base_dataset_path, dataset_size)
        
        # Create augmented dataset
        augmented_samples = create_augmented_dataset_v3_safetensors(base_samples, expansion_factor=3)
        
        # Save augmented dataset
        with open(augmented_path, 'w', encoding='utf-8') as f:
            json.dump(augmented_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Augmented dataset saved to {augmented_path}")
        return augmented_samples
    
    else:
        # Use original dataset
        logger.info(f"📊 Loading original dataset: {base_dataset_path}")
        return load_qa_dataset(base_dataset_path, dataset_size)

def train_v03_safetensors_model(
    qa_samples: List[Dict], 
    config: QATrainingConfig,
    output_dir: str = "models/v03_safetensors_enhanced"
) -> str:
    """Train v0.3 safetensors enhanced model"""
    
    logger.info(f"🏗️ Initializing v0.3 Safetensors Enhanced Training...")
    logger.info(f"   Model: {config.model_name}")
    logger.info(f"   BiLSTM: {config.bilstm_hidden_size}x{config.bilstm_num_layers}")
    logger.info(f"   End Position Weight: {config.end_position_weight}x")
    logger.info(f"   Samples: {len(qa_samples)}")
    
    # Initialize trainer
    trainer = LoRATrainerV3Safetensors(config)
    
    # Load model
    logger.info("🏗️ Loading v0.3 Safetensors Enhanced Architecture...")
    trainer.load_base_model()
    
    # Train model
    logger.info("🚀 Starting v0.3 Safetensors Training...")
    model_path = trainer.train_model(qa_samples, output_dir)
    
    logger.info(f"✅ v0.3 Safetensors Training completed!")
    logger.info(f"   Model saved to: {model_path}")
    
    return model_path

def evaluate_v03_safetensors_model(model_path: str, test_samples: List[Dict]) -> Dict:
    """Evaluate v0.3 safetensors model performance"""
    
    logger.info(f"🔍 Evaluating v0.3 Safetensors Model: {model_path}")
    
    try:
        # Load config
        config_path = Path(model_path) / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            config = QATrainingConfig(**config_dict)
        
        # Initialize trainer and load model
        trainer = LoRATrainerV3Safetensors(config)
        trainer.load_base_model()
        
        # Load trained weights
        model_weights_path = Path(model_path) / "enhanced_model.pth"
        trainer.enhanced_model.load_state_dict(torch.load(model_weights_path))
        
        # Prepare test dataset
        test_dataset, _ = trainer.prepare_datasets(test_samples, split_ratio=1.0)  # Use all as test
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Evaluate
        results = trainer.evaluate_model(test_loader)
        
        logger.info(f"📊 v0.3 Safetensors Model Evaluation Results:")
        logger.info(f"   Start Position Accuracy: {results['start_accuracy']:.1f}%")
        logger.info(f"   End Position Accuracy: {results['end_accuracy']:.1f}%")
        logger.info(f"   Test Samples: {results['total_samples']}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return {'start_accuracy': 0, 'end_accuracy': 0, 'total_samples': 0}

def main():
    parser = argparse.ArgumentParser(description='v0.3 Safetensors Enhanced QA Training Pipeline')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'], default='train', help='Operation mode')
    parser.add_argument('--dataset-size', type=int, default=500, choices=[100, 500, 1000], help='Dataset size')
    parser.add_argument('--use-augmented', action='store_true', help='Use augmented dataset (3x expansion)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--end-position-weight', type=float, default=2.5, help='End position loss weight')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting v0.3 Safetensors Enhanced Pipeline - Mode: {}".format(args.mode))
    
    # Ensure directories exist
    ensure_directories()
    
    try:
        if args.mode in ['train', 'both']:
            logger.info("🔧 Training v0.3 Safetensors Enhanced Model...")
            logger.info("🔧 Starting v0.3 Safetensors End Position Enhanced Training...")
            
            # Setup configuration
            config = QATrainingConfig(
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                end_position_weight=args.end_position_weight
            )
            
            logger.info(f"📋 Training Configuration:")
            logger.info(f"   Dataset Size: {args.dataset_size}")
            logger.info(f"   Use Augmented: {args.use_augmented}")
            logger.info(f"   Epochs: {args.epochs}")
            logger.info(f"   Learning Rate: {args.learning_rate}")
            logger.info(f"   Batch Size: {args.batch_size}")
            logger.info(f"   End Position Weight: {args.end_position_weight}x")
            
            # Load dataset
            base_dataset_path = f"data/processed/qa_dataset_v2/qa_samples_{args.dataset_size}.json"
            
            start_time = time.time()
            qa_samples = load_or_create_augmented_dataset(base_dataset_path, args.dataset_size, args.use_augmented)
            load_time = time.time() - start_time
            
            if args.use_augmented:
                logger.info(f"📈 Dataset Augmentation Complete:")
                logger.info(f"   Original: {args.dataset_size} samples")
                logger.info(f"   Augmented: {len(qa_samples)} samples")
                logger.info(f"   Expansion: {len(qa_samples)/args.dataset_size:.1f}x")
                logger.info(f"   Time: {load_time:.1f}s")
            
            logger.info(f"📊 Loading augmented dataset: data/processed/qa_dataset_v3/qa_samples_{args.dataset_size}_v3_safetensors_augmented.json")
            
            # Load v0.2 base dataset for statistics
            v02_samples = load_qa_dataset(base_dataset_path, args.dataset_size)
            
            # Calculate split
            split_ratio = 0.8
            train_size = int(len(qa_samples) * split_ratio)
            eval_size = len(qa_samples) - train_size
            
            logger.info(f"📊 Dataset Statistics:")
            logger.info(f"   Total samples: {len(qa_samples)}")
            logger.info(f"   Training: {train_size}")
            logger.info(f"   Evaluation: {eval_size}")
            
            # Train model
            model_path = train_v03_safetensors_model(qa_samples, config)
            
            logger.info(f"✅ v0.3 Safetensors Training completed!")
            logger.info(f"📁 Model saved to: {model_path}")
        
        if args.mode in ['evaluate', 'both']:
            logger.info("🔍 Evaluating v0.3 Safetensors Model...")
            
            # Load test dataset
            test_path = f"data/processed/test_dataset/test_samples_300.json"
            if Path(test_path).exists():
                test_samples = load_qa_dataset(test_path, 300)
                
                # Find the latest model if not training
                if args.mode == 'evaluate':
                    model_path = "models/v03_safetensors_enhanced"
                
                # Evaluate
                results = evaluate_v03_safetensors_model(model_path, test_samples)
                
                logger.info(f"📊 Final v0.3 Safetensors Results:")
                logger.info(f"   🎯 End Position Accuracy: {results['end_accuracy']:.1f}% (Target: 25%+)")
                logger.info(f"   🎯 Start Position Accuracy: {results['start_accuracy']:.1f}%")
                
                # Success check
                if results['end_accuracy'] >= 25.0:
                    logger.info("🎉 SUCCESS: End Position accuracy target achieved!")
                elif results['end_accuracy'] >= 20.0:
                    logger.info("✅ GOOD: Significant improvement achieved!")
                else:
                    logger.info("⚠️  More optimization needed...")
            
            else:
                logger.warning(f"Test dataset not found: {test_path}")
        
        logger.info("✅ v0.3 Safetensors Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ v0.3 Safetensors Pipeline Failed: {e}")
        raise

if __name__ == "__main__":
    main()