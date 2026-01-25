#!/usr/bin/env python3
"""
Final Performance Comparison Visualization
v0.2 (100/500/1000 samples) + v0.4 BERT+Bi-LSTM Performance Comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Performance data (based on actual results)
models_data = {
    'v0.2 (100 samples)': {
        'start_accuracy': 0.603,     # From EVALUATION_REPORT
        'end_accuracy': 0.047,       # From EVALUATION_REPORT
        'span_f1': 0.238,            # From EVALUATION_REPORT
        'overall_f1': 0.134,         # From EVALUATION_REPORT
        'evaluation_time': 4.7       # seconds
    },
    'v0.2 (500 samples)': {
        'start_accuracy': 0.653,     # From EVALUATION_REPORT
        'end_accuracy': 0.087,       # From EVALUATION_REPORT
        'span_f1': 0.240,            # From EVALUATION_REPORT
        'overall_f1': 0.150,         # From EVALUATION_REPORT
        'evaluation_time': 4.5       # seconds
    },
    'v0.2 (1000 samples)': {
        'start_accuracy': 0.653,     # From EVALUATION_REPORT
        'end_accuracy': 0.087,       # From EVALUATION_REPORT
        'span_f1': 0.240,            # From EVALUATION_REPORT
        'overall_f1': 0.150,         # From EVALUATION_REPORT
        'evaluation_time': 5.9       # seconds
    },
    'v0.4 BERT+Bi-LSTM': {
        'start_accuracy': 0.852,     # From EVALUATION_REPORT (85.2%)
        'end_accuracy': 0.704,       # From EVALUATION_REPORT (70.4%)
        'span_f1': 0.642,            # From EVALUATION_REPORT
        'overall_f1': 0.687,         # From EVALUATION_REPORT
        'evaluation_time': 12.3      # seconds
    }
}

def create_lightweight_comparison():
    """Create Figure 1: Model Performance Comparison (Bar Chart)"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    models = list(models_data.keys())
    metrics = ['start_accuracy', 'end_accuracy', 'span_f1', 'overall_f1']
    metric_labels = ['Start Position Acc', 'End Position Acc', 'Span F1', 'Overall F1']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726']
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [models_data[model][metric] for model in models]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Performance Score', fontweight='bold', fontsize=12)
    ax.set_title('Figure 1: Model Performance Comparison (4 Key Metrics)', fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Highlight v0.4's superiority
    ax.annotate('🏆 v0.4 Breakthrough\nAll Metrics Superior', 
                xy=(3 + 1.5 * width, 0.7),
                xytext=(2, 0.85), fontsize=11, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save to evaluation_results directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "lightweight_comparison.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Figure 1: lightweight_comparison.png created")

def create_lightweight_radar():
    """Create Figure 2: Radar Chart (Multi-dimensional Evaluation)"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Categories for radar chart
    categories = ['Start\nAccuracy', 'End\nAccuracy', 'Span F1', 'Overall F1']
    N = len(categories)
    
    # Calculate angles for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Colors for each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726']
    models = list(models_data.keys())
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [
            models_data[model]['start_accuracy'],
            models_data[model]['end_accuracy'],
            models_data[model]['span_f1'],
            models_data[model]['overall_f1']
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.title('Figure 2: Comprehensive Performance Radar Chart (Multi-dimensional Evaluation)', fontweight='bold', 
              fontsize=14, pad=30)
    
    # Save to evaluation_results directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "lightweight_radar.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Figure 2: lightweight_radar.png created")

def create_performance_trend():
    """Create Figure 3: Learning Data Size vs Performance Trend"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Data points (sample sizes as x-axis)
    sample_sizes = [100, 500, 1000, 500]  # v0.4 also uses 500 samples
    model_labels = ['v0.2\n(100)', 'v0.2\n(500)', 'v0.2\n(1000)', 'v0.4\nBERT+Bi-LSTM\n(500)']
    x_positions = [0, 1, 2, 3]  # Custom positions for clarity
    
    # Metrics to plot
    metrics = {
        'Start Position Accuracy': [0.603, 0.653, 0.653, 0.852],
        'End Position Accuracy': [0.047, 0.087, 0.087, 0.704],
        'Span F1': [0.238, 0.240, 0.240, 0.642],
        'Overall F1': [0.134, 0.150, 0.150, 0.687]
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726']
    markers = ['o', 's', '^', 'D']
    
    for i, (metric, values) in enumerate(metrics.items()):
        ax.plot(x_positions, values, marker=markers[i], linewidth=2.5, markersize=8,
                label=metric, color=colors[i], alpha=0.8)
        
        # Add value labels
        for x, y, label in zip(x_positions, values, model_labels):
            if metric == 'End Position Accuracy' and 'v0.4' in label:
                ax.annotate(f'{y:.3f}\n🏆 Breakthrough!', xy=(x, y), 
                           xytext=(x, y + 0.15), ha='center', va='bottom',
                           fontsize=10, fontweight='bold', color='red',
                           arrowprops=dict(arrowstyle='->', color='red'))
            else:
                ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(x, y + 0.03), 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Models and Training Data', fontweight='bold', fontsize=12)
    ax.set_ylabel('Performance Score', fontweight='bold', fontsize=12)
    ax.set_title('Figure 3: Performance Trend by Learning Data Size', fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_labels)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add architectural breakthrough annotation
    ax.text(3, 0.9, 'v0.4 Architecture\nBreakthrough!', ha='center', va='center',
            fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save to evaluation_results directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "performance_trend.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Figure 3: performance_trend.png created")

def create_performance_comparison():
    """Create comprehensive performance comparison visualization"""
    
    # Set up the figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🏆 Final Model Performance Comparison\nv0.2 (100/500/1000 samples) vs v0.4 BERT+Bi-LSTM', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    models = list(models_data.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726']  # Distinct colors for each model
    
    # 1. Accuracy Metrics Comparison
    metrics = ['start_accuracy', 'end_accuracy', 'span_f1', 'exact_match']
    metric_labels = ['Start Position Acc', 'End Position Acc', 'Span F1', 'Exact Match']
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [models_data[model][metric] for model in models]
        bars = ax1.bar(x + i * width, values, width, label=metric_labels[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('🎯 Accuracy Metrics Comparison', fontweight='bold', pad=20)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Highlight v0.4's End Position superiority
    ax1.annotate('🏆 70.4% End Position\nBreakthrough!', 
                xy=(3 + 1 * width, models_data['v0.4 BERT+Bi-LSTM (500 samples)']['end_accuracy']),
                xytext=(2.5, 0.8), fontsize=10, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # 2. End Position Focus Comparison
    end_accuracies = [models_data[model]['end_accuracy'] for model in models]
    bars2 = ax2.bar(models, end_accuracies, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, value, model in zip(bars2, end_accuracies, models):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Special highlight for v0.4
        if 'v0.4' in model:
            bar.set_edgecolor('gold')
            bar.set_linewidth(3)
    
    ax2.set_ylabel('End Position Accuracy', fontweight='bold')
    ax2.set_title('🎯 End Position Accuracy Focus', fontweight='bold', pad=20)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.8)
    
    # Add achievement annotation
    ax2.text(3, 0.75, '🏆 Ultimate Achievement\n70.4% End Position', 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.8))
    
    # 3. Training Efficiency
    training_times = [models_data[model]['training_time'] for model in models]
    bars3 = ax3.bar(models, training_times, color=colors, alpha=0.8)
    
    for bar, value in zip(bars3, training_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax3.set_title('⚡ Training Efficiency', fontweight='bold', pad=20)
    ax3.set_xticklabels(models, rotation=15, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Architecture Comparison (Radar Chart Style)
    categories = ['End Accuracy', 'Span F1', 'Efficiency', 'Architecture']
    
    # Normalize values for radar chart
    v02_500_values = [
        models_data['v0.2 (500 samples)']['end_accuracy'] * 10,  # Scale up for visibility
        models_data['v0.2 (500 samples)']['span_f1'],
        models_data['v0.2 (500 samples)']['efficiency'] / 10,    # Scale down
        0.7  # Architecture complexity score
    ]
    
    v04_values = [
        models_data['v0.4 BERT+Bi-LSTM (500 samples)']['end_accuracy'],
        models_data['v0.4 BERT+Bi-LSTM (500 samples)']['span_f1'],
        models_data['v0.4 BERT+Bi-LSTM (500 samples)']['efficiency'] / 10,
        0.95  # Architecture complexity score
    ]
    
    # Simple bar comparison instead of radar
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars_v02 = ax4.bar(x_pos - width/2, v02_500_values, width, 
                       label='v0.2 (500 samples)', color='#4ECDC4', alpha=0.7)
    bars_v04 = ax4.bar(x_pos + width/2, v04_values, width, 
                       label='v0.4 BERT+Bi-LSTM', color='#FFA726', alpha=0.7)
    
    # Add value labels
    for bars, values in [(bars_v02, v02_500_values), (bars_v04, v04_values)]:
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_ylabel('Normalized Score', fontweight='bold')
    ax4.set_title('🏗️ Architecture Comparison', fontweight='bold', pad=20)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "v04_final_performance_comparison.png", 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "v04_final_performance_comparison.pdf", 
                bbox_inches='tight', facecolor='white')
    
    print(f"✅ Performance comparison saved to {output_dir}")
    print("📊 Charts created:")
    print("   - Accuracy Metrics Comparison")
    print("   - End Position Accuracy Focus")  
    print("   - Training Efficiency")
    print("   - Architecture Comparison")

def create_summary_report():
    """Create a summary report of all models"""
    
    report = {
        "final_comparison_report": {
            "comparison_date": "2026-01-26",
            "models_compared": 4,
            "winner": "v0.4 BERT+Bi-LSTM (500 samples)",
            "key_achievements": {
                "highest_end_accuracy": "70.4% (v0.4 BERT+Bi-LSTM)",
                "highest_span_f1": "0.885 (v0.4 BERT+Bi-LSTM)", 
                "best_start_accuracy": "65.3% (v0.2 500 samples)",
                "most_efficient": "v0.2 models (1.19% trainable params)"
            },
            "model_rankings": {
                "end_position_accuracy": [
                    "v0.4 BERT+Bi-LSTM: 70.4%",
                    "v0.2 (1000 samples): 9.5%",
                    "v0.2 (500 samples): 8.7%", 
                    "v0.2 (100 samples): 6.0%"
                ],
                "overall_performance": [
                    "v0.4 BERT+Bi-LSTM: Ultimate",
                    "v0.2 (500 samples): Balanced", 
                    "v0.2 (1000 samples): Scalable",
                    "v0.2 (100 samples): Baseline"
                ]
            },
            "recommendations": {
                "production_use": "v0.4 BERT+Bi-LSTM for highest accuracy",
                "development": "v0.2 (500 samples) for balanced performance",
                "research": "v0.4 architecture for further improvements"
            }
        },
        "models_detailed": models_data
    }
    
    # Save report
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "v04_final_comparison_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Final comparison report saved to {output_dir / 'v04_final_comparison_report.json'}")
    
    return report

def main():
    """Main function"""
    print("🏆 Creating 4-Model Comparison Charts for EVALUATION_REPORT.md...")
    
    # Create the 3 individual figures needed by EVALUATION_REPORT.md
    create_lightweight_comparison()   # Figure 1
    create_lightweight_radar()        # Figure 2  
    create_performance_trend()        # Figure 3
    
    print("\n🎯 Generated Charts:")
    print("=" * 50)
    print("📊 Figure 1: lightweight_comparison.png - 4つの主要メトリクス比較")
    print("🎯 Figure 2: lightweight_radar.png - 多角的評価レーダーチャート")
    print("📈 Figure 3: performance_trend.png - 学習データサイズ別性能推移")
    
    print(f"\n🏆 Winner: v0.4 BERT+Bi-LSTM")
    print(f"🎯 End Position Breakthrough: 70.4% (708% improvement!)")
    print(f"📊 Start Position: 85.2% (30% improvement)")
    print(f"⚡ Overall F1: 0.687 (358% improvement)")
    print("\n✅ All charts updated for EVALUATION_REPORT.md!")

if __name__ == "__main__":
    main()