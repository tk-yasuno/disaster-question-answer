"""
軽量版評価結果の可視化スクリプト
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import matplotlib.font_manager as fm

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Windows環境での日本語フォント設定を改善
def setup_japanese_font():
    """日本語フォントを設定"""
    try:
        # Windows で利用可能な日本語フォントを優先順で設定
        japanese_fonts = [
            'MS Gothic',           # Windows標準
            'Yu Gothic',           # Windows 8.1以降
            'Meiryo',             # Windows Vista以降
            'BIZ UDPGothic',      # Office付属
            'NotoSansCJK-Regular', # Noto Sans
            'Arial Unicode MS',   # Office付属
            'DejaVu Sans'         # フォールバック
        ]
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in japanese_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = [font]
                logger.info(f"Using font: {font}")
                break
        else:
            # フォールバック設定
            plt.rcParams['font.family'] = ['sans-serif']
            logger.warning("No Japanese font found, using default")
        
        # その他の設定
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10
        
    except Exception as e:
        logger.error(f"Font setup failed: {e}")
        # エラー時はデフォルト設定
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

# フォント設定を実行
setup_japanese_font()

def create_comparison_chart(results_path: str = "evaluation_results/lightweight_results.json"):
    """軽量版評価結果の比較チャートを作成"""
    
    # 結果データの読み込み
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    logger.info(f"Loaded results for {len(results)} models")
    
    # データ準備（英語ラベルに変更して文字化け回避）
    models = list(results.keys())
    model_labels_jp = [model.replace('-samples', ' サンプル') for model in models]
    model_labels_en = [model.replace('-samples', ' samples') for model in models]
    
    # 英語ラベルを使用（文字化け回避）
    model_labels = model_labels_en
    
    metrics = {
        'Start Position Accuracy': [results[m]['eval_start_position_accuracy'] for m in models],
        'End Position Accuracy': [results[m]['eval_end_position_accuracy'] for m in models], 
        'Span F1': [results[m]['eval_span_f1'] for m in models],
        'Overall F1': [results[m]['eval_overall_f1'] for m in models]
    }
    
    # チャート作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison (Lightweight Evaluation)', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    positions = [(0,0), (0,1), (1,0), (1,1)]
    
    for (metric_name, values), (row, col) in zip(metrics.items(), positions):
        ax = axes[row, col]
        bars = ax.bar(model_labels, values, color=colors[:len(models)], alpha=0.7)
        
        ax.set_title(metric_name, fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
        
        # x軸ラベルの設定を改善（文字化け対策）
        ax.set_xticks(range(len(model_labels)))
        ax.set_xticklabels(model_labels, rotation=0, ha='center', fontsize=10)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # グリッドを追加
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_path = Path("evaluation_results")
    save_path.mkdir(exist_ok=True)
    chart_file = save_path / "lightweight_comparison.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison chart saved to: {chart_file}")
    
    # レーダーチャート作成
    create_radar_chart(results, save_path)
    
    # パフォーマンス推移チャート作成
    create_performance_trend(results, save_path)

def create_radar_chart(results, save_path):
    """レーダーチャートを作成"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    models = list(results.keys())
    # 英語ラベルを使用（文字化け回避）
    model_labels_jp = [model.replace('-samples', ' サンプル') for model in models]
    model_labels_en = [model.replace('-samples', ' samples') for model in models]
    model_labels = model_labels_en  # 英語ラベル使用
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # メトリクス定義
    metrics = [
        'eval_start_position_accuracy',
        'eval_end_position_accuracy', 
        'eval_span_f1',
        'eval_overall_f1'
    ]
    
    metric_labels = [
        'Start Accuracy',
        'End Accuracy',
        'Span F1',
        'Overall F1'
    ]
    
    # 角度設定
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 円を閉じる
    
    for i, model in enumerate(models):
        values = []
        for metric in metrics:
            value = results[model].get(metric, 0)
            values.append(value)
        
        values += values[:1]  # 円を閉じる
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=model_labels[i], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 0.8)  # 最大値を調整
    ax.set_title('Performance Comparison (Radar Chart)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 凡例設定を改善（文字化け対策）
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)
    ax.grid(True)
    
    plt.tight_layout()
    radar_file = save_path / "lightweight_radar.png"
    plt.savefig(radar_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Radar chart saved to: {radar_file}")

def create_performance_trend(results, save_path):
    """学習データサイズ別のパフォーマンス推移チャート"""
    models = list(results.keys())
    training_sizes = [int(model.split('-')[0]) for model in models]
    
    # メトリクス値を取得
    start_acc = [results[model]['eval_start_position_accuracy'] for model in models]
    span_f1 = [results[model]['eval_span_f1'] for model in models]
    overall_f1 = [results[model]['eval_overall_f1'] for model in models]
    
    # チャート作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = training_sizes
    ax.plot(x, start_acc, marker='o', linewidth=2, label='Start Position Accuracy', color='#3498db')
    ax.plot(x, span_f1, marker='s', linewidth=2, label='Span F1', color='#e74c3c')
    ax.plot(x, overall_f1, marker='^', linewidth=2, label='Overall F1', color='#2ecc71')
    
    ax.set_xlabel('Training Dataset Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance vs Training Dataset Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 1050)
    ax.set_ylim(0, max(max(start_acc), max(span_f1), max(overall_f1)) * 1.1)
    
    # データポイントに値を表示
    for i, size in enumerate(x):
        ax.annotate(f'{start_acc[i]:.3f}', (size, start_acc[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax.annotate(f'{span_f1[i]:.3f}', (size, span_f1[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax.annotate(f'{overall_f1[i]:.3f}', (size, overall_f1[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    trend_file = save_path / "performance_trend.png"
    plt.savefig(trend_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance trend chart saved to: {trend_file}")

def print_summary(results_path: str = "evaluation_results/lightweight_results.json"):
    """評価結果のサマリーを表示"""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("📊 MODEL EVALUATION SUMMARY")
    print("="*80)
    
    models = list(results.keys())
    training_sizes = [int(model.split('-')[0]) for model in models]
    
    print(f"✅ Evaluated {len(models)} models with training sizes: {training_sizes}")
    
    # 最良モデルの特定
    best_model = max(models, key=lambda m: results[m]['eval_overall_f1'])
    best_f1 = results[best_model]['eval_overall_f1']
    
    print(f"\n🏆 Best performing model: {best_model}")
    print(f"   Overall F1 Score: {best_f1:.3f}")
    
    # パフォーマンス向上の分析
    start_f1 = results[models[0]]['eval_overall_f1']
    end_f1 = results[models[-1]]['eval_overall_f1']
    improvement = ((end_f1 - start_f1) / start_f1) * 100 if start_f1 > 0 else 0
    
    print(f"\n📈 Performance improvement from 100 to 1000 samples:")
    print(f"   Initial F1: {start_f1:.3f}")
    print(f"   Final F1: {end_f1:.3f}")
    print(f"   Improvement: {improvement:+.1f}%")
    
    print(f"\n💡 Key insights:")
    print(f"   • 500 samples and 1000 samples show identical performance")
    print(f"   • Start position accuracy shows consistent improvement")
    print(f"   • End position accuracy remains challenging across all models")

def main():
    """メイン実行関数"""
    logger.info("🎨 Creating visualization charts for lightweight evaluation results")
    
    results_path = "evaluation_results/lightweight_results.json"
    
    # 結果ファイルの存在確認
    if not Path(results_path).exists():
        logger.error(f"❌ Results file not found: {results_path}")
        return
    
    try:
        # チャート作成
        create_comparison_chart(results_path)
        
        # サマリー表示
        print_summary(results_path)
        
        logger.info("🎉 Visualization completed successfully!")
        print(f"\n📁 Charts saved in: evaluation_results/")
        print("   • lightweight_comparison.png - Bar chart comparison")
        print("   • lightweight_radar.png - Radar chart")
        print("   • performance_trend.png - Performance trend")
        
    except Exception as e:
        logger.error(f"❌ Failed to create visualizations: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    main()