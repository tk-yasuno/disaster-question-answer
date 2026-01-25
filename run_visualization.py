"""
文字化け修正版 - 可視化スクリプト実行
"""
import subprocess
import sys

def run_visualization():
    """修正された可視化スクリプトを実行"""
    try:
        # Python スクリプトを実行
        result = subprocess.run([
            sys.executable, 
            "visualize_results.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ 可視化チャートの生成が完了しました")
            print(result.stdout)
        else:
            print("❌ エラーが発生しました")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ タイムアウトが発生しました")
    except Exception as e:
        print(f"❌ 実行エラー: {e}")

if __name__ == "__main__":
    run_visualization()