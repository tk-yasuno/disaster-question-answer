#!/usr/bin/env python3
"""
テストデータ生成器 - 300サンプル
訓練データに含まれない独立したテストケースを生成
"""

import json
import random
import os
from typing import List, Dict, Any


class DisasterTestDataGenerator:
    """災害QAテストデータ生成器 - 訓練データとは独立"""
    
    def __init__(self):
        self.test_templates = self._create_test_templates()
        
    def _create_test_templates(self) -> Dict[str, List[Dict]]:
        """テスト専用のQAテンプレート（訓練データとは異なる内容）"""
        
        return {
            "earthquake": [
                {
                    "question_patterns": [
                        "震度7の大地震が発生した場合の緊急対応は？",
                        "建物が倒壊した際の生存確率を高める方法は？",
                        "地震発生から72時間以内にすべきことは？",
                        "余震が続く中での行動指針は？",
                        "地震による火災から逃れる方法は？"
                    ],
                    "context_patterns": [
                        "震度7の大地震では建物の倒壊リスクが極めて高くなります。まず生存空間を確保し、大声で助けを求めることが重要です。72時間は生存の分かれ目とされています。",
                        "建物倒壊時は、机などの下の三角空間に避難することで生存確率が向上します。声を出し続けて救助隊に位置を知らせ、体力を温存してください。",
                        "地震発生直後の72時間は「災害の黄金時間」です。この間に水の確保、負傷者の応急処置、安全な場所への移動を完了する必要があります。",
                        "余震は本震より小さいとは限りません。倒壊した建物周辺から速やかに離れ、開けた場所で待機してください。余震のたびに避難経路を再確認することが大切です。",
                        "地震による火災では煙を吸わないよう低い姿勢で移動します。風上に向かって避難し、延焼経路を避けて安全な場所まで移動してください。"
                    ],
                    "answer_patterns": [
                        "まず生存空間を確保し、大声で助けを求めることが重要です",
                        "机などの下の三角空間に避難することで生存確率が向上します",
                        "水の確保、負傷者の応急処置、安全な場所への移動を完了する必要があります",
                        "倒壊した建物周辺から速やかに離れ、開けた場所で待機してください",
                        "煙を吸わないよう低い姿勢で移動します"
                    ]
                },
                {
                    "question_patterns": [
                        "地震による液状化現象が起きた場合の対処は？",
                        "地下街にいる時に地震が発生したら？",
                        "高層ビルのエレベーター内で地震に遭遇したら？",
                        "夜間の地震で停電した場合の対応は？",
                        "海外で地震に遭った時の対応方法は？"
                    ],
                    "context_patterns": [
                        "液状化現象では地盤が液体のようになり、建物が傾斜したり沈下したりします。建物から出られる場合は速やかに避難し、固い地盤の場所に移動してください。",
                        "地下街では出口への経路を確認し、停電に備えて壁づたいに移動します。パニックにならず、係員の指示に従って秩序立った避難を心がけてください。",
                        "高層ビルのエレベーター内では緊急停止装置を作動させます。各階のボタンを押して最寄り階で停止させ、ドアが開いたら速やかに階段で避難してください。",
                        "夜間の停電時は懐中電灯を使用し、ろうそくは火災危険のため避けます。家族の安否確認を行い、ラジオで情報収集を継続してください。",
                        "海外では言語の壁がありますが、現地の避難指示に従います。日本領事館への連絡、パスポートなど重要書類の確保、現地の緊急連絡先の確認が必要です。"
                    ],
                    "answer_patterns": [
                        "建物から出られる場合は速やかに避難し、固い地盤の場所に移動してください",
                        "停電に備えて壁づたいに移動します",
                        "緊急停止装置を作動させます",
                        "懐中電灯を使用し、ろうそくは火災危険のため避けます",
                        "現地の避難指示に従います"
                    ]
                }
            ],
            
            "tsunami": [
                {
                    "question_patterns": [
                        "津波の第二波、第三波への対策は？",
                        "津波避難タワーにいる時の注意点は？",
                        "車で津波から逃げる時の判断基準は？",
                        "津波警報解除後の帰宅タイミングは？",
                        "津波に巻き込まれた場合の生存方法は？"
                    ],
                    "context_patterns": [
                        "津波の第一波は偵察波とも呼ばれ、第二波・第三波がより大きくなることがあります。警報解除まで絶対に海岸や河口に近づかず、高台で待機を続けてください。",
                        "津波避難タワーでは備蓄品を確認し、他の避難者と協力して過ごします。強風や寒さに備え、毛布や防寒具を共有し、高齢者や子供を優先してください。",
                        "車での避難は渋滞リスクがあるため、避難距離が長い場合のみ推奨されます。水深30cmで車は動けなくなるため、浸水の兆候があれば車を捨てて高台に避難してください。",
                        "津波警報解除後も、がれきや汚染された水が残っている可能性があります。自治体の安全確認が完了してから段階的に帰宅し、建物の安全点検も必要です。",
                        "津波に巻き込まれた場合は、流木や浮遊物につかまり、体力を温存します。口と鼻を水面上に保ち、救助が来るまで諦めずに浮き続けてください。"
                    ],
                    "answer_patterns": [
                        "警報解除まで絶対に海岸や河口に近づかず、高台で待機を続けてください",
                        "他の避難者と協力して過ごします",
                        "浸水の兆候があれば車を捨てて高台に避難してください",
                        "自治体の安全確認が完了してから段階的に帰宅し、建物の安全点検も必要です",
                        "流木や浮遊物につかまり、体力を温存します"
                    ]
                }
            ],
            
            "typhoon": [
                {
                    "question_patterns": [
                        "台風の目に入った時の注意点は？",
                        "台風による高潮被害への対策は？",
                        "停電が3日間続く場合の対応は？",
                        "台風時のペットの避難方法は？",
                        "台風通過後の外出時の注意点は？"
                    ],
                    "context_patterns": [
                        "台風の目に入ると一時的に風が止みますが、再び激しい風が反対方向から吹きます。外に出ず、台風が完全に通過するまで屋内で待機してください。",
                        "高潮は海水面が普段より2-3m上昇する現象です。海岸や河口付近では早めの避難が必要で、車両は浸水リスクがあるため使用を控えてください。",
                        "長期停電では冷蔵庫の食品が腐敗します。クーラーボックスに氷を入れて保存し、カセットコンロで調理し、携帯電話の充電は計画的に行ってください。",
                        "ペットは指定避難所に入れないことが多いため、ペット可能な施設を事前に確認します。キャリーケース、リード、ペットフード、薬を準備してください。",
                        "台風通過後は倒木や飛来物、冠水箇所があります。外出前に安全を確認し、長靴を履き、懐中電灯を持参して十分注意して歩いてください。"
                    ],
                    "answer_patterns": [
                        "外に出ず、台風が完全に通過するまで屋内で待機してください",
                        "海岸や河口付近では早めの避難が必要です",
                        "クーラーボックスに氷を入れて保存し、カセットコンロで調理してください",
                        "キャリーケース、リード、ペットフード、薬を準備してください",
                        "長靴を履き、懐中電灯を持参して十分注意して歩いてください"
                    ]
                }
            ],
            
            "flood": [
                {
                    "question_patterns": [
                        "内水氾濫が発生した場合の避難方法は？",
                        "車が冠水した道路で立ち往生した時は？",
                        "地下室や地下駐車場にいる時の洪水対応は？",
                        "避難指示が出ても避難できない場合は？",
                        "洪水後の衛生管理で注意すべきことは？"
                    ],
                    "context_patterns": [
                        "内水氾濫では下水道が逆流し、市街地が浸水します。マンホールの蓋が外れている危険があるため、水深が膝上になったら無理な移動は避けてください。",
                        "冠水道路では水深30cmでエンジンが停止し、50cmで車が浮き始めます。車内に閉じ込められる前に、水深が浅いうちに車を捨てて高台に避難してください。",
                        "地下は浸水が早く進み、電気系統が停止します。エレベーターは使わず階段で地上に避難し、電気設備には近づかないでください。",
                        "避難が困難な場合は垂直避難を行います。建物の2階以上に避難し、救助要請の連絡を入れ、屋根やベランダで救助を待ってください。",
                        "洪水後の水は細菌やウイルスで汚染されています。直接触れないよう手袋を着用し、食器や衣服は十分消毒してから使用してください。"
                    ],
                    "answer_patterns": [
                        "水深が膝上になったら無理な移動は避けてください",
                        "水深が浅いうちに車を捨てて高台に避難してください",
                        "エレベーターは使わず階段で地上に避難してください",
                        "建物の2階以上に避難し、救助要請の連絡を入れてください",
                        "直接触れないよう手袋を着用し、食器や衣服は十分消毒してください"
                    ]
                }
            ],
            
            "volcano": [
                {
                    "question_patterns": [
                        "噴火警戒レベル5の時の対応は？",
                        "火山灰が積もった屋根の対処法は？",
                        "噴石から身を守る方法は？",
                        "火山性ガスの危険から避難するには？",
                        "長期避難時の生活再建支援は？"
                    ],
                    "context_patterns": [
                        "噴火警戒レベル5は最高レベルで、居住地域に重大な被害をもたらす噴火が発生または切迫している状態です。自治体の指示に従い、直ちに危険区域外へ避難してください。",
                        "火山灰は水分を含むと重くなり、屋根の倒壊リスクが高まります。できる限り除去作業を行いますが、高所での作業は危険なため複数人で安全確認しながら行ってください。",
                        "噴石は時速100-300kmで飛散し、直撃すると致命的です。登山時は山小屋やシェルター、大きな岩陰に避難し、ヘルメットで頭部を保護してください。",
                        "火山性ガスは低地にたまりやすく、呼吸困難を引き起こします。異臭を感じたら風上の高台に避難し、湿ったタオルで口鼻を覆って呼吸してください。",
                        "長期避難では生活再建支援として、仮設住宅の提供、就労支援、子どもの転校手続き支援があります。市町村の支援制度を確認し、早めに申請手続きを行ってください。"
                    ],
                    "answer_patterns": [
                        "直ちに危険区域外へ避難してください",
                        "複数人で安全確認しながら行ってください",
                        "山小屋やシェルター、大きな岩陰に避難してください",
                        "風上の高台に避難し、湿ったタオルで口鼻を覆って呼吸してください",
                        "市町村の支援制度を確認し、早めに申請手続きを行ってください"
                    ]
                }
            ],
            
            "general": [
                {
                    "question_patterns": [
                        "災害時の要配慮者への支援方法は？",
                        "避難所運営で重要なポイントは？",
                        "災害ボランティア活動で注意すべきことは？",
                        "PTSD予防のための心理的応急処置は？",
                        "災害時の情報収集手段の優先順位は？"
                    ],
                    "context_patterns": [
                        "要配慮者は高齢者、障害者、乳幼児、妊産婦、外国人等です。避難時は車椅子や担架の確保、薬の携帯、コミュニケーション手段の準備が必要です。",
                        "避難所運営ではプライバシー確保、衛生管理、食料配布の公平性が重要です。運営委員会を設置し、避難者同士で協力して秩序ある生活環境を作ってください。",
                        "災害ボランティアは現地のニーズに合わせた活動が重要です。作業服、手袋、マスクを持参し、ボランティア保険に加入して安全第一で活動してください。",
                        "災害後のPTSD予防には安心・安全感の提供、冷静さの促進、自己効力感の向上が重要です。話を聞き、実用的な情報を提供し、社会的つながりを促進してください。",
                        "災害時の情報収集は、まず安全確保、次に公的機関の情報、信頼できるメディア、SNSの順で優先します。デマに惑わされず、複数の情報源で確認してください。"
                    ],
                    "answer_patterns": [
                        "車椅子や担架の確保、薬の携帯、コミュニケーション手段の準備が必要です",
                        "運営委員会を設置し、避難者同士で協力して秩序ある生活環境を作ってください",
                        "作業服、手袋、マスクを持参し、ボランティア保険に加入してください",
                        "話を聞き、実用的な情報を提供し、社会的つながりを促進してください",
                        "デマに惑わされず、複数の情報源で確認してください"
                    ]
                }
            ]
        }
    
    def generate_test_samples(self, num_samples: int = 300, output_dir: str = "data/processed/test_dataset") -> List[Dict[str, Any]]:
        """テストサンプルを生成（訓練データとは独立）"""
        
        samples = []
        disaster_types = list(self.test_templates.keys())
        samples_per_type = num_samples // len(disaster_types)
        
        for disaster_type in disaster_types:
            type_samples = self._generate_test_samples_for_type(
                disaster_type, 
                samples_per_type if disaster_type != 'general' else samples_per_type // 2  # generalは少なめ
            )
            samples.extend(type_samples)
        
        # 不足分を補完
        remaining = num_samples - len(samples)
        if remaining > 0:
            extra_samples = self._generate_test_samples_for_type(
                random.choice(['earthquake', 'tsunami', 'typhoon']), 
                remaining
            )
            samples.extend(extra_samples)
        
        # データを保存
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/test_samples_{num_samples}.json", "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        return samples
    
    def _generate_test_samples_for_type(self, disaster_type: str, num_samples: int) -> List[Dict[str, Any]]:
        """特定の災害タイプのテストサンプルを生成"""
        
        samples = []
        templates = self.test_templates[disaster_type]
        
        for i in range(num_samples):
            template = random.choice(templates)
            
            question = random.choice(template["question_patterns"])
            context = random.choice(template["context_patterns"])
            answer = random.choice(template["answer_patterns"])
            
            # 回答がcontextに含まれていることを確認・修正
            if answer not in context:
                # contextから最も適切な部分を抽出
                sentences = context.split('。')
                for sentence in sentences:
                    if len(sentence) > 10:  # 適切な長さの文を選択
                        answer = sentence.strip()
                        if answer and not answer.endswith('。'):
                            answer += '。'
                        break
                
                if answer not in context:
                    answer = sentences[0] + '。' if sentences[0] else context[:50]
            
            start_char = context.find(answer)
            if start_char == -1:
                start_char = 0
            
            # 質問タイプを判定
            question_type = "WhatAct" if any(word in question for word in ["方法", "対処", "対応", "すべき", "注意"]) else "WhatIs"
            
            sample = {
                "question": question,
                "context": context,
                "answer": answer,
                "start_char": start_char,
                "disaster_type": disaster_type,
                "question_type": question_type,
                "document_source": f"test_{disaster_type}_manual_{i%20+1:02d}.pdf",
                "sample_id": f"test_{disaster_type}_{i+1:03d}",
                "is_test_data": True,  # テストデータであることを明示
                "difficulty_level": random.choice(["basic", "intermediate", "advanced"])
            }
            
            samples.append(sample)
        
        return samples


def main():
    """テストデータ生成のメイン実行"""
    
    generator = DisasterTestDataGenerator()
    
    print("🧪 Disaster Test Dataset Generator")
    print("=" * 50)
    
    # 300サンプルのテストデータ生成
    print("📊 Generating 300 test samples...")
    test_samples = generator.generate_test_samples(300)
    print(f"✅ Generated {len(test_samples)} test samples")
    
    # テストデータの内訳を表示
    disaster_counts = {}
    difficulty_counts = {}
    
    for sample in test_samples:
        # 災害タイプ別カウント
        disaster_type = sample["disaster_type"]
        disaster_counts[disaster_type] = disaster_counts.get(disaster_type, 0) + 1
        
        # 難易度別カウント
        difficulty = sample["difficulty_level"]
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    print("\n📈 Test Data Distribution:")
    print("🏷️  By Disaster Type:")
    for disaster_type, count in disaster_counts.items():
        print(f"   - {disaster_type}: {count} samples")
    
    print("\n🎯 By Difficulty Level:")
    for difficulty, count in difficulty_counts.items():
        print(f"   - {difficulty}: {count} samples")
    
    # サンプル例を表示
    print(f"\n📝 Sample Test Case:")
    sample_case = test_samples[0]
    print(f"   Question: {sample_case['question']}")
    print(f"   Answer: {sample_case['answer'][:60]}...")
    print(f"   Type: {sample_case['disaster_type']}, Level: {sample_case['difficulty_level']}")
    
    print(f"\n💾 Test data saved to: data/processed/test_dataset/test_samples_300.json")
    print("🔬 This test dataset is independent from training data!")
    
    return test_samples


if __name__ == "__main__":
    main()