#!/usr/bin/env python3
"""
v0.2: DisaQuADデータセット拡張ジェネレーター
5サンプル → 100サンプル → 500サンプル → 1000サンプルへの段階的拡張
"""

import json
import random
import os
from typing import List, Dict, Any


class DisaQuADDatasetGenerator:
    """災害QAデータセット生成器"""
    
    def __init__(self):
        self.disaster_templates = self._create_disaster_templates()
        
    def _create_disaster_templates(self) -> Dict[str, List[Dict]]:
        """災害タイプ別のQAテンプレートを作成"""
        
        return {
            "earthquake": [
                {
                    "question_patterns": [
                        "地震が発生したときにまず何をすべきですか？",
                        "地震の揺れを感じた場合の対処法は？",
                        "地震発生時の安全確保の方法は？",
                        "地震の時に避難する際の注意点は？"
                    ],
                    "context_patterns": [
                        "地震が発生した場合、まず自分の安全を確保することが重要です。机の下に隠れるか、頭を保護してください。その後、火の元を確認し、ガスの元栓を閉めてください。",
                        "地震の揺れを感じたら、まず落下物から身を守るため安全な場所に避難してください。慌てて外に飛び出すのは危険です。揺れが収まってから避難経路を確認してください。",
                        "強い地震が発生した際は、まずドロップ・カバー・ホールドオン（身を低くし、隠れ、しがみつく）を実行してください。机の下に潜り込み、机の脚をしっかりと掴んでください。"
                    ],
                    "answer_patterns": [
                        "まず自分の安全を確保することが重要です",
                        "まず落下物から身を守るため安全な場所に避難してください",
                        "まずドロップ・カバー・ホールドオンを実行してください"
                    ]
                },
                {
                    "question_patterns": [
                        "地震後の二次災害で注意すべきことは？",
                        "地震による停電時の対応は？", 
                        "地震でエレベーターに閉じ込められた場合は？"
                    ],
                    "context_patterns": [
                        "地震後は火災や建物倒壊などの二次災害に注意が必要です。ガス漏れがないか確認し、損傷した建物には近づかないでください。余震にも備えてください。",
                        "地震による停電が発生した場合、懐中電灯やラジオを使用してください。ロウソクの使用は火災の危険があるため避けてください。",
                        "エレベーターに閉じ込められた場合は、非常ボタンを押して外部に連絡してください。無理にドアを開けようとせず、救助を待ってください。"
                    ],
                    "answer_patterns": [
                        "ガス漏れがないか確認し、損傷した建物には近づかないでください",
                        "懐中電灯やラジオを使用してください",
                        "非常ボタンを押して外部に連絡してください"
                    ]
                }
            ],
            
            "tsunami": [
                {
                    "question_patterns": [
                        "津波警報が出された場合の対応は？",
                        "津波から避難する際の注意点は？",
                        "津波警報を聞いた時にすべきことは？",
                        "沿岸部にいる時に津波警報が出たらどうする？"
                    ],
                    "context_patterns": [
                        "津波警報が発令された場合、直ちに高台または津波避難ビルに避難してください。海岸や河川の近くにいる場合は、すぐに離れてください。",
                        "津波の避難は時間との勝負です。車での避難は渋滞の可能性があるため、徒歩での避難を基本としてください。",
                        "津波注意報や警報が発表されたら、海や川に近づかないことが重要です。津波は第一波よりも第二波、第三波の方が大きくなることもあります。"
                    ],
                    "answer_patterns": [
                        "直ちに高台または津波避難ビルに避難してください",
                        "徒歩での避難を基本としてください",
                        "海や川に近づかないことが重要です"
                    ]
                }
            ],
            
            "typhoon": [
                {
                    "question_patterns": [
                        "台風接近時の家庭での準備は？",
                        "台風が来る前にしておくべきことは？",
                        "台風対策として窓の補強方法は？",
                        "台風による停電に備えるには？"
                    ],
                    "context_patterns": [
                        "台風が接近している場合、窓ガラスに飛散防止フィルムを貼るか、カーテンを閉めてください。また、停電に備えて懐中電灯や非常用電源を準備してください。",
                        "台風対策では、屋外の物を室内に取り込むか、しっかりと固定することが重要です。食料や水を3日分確保し、ラジオや懐中電灯も用意してください。",
                        "台風による強風から窓を守るため、雨戸やシャッターを閉める、または段ボールやベニヤ板で補強してください。飛散防止フィルムも効果的です。"
                    ],
                    "answer_patterns": [
                        "窓ガラスに飛散防止フィルムを貼るか、カーテンを閉めてください",
                        "屋外の物を室内に取り込むか、しっかりと固定してください",
                        "雨戸やシャッターを閉める、または段ボールやベニヤ板で補強してください"
                    ]
                }
            ],
            
            "flood": [
                {
                    "question_patterns": [
                        "洪水警報とは何ですか？",
                        "河川氾濫の危険がある時の対応は？",
                        "浸水被害から身を守る方法は？",
                        "洪水発生時の避難のタイミングは？"
                    ],
                    "context_patterns": [
                        "洪水警報は、河川の水位が上昇し、氾濫の恐れがある場合に気象庁が発表する警報です。この警報が出された地域では、河川や用水路の近くには近づかず、避難の準備をしてください。",
                        "河川氾濫の危険性が高まった場合は、早めの避難が重要です。浸水が始まってからの避難は危険を伴います。警戒レベル3で避難準備を開始してください。",
                        "浸水が予想される地域では、電気のブレーカーを切り、ガスの元栓を閉めてください。また、重要な書類や貴重品は高い場所に移してください。"
                    ],
                    "answer_patterns": [
                        "河川の水位が上昇し、氾濫の恐れがある場合に気象庁が発表する警報です",
                        "早めの避難が重要です",
                        "電気のブレーカーを切り、ガスの元栓を閉めてください"
                    ]
                }
            ],
            
            "volcano": [
                {
                    "question_patterns": [
                        "火山噴火警報とは何ですか？",
                        "噴火時の避難方法は？",
                        "火山灰から身を守る方法は？",
                        "火砕流の危険性について教えて"
                    ],
                    "context_patterns": [
                        "火山噴火警報は、火山活動が活発化し、噴火の可能性が高まった場合に発表されます。警戒が必要なレベルに応じて、入山規制や避難が実施されます。",
                        "火山噴火時の避難では、風向きを考慮して火山灰の飛散方向を避けることが重要です。また、火砕流の危険がある場合は直ちに安全な場所に避難してください。",
                        "火山灰から身を守るには、マスクやタオルで口鼻を覆い、ゴーグルで目を保護してください。また、建物内に避難し、窓を閉めて火山灰の侵入を防いでください。"
                    ],
                    "answer_patterns": [
                        "火山活動が活発化し、噴火の可能性が高まった場合に発表されます",
                        "風向きを考慮して火山灰の飛散方向を避けることが重要です",
                        "マスクやタオルで口鼻を覆い、ゴーグルで目を保護してください"
                    ]
                }
            ]
        }
        
    def generate_samples(self, num_samples: int = 100, output_dir: str = "data/processed/qa_dataset_v2") -> List[Dict[str, Any]]:
        """指定された数のQAサンプルを生成"""
        
        samples = []
        disaster_types = list(self.disaster_templates.keys())
        samples_per_type = num_samples // len(disaster_types)
        
        for disaster_type in disaster_types:
            type_samples = self._generate_disaster_samples(
                disaster_type, 
                samples_per_type
            )
            samples.extend(type_samples)
        
        # 不足分を補完
        remaining = num_samples - len(samples)
        if remaining > 0:
            extra_samples = self._generate_disaster_samples(
                random.choice(disaster_types), 
                remaining
            )
            samples.extend(extra_samples)
        
        # データを保存
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/qa_samples_{num_samples}.json", "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        return samples
    
    def _generate_disaster_samples(self, disaster_type: str, num_samples: int) -> List[Dict[str, Any]]:
        """特定の災害タイプのサンプルを生成"""
        
        samples = []
        templates = self.disaster_templates[disaster_type]
        
        for i in range(num_samples):
            template = random.choice(templates)
            
            question = random.choice(template["question_patterns"])
            context = random.choice(template["context_patterns"]) 
            answer = random.choice(template["answer_patterns"])
            
            # 回答の開始位置を計算（確実にcontextに含まれるように）
            if answer in context:
                start_char = context.find(answer)
            else:
                # 答えがcontextに含まれていない場合、contextから適切な部分を抽出
                # 最初の重要な文を答えとする
                sentences = context.split('。')
                if len(sentences) > 1:
                    answer = sentences[0] + '。'
                    start_char = 0
                else:
                    answer = context[:30] + '...'  # 最初の30文字を答えとする
                    start_char = 0
            
            # 質問タイプを判定
            question_type = "WhatAct" if any(word in question for word in ["方法", "対処", "対応", "すべき"]) else "WhatIs"
            
            sample = {
                "question": question,
                "context": context,
                "answer": answer,
                "start_char": start_char,
                "disaster_type": disaster_type,
                "question_type": question_type,
                "document_source": f"{disaster_type}_manual_{i%10+1:02d}.pdf",
                "sample_id": f"{disaster_type}_{i+1:03d}"
            }
            
            samples.append(sample)
        
        return samples


def main():
    """データセット生成のメイン実行"""
    
    generator = DisaQuADDatasetGenerator()
    
    print("🎯 DisaQuAD Dataset Generator v0.2")
    print("=" * 50)
    
    # 段階1: 100サンプル生成
    print("📊 Step 1: Generating 100 samples...")
    samples_100 = generator.generate_samples(100)
    print(f"✅ Generated {len(samples_100)} samples")
    
    # 段階2: 500サンプル生成
    print("\n📊 Step 2: Generating 500 samples...")
    samples_500 = generator.generate_samples(500)
    print(f"✅ Generated {len(samples_500)} samples")
    
    # 段階3: 1000サンプル生成
    print("\n📊 Step 3: Generating 1000 samples...")
    samples_1000 = generator.generate_samples(1000)
    print(f"✅ Generated {len(samples_1000)} samples")
    
    # 最終統計
    print("\n📈 Generation Summary:")
    for size, samples in [(100, samples_100), (500, samples_500), (1000, samples_1000)]:
        disaster_counts = {}
        for sample in samples:
            disaster_type = sample["disaster_type"] 
            disaster_counts[disaster_type] = disaster_counts.get(disaster_type, 0) + 1
        
        print(f"\n🔢 {size} samples distribution:")
        for disaster_type, count in disaster_counts.items():
            print(f"  - {disaster_type}: {count} samples")
    
    print(f"\n💾 All datasets saved to: data/processed/qa_dataset_v2/")
    
    return samples_100, samples_500, samples_1000


if __name__ == "__main__":
    main()