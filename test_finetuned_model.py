#!/usr/bin/env python3
"""
ファインチューニング済みモデルのテスト
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from peft import PeftModel

def test_finetuned_model():
    """ファインチューニング済みモデルをテスト"""
    
    # ベースモデルとトークナイザーを読み込み
    base_model_name = "cl-tohoku/bert-base-japanese-v3"
    peft_model_path = "models/lora_finetuned_bert-base-japanese-v3"
    
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForQuestionAnswering.from_pretrained(
        base_model_name, 
        use_safetensors=True
    )
    
    print("Loading fine-tuned LoRA model...")
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    # テストデータ
    test_cases = [
        {
            "question": "地震が発生したときにまず何をすべきですか？",
            "context": "地震が発生した場合、まず自分の安全を確保することが重要です。机の下に隠れるか、頭を保護してください。その後、火の元を確認し、ガスの元栓を閉めてください。"
        },
        {
            "question": "津波警報が出たらどうすればいいですか？",
            "context": "津波警報が発表されたら、直ちに高台や頑丈な建物の3階以上に避難してください。海や川から離れ、津波の到達予想時間を確認してください。"
        }
    ]
    
    print("\nTesting fine-tuned model:")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        context = test_case["context"]
        
        print(f"\nTest {i}:")
        print(f"Question: {question}")
        print(f"Context: {context}")
        
        # トークナイズ
        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        
        # 予測実行
        with torch.no_grad():
            outputs = model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            start_idx = torch.argmax(start_scores, dim=1).item()
            end_idx = torch.argmax(end_scores, dim=1).item()
            
            # 回答抽出
            input_ids = inputs["input_ids"][0]
            if start_idx <= end_idx:
                answer_tokens = input_ids[start_idx:end_idx+1]
                answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            else:
                answer = "[No answer found]"
        
        print(f"Answer: {answer}")
        print(f"Start position: {start_idx}, End position: {end_idx}")
        print("-" * 30)
    
    print("\nFine-tuned model test completed!")

if __name__ == "__main__":
    test_finetuned_model()