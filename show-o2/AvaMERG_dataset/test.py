from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")  # 실제 사용 모델명으로 교체
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_tokens(['<image>', '<|vid_start|>', '<|vid_end|>'])

text = "A man walks with little kitten in sunny day."

try:
    encoding = tokenizer(text, add_special_tokens=False, truncation=True, max_length=512)
    print("Tokens:", encoding.input_ids)
except Exception as e:
    print("Tokenizer error:", e)
