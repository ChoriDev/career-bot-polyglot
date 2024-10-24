import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from model import Item
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 모델 파일 경로 설정
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model/career_counseler.pth')

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 아키텍처 정의
model_id = 'beomi/KoAlpaca-Polyglot-5.8B'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={'':0})

tokenizer.pad_token = tokenizer.eos_token

# Low bit 학습 준비
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA 파라미터 정의
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=['query_key_value'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, config)

# 상태 딕셔너리 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)  # strict=False를 사용하여 불일치하는 키 무시

model.eval()
model.config.use_cache = True # silence the warnings. Please re-enable for inference!

def gen(x):
    inputs = tokenizer(
        f'### 질문: {x}\n\n### 답변:',
        return_tensors='pt',
        return_token_type_ids=False
    )
    gened = model.generate(
        **inputs,
        max_new_tokens=100,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    generated_text = tokenizer.decode(gened[0]).split('###')[2]
    return generated_text.strip()

@app.post('/comment')
async def receive_data(item: Item):
    comment = gen(item.sentence)
    return {'comment': comment}