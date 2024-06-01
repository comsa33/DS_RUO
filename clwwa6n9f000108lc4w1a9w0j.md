---
title: "도메인 최적화 파인튜닝: Ai 기반 Toeic 문제 생성 모델 개발"
datePublished: Sat Jun 01 2024 15:41:44 GMT+0000 (Coordinated Universal Time)
cuid: clwwa6n9f000108lc4w1a9w0j
slug: ai-toeic
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1717256313669/5b4b7f28-8aff-4d22-85ee-4f0d2bd3a4f7.webp
tags: ai, machine-learning, nlp, deep-learning, pytorch, llm, gradient-descent, finetuning, cudos, toeic, unsloth, language-model, hugging-face, domain-optimization, model-training

---

## 1\. 서론

### 프로젝트 배경 및 목적

이 프로젝트는 TOEIC 문제를 생성하는 AI 모델을 도메인 최적화 파인튜닝을 통해 개발하는 것입니다. 이 모델은 [https://toeic4all.com](https://toeic4all.com) 서비스에서 문제를 생성하는 데 사용됩니다.

### 사용한 기술 스택

* Python
    
* PyTorch
    
* Hugging Face Transformers
    
* Unsloth
    
* Weights & Biases
    
* CUDA 12.1
    

## 2\. 환경 설정

### Unsloth 설치 방법 (CUDA 12.1 기준)

[https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)

```bash
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

### 프로젝트 구조 소개

```plaintext
project/
├── config_llama3_openko_8b.yaml
├── data_prep.py
├── model_prep.py
├── model_save.py
├── train.py
└── test.py
```

## 3\. 데이터 준비

### 데이터셋 로드 및 전처리

training 데이터셋을 로드하고 전처리하는 코드입니다.

**data\_**[**prep.py**](http://prep.py)

```python
from datasets import load_dataset

def load_and_format_dataset(token, repo_name, eos_token, alpaca_prompt):
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction=instruction, input=input, response=output) + eos_token
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset(repo_name, split="train", token=token)
    return dataset.map(formatting_prompts_func, batched=True)
```

## 4\. 모델 준비

### 모델 로드 및 준비 코드

모델을 로드하고 PEFT 모델을 설정하는 코드입니다.

**model\_**[**prep.py**](http://prep.py)

```python
from unsloth import FastLanguageModel

def prepare_model(model_name, max_seq_length, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens",
            "lm_head"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=123,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer
```

## 5\. 모델 학습

### 학습 설정 및 초기화

학습 설정 및 초기화 코드입니다.

[**train.py**](http://train.py)

```python
import yaml

import wandb
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

from config import load_config
from prompts import alpaca_prompt
from model_prep import prepare_model
from data_prep import load_and_format_dataset

class ModelTrainer:
    def __init__(self, yaml_config_path, output_dir='outputs'):
        self.config = load_config()
        self.yaml_config = self.load_yaml_config(yaml_config_path)
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.dataset = None

        # Initialize Weights & Biases
        wandb.init(
            project=self.yaml_config['wandb']['project_name'],
            config=self.yaml_config,
        )

    def load_yaml_config(self, yaml_file):
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)

    def prepare_model_and_dataset(self):
        model_name = self.yaml_config['model']['name']
        dataset_repo_name = self.yaml_config['dataset']['repo_name']
        max_seq_length = self.yaml_config['model']['max_seq_length']
        prompt = alpaca_prompt

        self.model, self.tokenizer = prepare_model(model_name, max_seq_length)
        self.dataset = load_and_format_dataset(
            self.config['HUB_TOKEN'],
            dataset_repo_name,
            self.tokenizer.eos_token,
            prompt
        )

        self.tokenizer.padding_side = self.yaml_config['tokenizer']['padding_side']

    def train_model(self):
        self.prepare_model_and_dataset()
        
        training_args = TrainingArguments(
            per_device_train_batch_size=self.yaml_config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=self.yaml_config['training']['gradient_accumulation_steps'],
            warmup_steps=self.yaml_config['training']['warmup_steps'],
            num_train_epochs=self.yaml_config['training']['num_train_epochs'],
            max_steps=self.yaml_config['training']['max_steps'],
            logging_steps=self.yaml_config['training']['logging_steps'],
            save_steps=self.yaml_config['training']['save_steps'],
            learning_rate=self.yaml_config['training']['learning_rate'],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim=self.yaml_config['training']['optim'],
            weight_decay=self.yaml_config['training']['weight_decay'],
            lr_scheduler_type=self.yaml_config['training']['lr_scheduler_type'],
            seed=self.yaml_config['training']['seed'],
            output_dir=self.yaml_config['training']['output_dir'],
            ddp_find_unused_parameters=self.yaml_config['training']['ddp_find_unused_parameters'],
            report_to="wandb"
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.yaml_config['model']['max_seq_length'],
            dataset_num_proc=self.yaml_config['dataset']['num_proc'],
            packing=self.yaml_config['dataset']['packing'],
            args=training_args,
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory}GB.")
        print(f"{start_gpu_memory}GB of memory reserved.")

        trainer_stats = trainer.train()
        return trainer_stats

# Example usage
if __name__ == "__main__":
    from model_save import ModelManager

    model_config_path = "model_config/config_llama3_openko_8b.yaml"
    trainer = ModelTrainer(model_config_path)
    trainer.train_model()

    print("Training complete. Pushing model to Hugging Face Hub.")

    base_model_name = "beomi/Llama-3-Open-Ko-8B"
    finetuned_model_name = "Llama3-Open-Ko-8B-Instruct-toeic4all"
    quantization_method = 'q8_0'    # "f16", "q8_0", "q4_k_m", "q5_k_m"
    
    manager = ModelManager()
    manager.load_model_from_checkpoint()
    manager.push_model_to_hub(
        base_model_name,
        finetuned_model_name
        )
    manager.push_model_to_hub_gguf(
        finetuned_model_name,
        quantization_method=quantization_method
        )
```

## 6\. 모델 저장 및 푸시

### 모델 체크포인트 저장 및 불러오기

모델 체크포인트를 저장하고 불러오는 코드입니다.

**model\_**[**save.py**](http://save.py)

```python
import os
from unsloth import FastLanguageModel
from config import load_config

class ModelManager:
    def __init__(self, config_path='config.yaml', output_dir='outputs'):
        self.config = load_config(config_path)
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None

    def _get_latest_checkpoint(self):
        checkpoints = [
            int(x.split('-')[-1]) for x in os.listdir(self.output_dir) if 'checkpoint' in x
        ]
        if not checkpoints:
            raise ValueError("No checkpoints found in the output directory.")
        latest_checkpoint = max(checkpoints)
        return f"{self.output_dir}/checkpoint

-{latest_checkpoint}"

    def load_model_from_checkpoint(self, max_seq_length=4096, dtype=None, load_in_4bit=True):
        checkpoint_dir = self._get_latest_checkpoint()
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_dir,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit
        )
        return self.model, self.tokenizer

    def push_model_to_hub(self, base_model, huggingface_repo, save_method="merged_16bit"):
        hugginface_token = self.config['HUB_TOKEN']
        self.model.push_to_hub_merged(
            huggingface_repo,
            self.tokenizer,
            save_method=save_method,
            token=hugginface_token,
        )

    def push_model_to_hub_gguf(self, huggingface_repo, quantization_method='q8_0'):
        hugginface_token = self.config['HUB_TOKEN']
        self.model.push_to_hub_gguf(
            huggingface_repo + "-gguf",
            self.tokenizer,
            quantization_method=quantization_method,
            token=hugginface_token,
        )

# Example usage
if __name__ == "__main__":
    manager = ModelManager()
    manager.load_model_from_checkpoint()
    manager.push_model_to_hub("beomi/Llama-3-Open-Ko-8B", "Llama3-Open-Ko-8B-Instruct-ruolee")
    manager.push_model_to_hub_gguf("Llama3-Open-Ko-8B-Instruct-ruolee", quantization_method='q8_0')
```

## 7\. 모델 테스트

### 모델 테스트 코드 및 결과

모델을 테스트하는 코드와 그 결과입니다.

[**test.py**](http://test.py)

```python
import torch
from unsloth import FastLanguageModel
from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

from prompts import alpaca_prompt

# 체크포인트 디렉토리 설정
checkpoint_dir = 'outputs/checkpoint-1000'

# 체크포인트에서 모델과 토크나이저 불러오기
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_dir,
    max_seq_length=4096,
    dtype=None,  # 체크포인트 저장 시 사용된 데이터 타입 설정
    load_in_4bit=True  # 체크포인트 저장 시 사용된 설정
)


class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return torch.any(input_ids == self.stop_token_id)


stop_token = ""
stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)[0]

stopping_criteria = StoppingCriteriaList(
    [StopOnToken(stop_token_id)]
)


question_level = 3
question_type = {
    "name_eng": "Vocabulary-focused questions",
    "description": "이 유형의 문제는 단어의 의미, 사용 방법, 문맥 이해 등에 대한 이해를 물어봅니다. 각 세부 유형은 특정 단어 유형(예: 명사, 동사, 형용사 등)에 초점을 맞춥니다. 오답 선택지에는 같은 품사이면서 의미가 유사하지만 문맥에 맞지 않는 단어가 포함될 수 있습니다."
}
question_subtype = {
    "name_eng": "Adjectives",
    "description": "이 유형의 문제는 형용사의 적절한 선택과 사용을 물어봅니다. 형용사는 명사나 대명사를 수식하여 그것의 성질, 상태, 양 등을 나타냅니다."
}

FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction=f"""Generate a quiz of TOEIC Part 5 complying with the following quiz level(1 ~ 5) and question type, subtype.
                question_level: {question_level}, 
                question_type: {question_type['name_eng']} ({question_type['description']}),
                question_subtype: {question_subtype['name_eng']} ({question_subtype['description']})
                "You must follow the json format given.""",
            input="""You must follow the json format given.
                question_text: The question text of the quiz.
                choices: The list of 4 choices."
                correct_answer: The correct answer of the quiz."
                translation: The translation of the question text in Korean."
                explanation: The explanation of the quiz in Korean."
                vocabularies: The list of json objects of vocabularies in the question text and choices. The attributes are as follows."
                   word: The word of the vocabulary."
                   translation: The translation of the vocabulary."
                   difficulty: The difficulty of the vocabulary."
                   explanation: The explanation of the vocabulary."
                   part_of_speech: The part of speech of the vocabulary."
                   example: The example of the vocabulary."
                   example_translation: The translation of the example in Korean.""",
            response=""
        )
    ],
    return_tensors="pt",
).to("cuda")

if __name__ == "__main__":

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=4096,
        stopping_criteria=stopping_criteria
    )
```

* test 결과
    

```python
{
    'question_text': "The manager praised his team's ------ performance during the last quarter.",
    'choices': ['outstanding', 'outstandingly', 'outstandinglyness', 'outstandingnesses'],
    'correct_answer': 'outstanding',
    'translation': '매니저는 지난 분기 팀의 ------ 성과를 칭찬했습니다.',
    'explanation': "여기서 필요한 단어는 팀의 성과를 설명하는 형용사이며, 그 결과가 탁월하다는 것을 나타내는 단어가 필요합니다. 'outstanding'은 이런 상황에서 적절한 형용사입니다.",
    'vocabularies': [
        {
            'word': 'outstanding',
            'translation': '탁월한, 뛰어난',
            'difficulty': 3,
            'explanation': "'Outstanding'은 매우 높이 평가되는, 탁월한 등을 의미하는 형용사입니다. 보통 사람의 성과나 능력 등을 표현할 때 사용합니다.",
            'part_of_speech': 'adjective',
            'example': "It's an outstanding performance from the new player.",
            'example_translation': '신예 선수의 뛰어난 경기다.'
        },
        {
            'word': 'outstanding',
            'translation': '탁월한, 뛰어난',
            'difficulty': 3,
            'explanation': "'Outstanding'은 매우 높이 평가되는, 탁월한 등을 의미하는 형용사입니다. 보통 사람의 성과나 능력 등을 표현할 때 사용합니다.",
            'part_of_speech': 'adjective',
            'example': "It's an outstanding performance from the new player.",
            'example_translation': '신예 선수의 뛰어난 경기다.'
        },
        {
            'word': 'outstanding',
            'translation': '탁월한, 뛰어난',
            'difficulty': 3,
            'explanation': "'Outstanding'은 매우 높이 평가되는, 탁월한 등을 의미하는 형용사입니다. 보통 사람의 성과나 능력 등을 표현할 때 사용합니다.",
            'part_of_speech': 'adjective',
            'example': "It's an outstanding performance from the new player.",
            'example_translation': '신예 선수의 뛰어난 경기다.'
        },
        {
            'word': 'outstanding',
            'translation': '탁월한, 뛰어난',
            'difficulty': 3,
            'explanation': "'Outstanding'은 매우 높이 평가되는, 탁월한 등을 의미하는 형용사입니다. 보통 사람의 성과나 능력 등을 표현할 때 사용합니다.",
            'part_of_speech': 'adjective',
            'example': "It's an outstanding performance from the new player.",
            'example_translation': '신예 선수의 뛰어난 경기다.'
        }
    ]
}
```

## 8\. 결론

### 훈련 셋팅 환경

**모델 설정:**

* 모델 이름: `beomi/Llama-3-Open-Ko-8B`
    
* 최대 시퀀스 길이: 4096
    

**데이터셋 설정:**

* 레포 이름: `comsa33/toeic_p5_qa_pair`
    
* 프로세스 수: 8
    
* 패킹: False
    

**토크나이저 설정:**

* 패딩 방향: 오른쪽
    

**훈련 설정:**

* 디바이스 당 배치 크기: 1
    
* 그래디언트 누적 스텝: 8
    
* 워밍업 스텝: 20
    
* 학습 에포크: 40
    
* 최대 스텝: 5000
    
* 로깅 스텝: 50
    
* 저장 스텝: 500
    
* 학습률: 2.0e-4
    
* 옵티마이저: `adamw_8bit`
    
* 가중치 감소: 0.01
    
* 학습률 스케줄러 타입: `cosine`
    
* 시드: 123
    
* 출력 디렉토리: `outputs`
    
* DDP 사용 안함: `False`
    

**WandB 설정:**

* 프로젝트 이름: `toeic4all`
    

### 훈련 로그 주요 부분

| Step | Loss | Grad Norm | Learning Rate | Epoch |
| --- | --- | --- | --- | --- |
| 50 | 0.1556 | 0.3741 | 0.00019998209226697376 | 0.09 |
| 100 | 0.1252 | 0.3912 | 0.00019987267934654538 | 0.17 |
| 150 | 0.1264 | 0.3798 | 0.00019966391096058346 | 0.26 |
| 200 | 0.1297 | 0.3491 | 0.0001993559947963185 | 0.35 |
| 250 | 0.1346 | 0.3719 | 0.00019894923717529955 | 0.43 |
| 300 | 0.1340 | 0.3471 | 0.0001984440427486591 | 0.52 |
| 350 | 0.1361 | 0.3393 | 0.00019784091409455728 | 0.60 |
| 400 | 0.1381 | 0.4605 | 0.00019714045121820676 | 0.69 |
| 450 | 0.1465 | 0.3611 | 0.00019634335095497458 | 0.78 |
| 500 | 0.1444 | 0.3527 | 0.0001954504062771555 | 0.86 |
| 550 | 0.1472 | 0.7729 | 0.0001944625055051065 | 0.95 |
| 600 | 0.1243 | 0.2959 | 0.00019338063142352644 | 1.04 |
| 650 | 0.1000 | 0.2884 | 0.00019220586030376134 | 1.12 |
| 700 | 0.1020 | 0.3257 | 0.00019093936083310653 | 1.21 |
| 1400 | 0.0832 | 0.2512 | 0.00014890339920698334 | 2.94 |
| 1450 | 0.0739 | 0.3069 | 0.00014612822464534059 | 3.02 |
| 1500 | 0.0557 | 0.2003 | 0.00014330716074475286 | 3.11 |
| 1550 | 0.0553 | 0.2280 | 0.0001404430139595877 | 3.20 |
| 1600 | 0.0582 | 0.2564 | 0.00013753863360398241 | 3.28 |
| 1650 | 0.0368 | 0.2047 | 0.00010693369307880816 | 4.15 |
| 1700 | 0.0224 | 0.1552 | 6.957866508871068e-05 | 5.18 |
| 1750 | 0.0140 | 0.1082 | 3.653019855400123e-05 | 6.22 |
| 1800 | 0.0137 | 0.0991 | 2.312590237161335e-05 | 6.74 |

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1717256170063/1d840d61-9410-488e-b4d2-627033cf2c93.png align="center")

### 인사이트

1. **Loss 감소**: 훈련이 진행됨에 따라 Loss가 지속적으로 감소하는 것을 볼 수 있습니다. 이는 모델이 점차적으로 학습 데이터에 더 잘 맞아가고 있다는 것을 나타냅니다.
    
2. **Gradient Norm**: Gradient Norm은 전반적으로 일정한 범위 내에서 변동하지만, 중간중간 급격히 증가하는 경우가 있습니다. 이는 학습 과정에서 모델이 더 큰 업데이트를 필요로 하는 순간이 있었음을 의미할 수 있습니다.
    
3. **Learning Rate 감소**: 학습률이 점차적으로 감소하고 있습니다. 이는 Cosine 학습률 스케줄러가 적용된 결과로, 학습이 진행됨에 따라 학습률을 감소시켜 안정적인 최적화를 돕고 있습니다.
    
4. **Epoch 진행**: 각 로그는 epoch 진행 상황을 나타내며, 초기 단계에서의 빠른 개선이 보입니다. Epoch 1을 넘어가면서 Loss가 더욱 낮아지고 있습니다.
    

이러한 로그를 통해 모델이 안정적으로 학습되고 있음을 확인할 수 있습니다. 특히 Loss가 지속적으로 감소하는 것은 모델 성능이 개선되고 있음을 의미합니다. Grad Norm의 급격한 변화는 학습 과정 중 특정 시점에서의 모델 업데이트 필요성을 반영하며, 이는 추가적인 검토가 필요할 수 있는 부분입니다.

### 프로젝트 결과 및 성과

이 프로젝트를 통해 도메인 최적화 파인튜닝을 성공적으로 수행하여 TOEIC 문제 생성 모델을 개발하였습니다. 모델은 [https://toeic4all.com](https://toeic4all.com) 서비스에서 사용되고 있으며, 사용자의 학습 효율을 높이는데 기여하고 있습니다.

### 향후 개선 방향

* 더 많은 데이터셋을 활용한 추가 학습
    
* 사용자 피드백을 반영한 모델 개선
    
* 모델 경량화 및 최적화
    

이로써 AI 기반 TOEIC 문제 생성 모델의 개발 및 배포 과정에 대해 알아보았습니다. 앞으로도 지속적인 개선과 발전을 통해 더 나은 성능을 제공할 수 있도록 노력하겠습니다.