import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth import is_bfloat16_supported
import torch
import wandb
from peft import PeftModel


max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/cpfs01/user/xujiahao1/HuatuoGPT-o1-7B",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

lora_path = "/mnt/workspace/xujiahao/Benchmark/MedQA/outputs/GRPO_Huatuo/checkpoint-500"
lora_model = PeftModel.from_pretrained(model, lora_path)
model = lora_model.merge_and_unload()


model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

import re
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
Now the user asks you to solve a medical reasoning problem.
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

QUESTION_PROMPT = """{}
(A): {}
(B): {}
(C): {}
(D): {}"""


EOS_TOKEN = tokenizer.eos_token
def formatting_prompts(examples):
    questions = examples["sent1"]
    choice_0 = examples["ending0"]
    choice_1 = examples["ending1"]
    choice_2 = examples["ending2"]
    choice_3 = examples["ending3"]
    answers = examples["label"]
    texts = []
    labels = []
    for i in range(len(questions)):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": QUESTION_PROMPT.format(questions[i], choice_0[i], choice_1[i], choice_2[i], choice_3[i])}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)

        if answers[i] == 0:
            labels.append("(A): " + choice_0[i])
        elif answers[i] == 1:
            labels.append("(B): " + choice_1[i])
        elif answers[i] == 2:
            labels.append("(C): " + choice_2[i])
        elif answers[i] == 3:
            labels.append("(D): " + choice_3[i])
        
    return {"prompt": texts,
            "answers": labels,}


dataset = load_dataset("/cpfs01/user/xujiahao1/MedQA")["train"]
# dataset = dataset.select(range(10))
dataset = dataset.map(formatting_prompts, batched=True, remove_columns=dataset.column_names)

from typing import Tuple, Optional
def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:

    processed_str = solution_str

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def validate_response_structure(processed_str: str) -> bool:

    print("\n[Structure Validation]")
    validation_passed = True
    if processed_str.startswith('<think>'):
        print("The string starts with '<think>'")
    else:
        validation_passed = False
        print("The string does not start with '<think>'")
        
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }
    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def reward_func_format(prompts, completions, answers, **kwargs):

    scores = []
    for i in range(len(completions)):
        prompt, completion, answer = prompts[i], completions[i], answers[i]

        answer_text, processed_str = extract_solution(completion)

        format_correct = validate_response_structure(processed_str)
        format_score = 1.0 if format_correct else -1.0

        scores.append(format_score)

    return scores

def reward_func_answer(prompts, completions, answers, **kwargs):

    scores = []
    for i in range(len(completions)):
        prompt, completion, answer = prompts[i], completions[i], answers[i]

        print("\n" + "="*80)
        print(" Processing New Sample ".center(80, '='))

        print(f"[Prompt]\n{prompt}")
        print(f"[Completion]\n{completion}")
        print(f"[Ground Truth]\n{answer}")
        print(f"[Length]\n{len(completion)}")

        answer_text, processed_str = extract_solution(completion)

        # Validate response structure
        format_correct = validate_response_structure(processed_str)
        format_score = 1.0 if format_correct else -1.0

        # Validate answer content
        answer_score = 0.0
        if format_correct and answer_text:
            print(f"\n[Content Validation]")
            print(f"Expected: {answer}")
            print(f"Predicted: {answer_text}")
            
            if answer == answer_text:
                answer_score = 2.0
                print("Content validation: FULL MATCH")
            else:
                match = re.search(r'\(([A-Z])\)', answer)
                if match:
                    answer = match.group(1)
                else:
                    answer = ""

                lines = answer_text.split('\n')
                if lines:
                    first_line = lines[0].strip()
                    match = re.search(r'\(([A-Z])\)', first_line)
                    if match:
                        answer_text = match.group(1)
                    else:
                        last_line = lines[-1].strip()
                        match = re.search(r'\(([A-Z])\)', last_line)
                        if match:
                            answer_text = match.group(1)
                        else:
                            answer_text = ""

                if answer == answer_text:
                    answer_score = 2.0
                    print("Content validation: CORRET")
                else:
                    answer_score = -2.0
                    print("Content validation: WRONG ANSWER")
        else:
            answer_score = -2
            print("\n[Content Validation] Skipped due to format errors or missing answer")

        total_score = format_score + answer_score
        scores.append(answer_score)
        print("\n" + "-"*80)
        print(f"Final Score ".center(80, '-'))
        print(f"Format: {format_score}")
        print(f"Answer: {answer_score}")
        print(f"Total: {total_score}")
        print("="*80 + "\n")

    return scores

def length_reward(completions, **kwargs):
    scores = []
    for i in range(len(completions)):
        completion = completions[i]
        answer_text, processed_str = extract_solution(completion)

        format_correct = validate_response_structure(processed_str)

        length_score = 0.0
        if format_correct and answer_text:
            if len(processed_str) > 2000:
                length_score += (len(processed_str) - 2000) / 2000
            if len(answer_text) > 200:
                length_score -= (len(answer_text) - 200) / 200
        scores.append(length_score)
    return scores

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count+=0.25
    if text.count("</think>") == 1:
        count+=0.25
    if text.count("<answer>") == 1:
        count+=0.25
    if text.count("</answer>") == 1:
        count+=0.25
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    scores = []
    for i in range(len(completions)):
        comletion = completions[i]
        scores.append(count_xml(comletion))
    return scores

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    # lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = 2048,
    max_completion_length = 4096,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 2000,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs/GRPO_Huatuo3",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        reward_func_format,
        reward_func_answer,
        xmlcount_reward_func,
        length_reward,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
trainer.save_model("/outputs/GRPO_Huatuo3")
model.save_lora("/outputs/GRPO_Huatuo3/lora")