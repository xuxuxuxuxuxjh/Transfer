from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
import torch
import wandb

model_name = "/nas/shared/Gvlab_A100/xujiahao/model/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")


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


# dataset = load_dataset("/nas/shared/Gvlab_A100/xujiahao/data/MedQA")["train"]
# dataset = dataset.select(range(10))
# dataset = dataset.map(formatting_prompts, batched=True, remove_columns=dataset.column_names)
# print(dataset[0])

predicted = """Predicted: The patient's symptoms—nocturia, difficulty initiating urination, and post-void dribbling—are classic for **benign prostatic hyperplasia (BPH)**. The enlarged, smooth prostate on digital rectal exam further supports this diagnosis. 

**Key considerations:**  
- **First-line treatment for BPH** includes alpha-blockers (e.g., tamsulosin), which relax smooth muscle in the prostate and bladder neck, improving urine flow.  
- **Oxybutynin (C)** is a bladder antimuscarinic, used for overactive bladder (e.g., urgency/frequency), but does not address BPH-related obstructive symptoms.  
- **Midodrine (B)** is for orthostatic hypotension, not BPH.  
- **Hydrochlorothiazide (A)** is a diuretic, possibly part of his hypertension regimen but irrelevant to his current urinary complaints.  

**Answer:**  
**(D) Tamsulosin** is indicated for BPH-related urinary obstruction.
Label: (D): Tamsulosin"""

lines = predicted.split('\n')
if lines:
    first_line = lines[0].strip()
    match = re.search(r'\(([A-Z])\)', first_line)
    if match:
        predicted = match.group(1)
    else:
        last_line = lines[-1].strip()
        match = re.search(r'\(([A-Z])\)', last_line)
        if match:
            predicted = match.group(1)
        else:
            predicted = ""
print(predicted)