import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from peft import PeftModel
from tqdm import *

model_name = "/cpfs01/user/xujiahao1/HuatuoGPT-o1-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

lora_path = "/mnt/workspace/xujiahao/Benchmark/MedQA/outputs/GRPO_Huatuo/checkpoint-1500"
lora_model = PeftModel.from_pretrained(model, lora_path)
model = lora_model.merge_and_unload()


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
        
    return {"messages": texts,
            "answers": labels,}



dataset = load_dataset("/mnt/workspace/xujiahao/data/MedQA")["test"]
dataset= dataset.select(range(5))
dataset = dataset.map(formatting_prompts, batched=True, remove_columns=dataset.column_names)
# print(dataset)
# print(dataset[0])
# print(dataset["answers"])

# messages = [
#     {"role": "system", "content": SYSTEM_PROMPT},
#     {"role": "user", "content": QUESTION_PROMPT.format(dataset[0]['sent1'], dataset[0]['ending0'], dataset[0]['ending1'], dataset[0]['ending2'], dataset[0]['ending3'])}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512
# )

# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def extract_solution(solution_str: str) -> str:
    processed_str = solution_str
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return ""
        
    final_answer = matches[-1].group(1).strip()
    return final_answer


# def evaluate_model(example):
#     model_inputs = tokenizer(example["messages"], return_tensors="pt").to(model.device)
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=1024)
#     output = pipe(model_inputs)[0]["generated_text"]
#     predicted = extract_solution(output)
#     example["is_correct"] = (predicted == example["answers"])
#     return example

# # 运行评估
# dataset = dataset.map(evaluate_model, batched=False)
# accuracy = sum(dataset["is_correct"]) / len(dataset)
# print(f"Accuracy: {accuracy * 100:.2f}%")


num = 0
for i in tqdm(range(len(dataset))):
    text = dataset[i]["messages"]
    label = dataset[i]["answers"]
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    predicted = extract_solution(response)
    print("##########################################################")
    print(f"Response: {response}")
    # print(f"Predicted: {predicted}")
    print(f"Label: {label}")

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

    match = re.search(r'\(([A-Z])\)', label)
    if match:
        label = match.group(1)
    else:
        label = ""
    print(predicted, label)
    if predicted == label:
        print("Correct")
        num += 1
    else:
        print("Wrong")
    print(f"Accuracy: {num / (i+1) * 100:.2f}%")

print(f"Accuracy: {num / len(dataset)* 100:.2f}%")