# sft Qwen2-7B-Instruct
import deepspeed
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import datasets
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments

PaddingID = -100

prompt_template = '<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n'
system_prompt = """You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
Now the user asks you to solve a medical reasoning problem.
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""
answer_template = """
<think>
{}
</think>
<answer>
{}
</answer>
"""
def preprocess_inputs(examples, max_len=1024, overflow_strategy='truncate'):

    model_inputs = {'input_ids': [], 'labels': [], 'input_len': [], 'output_len': []}
    for i in range(len(examples['question'])):
        prompt = prompt_template.format(
                    system_prompt=system_prompt,
                    user_prompt=examples['question'][i]
                )
        a_ids = tokenizer.encode(prompt)
        b_ids = tokenizer.encode(answer_template.format(examples['reasoning (reasoning_content)'][i], examples['response (content)'][i]), add_special_tokens=False) + [tokenizer.eos_token_id]
        context_length = len(a_ids)
        input_ids = a_ids + b_ids

        if len(input_ids) > max_len and overflow_strategy == 'drop':
            input_ids = []
            labels = []
        else:
            if max_len > len(input_ids):
                pad_length = max_len - len(input_ids)
                labels = [PaddingID] * context_length + b_ids + [PaddingID] * pad_length
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
            else:
                labels = [PaddingID] * context_length + b_ids
                labels = labels[:max_len]
                input_ids = input_ids[:max_len]
        model_inputs['input_ids'].append(input_ids)
        model_inputs['labels'].append(labels)
        model_inputs['input_len'].append(len(a_ids))
        model_inputs['output_len'].append(len(b_ids))
    return model_inputs

if __name__=="__main__":

    model_path = "/cpfs01/user/xujiahao1/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    dataset_folder_path='/mnt/workspace/xujiahao/data/medical-r1-distill-data'
    raw_datasets = load_dataset(dataset_folder_path)['train']
    raw_datasets = raw_datasets.select(range(100))
    train_dataset = raw_datasets.map(preprocess_inputs, batched=True, num_proc=1, load_from_cache_file=False)

    gpu_type = 'A100'   
    assert gpu_type in ['A100','V100']
    if gpu_type=='A100':
        if_bf16=True
        data_type=torch.bfloat16
    if gpu_type=='V100':   
        if_bf16=False
        data_type=torch.float16   

    model = AutoModelForCausalLM.from_pretrained(model_path)

    model.gradient_checkpointing_enable()

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, 
                                        model=model, 
                                        label_pad_token_id=PaddingID, 
                                        pad_to_multiple_of=None, 
                                        padding=False)
    """
    创建了一个 Seq2Seq任务 的数据整理器, 用于将多个样本组合成一个批次。
    label_pad_token_id：指定用于填充标签的 padding token 的 id, 默认为-100
    pad_to_multiple_of = None：指定padding后序列长度应该是多少的倍数。如果设置为None（默认值），则不进行这种类型的padding。
    padding = False：指定是否对数据进行padding。设置为False 通常意味着数据的 padding 将在模型内部或通过其他方式处理。
    """

    # 训练参数
    args = TrainingArguments(
        output_dir='./outputs/SFT',             # 模型保存路径
        per_device_train_batch_size=4,      # 全局 batch_size，注意不是单个 GPU 上的  batch_size
        logging_steps=1,
        gradient_accumulation_steps=32,     # 梯度累计，在显存较小的设备中，每隔多个 batch_size 更新一次梯度；
                                            # 真正更新梯度的 batch = per_device_train_batch_size * gradient_accumulation_steps
                                            # 即 4*32=128 个 batch 更新一次梯度
        num_train_epochs=1,                 # sft llm 的 epoch 一般不需要太大，1～3轮即可
        weight_decay=0.003,                 # 权重衰减正则化，将一个与权重向量的L2范数成比例的惩罚项加到总损失中
        warmup_ratio=0.03,                  # 预热，在训练初期逐渐增加学习率，而不是从一开始就使用预设的最大学习率，避免一开始就使用过高的学习率可能导致的训练不稳定。
                                            # 如果设置 warmup_ratio=0.1，共有100个epochs，那么在前10个epochs（即前10%的训练时间），学习率会从0逐渐增加到最大值。
        optim='adamw_hf',
        lr_scheduler_type="cosine",         # 根据余弦函数的形状来逐渐减小学习率，一般有 "linear" 和 "cosine" 两种方式   
        learning_rate=1e-5,                 # 最大学习率
        save_strategy='steps',
        save_steps=10,                       # 保存模型的步骤，save_steps 是 per_device_train_batch_size * gradient_accumulation_steps，而不是 per_device_train_batch_size
        bf16=if_bf16,                       # 是否使用 bfloat16 数据格式
        run_name='qwen2.5-7B-sft',
        report_to='wandb',                  # 使用 wandb 打印日志
        deepspeed="/nas/shared/Gvlab_A100/xujiahao/Benchmark/MedQA/zero3.json",
        gradient_checkpointing=True,  # 启用梯度检查点
    )

    # train
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    deepspeed.init_distributed()
    trainer.train()

    # if trainer.is_world_process_zero():
    #     trainer.save_model("/outputs/SFT")

    question = "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?"
    prompt = prompt_template.format(
                    system_prompt=system_prompt,
                    user_prompt=question
                )
    
    model.eval()
    text = prompt

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
