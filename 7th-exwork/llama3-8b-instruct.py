# llama3-8b-instruct.py

# 安装依赖 Terminal 中执行
# pip install --upgrade modelscope requests urllib3 tqdm pandas mindspore mindnlp
# pip uninstall mindformers
# Modelarts中可以不执行此句
#!apt update > /dev/null; apt install aria2 git-lfs axel -y > /dev/null

import mindspore
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "LLM-Research/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, mirror='modelscope')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    ms_dtype=mindspore.float16,
    mirror='modelscope'
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="ms"
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=50,
    eos_token_id=terminators,
    # do_sample=False,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

import pandas as pd
test = pd.read_csv('./test_input.csv', header=None)

from tqdm import tqdm
import os


i = 1
# 假设 test 是一个 DataFrame
# 遍历测试数据集的第一项的值，目的是生成与给定句子最相关的五个成语
for test_prompt in tqdm(test[0].values, total=len(test[0].values), desc="处理进度"):
    i = i + 1
    # 构造提示信息，要求模型输出与句子最相关的五个成语
    prompt = f"列举与下面句子最符合的五个成语。只需要输出五个成语，不需要有其他的输出，写在一行中：{test_prompt}"

    # 初始化一个长度为5的列表，填充默认成语“同舟共济”
    words = ['同舟共济'] * 5

    # 构建聊天消息格式，用于提示模型进行生成
    messages = [
    {"role": "system", "content": "You are a helpful chinese teacher."},
    {"role": "user", "content": f"{prompt}"},
    ]
    # 应用聊天模板对消息进行处理，准备模型输入
    input_ids = tokenizer.apply_chat_template(
           messages,
           add_generation_prompt=True,
           return_tensors="ms"
    )
    # 对输入文本进行编码，准备模型输入数据
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # 生成回答，限制最大生成长度
    outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    eos_token_id=terminators,
    no_repeat_ngram_size=2,
    num_beams=5,
    num_return_sequences=5,
    do_sample=False,
    remove_invalid_values=True,
    #temperature=0.6,
    #top_p=0.9,
    #top_k=50,
    #length_penalty=1.0,
    )
    # 提取模型输出，去除输入部分
    response = outputs[0][input_ids.shape[-1]:]
    
    # 解码模型输出，去除特殊标记
    response = tokenizer.decode(response, skip_special_tokens=True)
    
    # 清理回答文本，确保格式统一
    response = response.replace('\n', ' ').replace('、', ' ')
    # 提取回答中的成语，确保每个成语长度为4且非空
    words = [x for x in response.split() if len(x) == 4 and x.strip() != '']
    
    

    # 如果生成的成语列表长度不满足要求（即20个字符），则使用默认成语列表
   #if len(' '.join(words).strip()) != 24:
       # words = ['同舟共济'] * 5
    while True:
        text = ' '.join(words).strip()
        if len(text) < 24:
            words.append('同舟共济')
        else:
            break

    # 将最终的成语列表写入提交文件
    with open('submit.csv', 'a+', encoding='utf-8') as up:
        up.write(' '.join(words) + '\n')

    
    # 查看阶段性结果
    if i % 50 == 0:
        tqdm.write(f"大模型第{i}次返回的结果是：\n   {response}\n")
        tqdm.write(f"submit.cvs第{i}行输出结果：\n   {words}\n")
    
    # 完整的循环数为2973，如果想要测试，可以设置为10
    if i == 2973:
        break

print('submit.csv 已生成')