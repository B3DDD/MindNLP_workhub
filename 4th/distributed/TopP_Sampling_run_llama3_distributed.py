import mindspore
from mindspore.communication import init
from mindnlp.transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "LLM-Research/Meta-Llama-3-8B-Instruct"

init()
tokenizer = AutoTokenizer.from_pretrained(model_id, mirror='modelscope')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    ms_dtype=mindspore.float16,
    mirror='modelscope',
    device_map="auto"
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
    max_new_tokens=100,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_k=50,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

# 参数设置
# 1. 核心参数
# (1) top_p
# 作用：限制候选词的范围，仅从累积概率超过 p 的词中采样。

# 建议：

# 通常设置为 0.9-0.95。

# 较小的值（如 0.8）会限制候选词范围，生成结果更确定。

# 较大的值（如 0.98）会增加候选词范围，生成结果更多样。

# (2) max_length
# 作用：生成序列的最大长度。

# 建议：根据任务需求设置，避免过长或过短。

# 例如，文本摘要任务可以设置为 50-100，对话生成任务可以设置为 100-150。

# (3) min_length
# 作用：生成序列的最小长度。

# 建议：避免生成过短的结果，例如在摘要任务中可以设置为 10-20。

# (4) num_return_sequences
# 作用：返回的最终序列数量。

# 建议：根据需求设置，通常设置为 1-5。

# 2. 多样性控制参数
# (1) temperature
# 作用：控制生成结果的随机性。

# temperature < 1：更确定性的输出。

# temperature > 1：更多样化的输出。

# temperature = 1：无偏好的原始概率分布。

# 建议：通常设置为 0.7-1.0，与 top_p 结合使用效果更好。

# (2) no_repeat_ngram_size
# 作用：避免生成重复的 n-gram。

# 建议：通常设置为 2 或 3，避免生成重复内容。

# 3. 其他参数
# (1) do_sample
# 作用：是否使用采样策略。

# 建议：Top-P Sampling 必须设置为 True。

# (2) early_stopping
# 作用：是否在生成满足条件的序列时提前停止。

# 建议：可以设置为 True，以节省计算资源。