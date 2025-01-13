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
    num_beams=1,
    do_sample=False,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

# 参数设置

# 1. 核心参数
# (1) max_length
# 作用：生成序列的最大长度。

# 建议：根据任务需求设置，避免过长或过短。

# 例如，文本摘要任务可以设置为 50-100，机器翻译任务可以设置为 100-150。

# (2) min_length
# 作用：生成序列的最小长度。

# 建议：避免生成过短的结果，例如在摘要任务中可以设置为 10-20。

# (3) num_return_sequences
# 作用：返回的最终序列数量。

# 建议：Greedy Search 每次只能生成一个最优序列，因此通常设置为 1。