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
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    do_sample=False,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

# num_beams 的设置范围
# 小规模任务：
# 范围：1 到 10
# 说明：num_beams=1 相当于贪心搜索，适合计算资源有限或对生成速度要求高的场景。num_beams 在 2 到 10 之间时，生成质量通常会有明显提升。

# 中等规模任务：
#   范围：10 到 20
# 说明：适用于大多数文本生成任务，能在生成质量和计算开销之间取得较好平衡。

# 大规模任务：
# 范围：20 到 50 或更高

# 其他建议
# 初始设置：建议从 num_beams=5 开始，逐步调整。

# 结合其他参数：num_beams 可以与 length_penalty、early_stopping 等参数配合使用，进一步优化生成效果。

# 性能权衡：增加 num_beams 会提升生成质量，但也会增加计算开销，需根据具体任务需求权衡。

#参数设置策略
# 1. 核心参数
# (1) num_beams
# 作用：控制每次保留的候选序列数量。

# 建议：
# 小任务：5-10
# 中等任务：10-20
# 大任务：20-50

# 注意：值越大，生成质量通常越高，但计算开销也越大。

# (2) max_length
# 作用：生成序列的最大长度。

# 建议：根据任务需求设置，避免过长或过短。

# 例如，文本摘要任务可以设置为 50-100，机器翻译任务可以设置为 100-150。

# (3) min_length
# 作用：生成序列的最小长度。

# 建议：避免生成过短的结果，例如在摘要任务中可以设置为 10-20。

# (4) length_penalty
# 作用：控制生成序列长度的偏好。

# length_penalty > 1：鼓励生成长序列。

# length_penalty < 1：鼓励生成短序列。

# length_penalty = 1：无偏好。

# 建议：通常设置为 0.6 到 2.0 之间，默认值为 1.0。

# (5) early_stopping
# 作用：是否在生成满足条件的序列时提前停止。

# 建议：

# 如果希望生成固定数量的序列，设置为 False。

# 如果希望尽早停止以节省计算资源，设置为 True。

# 2. 多样性控制参数
# (1) no_repeat_ngram_size
# 作用：避免生成重复的 n-gram。

# 建议：通常设置为 2 或 3，避免生成重复内容。

# (2) top_k 和 top_p
# 作用：结合 Beam Search 时，可以通过 top_k 或 top_p 进一步控制候选词的多样性。

# 建议：

# top_k：通常设置为 50-100。

# top_p：通常设置为 0.9-0.95。

# 3. 其他参数
# (1) temperature
# 作用：控制生成结果的随机性。

# temperature < 1：更确定性的输出。

# temperature > 1：更多样化的输出。

# 建议：通常设置为 0.7-1.0。

# (2) num_return_sequences
# 作用：返回的最终序列数量。

# 建议：根据需求设置，通常设置为 1-5。

# (3) do_sample
# 作用：是否使用采样策略。

# 建议：

# 如果使用 Beam Search，通常设置为 False。

# 如果希望结合采样策略，可以设置为 True。