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
    do_sample=False,
    num_beams=10,
    num_beam_groups=5,
    diversity_penalty=1.0
    #no_repeat_ngram_size=2,
    #num_return_sequences=5,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

# 参数设置
# 1. 核心参数
# (1) num_beams
# 作用：控制每组中保留的候选序列数量。

# 建议：

# 通常设置为 num_beam_groups 的倍数。

# 例如，如果 num_beam_groups=5，则 num_beams 可以设置为 10、15 或 20。

# (2) num_beam_groups
# 作用：将 Beam Search 分成多个组，每组独立生成候选序列。

# 建议：

# 通常设置为 2-5。

# 较大的值会增加多样性，但也会增加计算开销。

# 2. 参数设置建议
# (1) 小规模任务
# num_beams：10-20

# num_beam_groups：2-3

# 说明：适合计算资源有限或对生成速度要求高的场景。

# (2) 中等规模任务
# num_beams：20-30

# num_beam_groups：3-4

# 说明：适合大多数文本生成任务，能在生成质量和多样性之间取得较好平衡。

# (3) 大规模任务
# num_beams：30-50

# num_beam_groups：4-5

# 说明：适合对生成质量和多样性要求极高的任务，如故事生成或创意写作。

# 3. 参数关系
# num_beams 必须是 num_beam_groups 的倍数：

# 例如，num_beam_groups=3 时，num_beams 可以设置为 6、9、12 等。

# 每组中的 Beam 数量：

# 每组中的 Beam 数量为 num_beams / num_beam_groups。

# 例如，num_beams=12 且 num_beam_groups=3 时，每组有 4 个 Beam。

# 4. 其他参数
# (1) diversity_penalty
# 作用：控制组间多样性惩罚的强度。

# 较大的值会增加组间多样性。

# 较小的值会减少组间多样性。

# 建议：通常设置为 1.0-2.0。

# (2) max_length 和 min_length
# 作用：控制生成序列的最大和最小长度。

# 建议：根据任务需求设置，例如：

# 文本摘要任务：max_length=50-100，min_length=10-20。

# 机器翻译任务：max_length=100-150，min_length=20-30。

# (1) 机器翻译
# 特点：输入和输出的长度通常接近。

# max_length：

# 设置为输入长度的 1.2-1.5 倍。

#   例如，输入长度为 50，则 max_length 可以设置为 60-75。

# min_length：

# 设置为输入长度的 0.8 倍。

# 例如，输入长度为 50，则 min_length 可以设置为 40。

# (2) 文本摘要
# 特点：输出长度通常远小于输入长度。

# max_length：

# 设置为输入长度的 0.3-0.5 倍。

# 例如，输入长度为 500，则 max_length 可以设置为 150-250。

# min_length：

# 设置为输入长度的 0.1 倍。

# 例如，输入长度为 500，则 min_length 可以设置为 50。

# (3) 对话生成
# 特点：输出长度通常较短，但需要上下文连贯。

# max_length：

# 设置为 50-100。

# min_length：

# 设置为 10-20。

# (4) 故事生成/创意写作
# 特点：输出长度较长，需要创造性和多样性。

# max_length：

# 设置为 100-200。

# min_length：

# 设置为 50-100。

# (5) 文本补全
# 特点：输出长度通常较短，补充输入内容。

# max_length：

# 设置为输入长度的 0.5-1 倍。

# 例如，输入长度为 50，则 max_length 可以设置为 25-50。

# min_length：

# 设置为输入长度的 0.2 倍。

# 例如，输入长度为 50，则 min_length 可以设置为 10。

# (6) 问答系统
# 特点：输出长度通常较短，直接回答问题。

# max_length：

# 设置为 20-50。

# min_length：

# 设置为 5-10。

# (3) num_return_sequences
# 作用：返回的最终序列数量。

# 建议：通常设置为 num_beam_groups 或更小。