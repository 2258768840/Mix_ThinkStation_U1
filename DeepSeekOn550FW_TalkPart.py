from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 设置本地模型路径
model_path = "/path/to/your/deepseek-r1-model"  # 替换为你的实际路径

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # 自动选择设备（GPU/CPU）
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True  # 如果模型需要自定义代码
)

# 将模型设置为评估模式
model.eval()

# 对话示例
def chat(prompt):
    # 编码输入
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt"
    ).to(model.device)
    
    # 生成响应
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # 解码输出
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response

# 进行对话
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat(user_input)
    print(f"Assistant: {response}")