import requests
import json

# ChatGPT API密钥
api_key = "YOUR_API_KEY"

# ChatGPT API地址
url = "https://api.openai.com/v1/engines/davinci-codex/completions"

# 输入的句子
input_text = "This is a long and complex sentence that needs to be simplified."

# 发送API请求
response = requests.post(
    url,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    },
    json={
        "prompt": f"Simplify this sentence: {input_text}\n",
        "temperature": 0.5,
        "max_tokens": 50,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
)

# 解析API响应
if response.status_code == 200:
    response_data = json.loads(response.content.decode("utf-8"))
    output_text = response_data["choices"][0]["text"].strip()
    print(f"Simplified sentence: {output_text}")
else:
    print(f"API error: {response.content}")
