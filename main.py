import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# ===== 方式一：原生 OpenAI SDK =====
from openai import OpenAI

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

response1 = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好，用一句话介绍你自己"}],
)
print("=== 原生 SDK ===")
print(response1.choices[0].message.content)
print()

# ===== 方式二：LangChain ChatOpenAI =====
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
)


print("=== LangChain ===")

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{lang}翻译助手，翻译时保留原文的语气"),
    ("user", "{text}"),
])

def StrOutputParser():
    def parse(result):
        return result.content
    return parse

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"lang": "英中", "text": "Hello"})  # 直接返回字符串
print(result)

# response2 = llm.invoke("你好，用一句话介绍你自己")

# print(response2.content)
print()

# ===== 对比三处不同 =====
print("=== 三处不同 ===")
print("1. 初始化: OpenAI(api_key=..., base_url=...)  →  ChatOpenAI(api_key=..., base_url=...)")
print("2. 调用: client.chat.completions.create(messages=[...])  →  llm.invoke(...)")
print("3. 取值: response.choices[0].message.content  →  response.content")
