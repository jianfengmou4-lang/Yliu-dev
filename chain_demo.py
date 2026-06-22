import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# ===== 链1: 英→中翻译 =====
en_to_cn_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的英中翻译。只输出翻译结果，不要多余文字。"),
    ("user", "{text}"),
])
en_to_cn_chain = en_to_cn_prompt | llm | StrOutputParser()

# ===== 链2: 中→英翻译 =====
cn_to_en_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional Chinese-to-English translator. Output only the translation."),
    ("user", "{text}"),
])
cn_to_en_chain = cn_to_en_prompt | llm | StrOutputParser()

print("=== 链1: 英→中 ===")
result1 = en_to_cn_chain.invoke({"text": "LangChain makes it easy to build LLM applications."})
print(result1)

print("\n=== 链2: 中→英 ===")
result2 = cn_to_en_chain.invoke({"text": "管道符让代码更简洁"})
print(result2)

# ===== 链3: 情感分析 (JSON) =====
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "分析文本的情感，以JSON格式输出，包含 score(1-10) 和 reason 两个字段。"),
    ("user", "{text}"),
])
sentiment_chain = sentiment_prompt | llm | JsonOutputParser()

print("\n=== 链3: 情感分析 (JSON) ===")
result3 = sentiment_chain.invoke({"text": "等了两个小时还没修好，太失望了"})
print(result3)
print(f"情感分数: {result3['score']}")
print(f"分析理由: {result3['reason']}")

# ===== 链4: 两条链拼接 —— 先中→英翻译，再分析英文的情感 =====
print("\n=== 链4: 链式拼接（翻译 + 情感分析）===")
# 先翻译
english_text = cn_to_en_chain.invoke({"text": "今天天气真好，心情特别愉快"})
print(f"翻译结果: {english_text}")
# 再分析情感
sentiment_result = sentiment_chain.invoke({"text": english_text})
print(f"情感分析: {sentiment_result}")
