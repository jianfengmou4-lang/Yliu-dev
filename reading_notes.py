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

# ===== 第1步：提取核心观点 =====
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个读书笔记助手。从用户的输入中提取3个核心观点，每点用一句话概括。"),
    ("user", "{text}"),
])
extract_chain = extract_prompt | llm | StrOutputParser()

# ===== 第2步：生成笔记 =====
note_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据核心观点，生成3条读书笔记。每条笔记包含：观点、感悟、行动建议。以JSON格式输出。"),
    ("user", "核心观点：\n{points}"),
])
note_chain = note_prompt | llm | JsonOutputParser()

# ===== 第3步：翻译成英文 =====
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "将以下中文笔记翻译成英文，保留原意。"),
    ("user", "{notes}"),
])
translate_chain = translate_prompt | llm | StrOutputParser()

# ===== 测试文本 =====
article = """
《原子习惯》这本书的核心思想是：习惯的养成不靠意志力，而靠系统。
每天进步1%，一年后你会进步37倍。
不要关注目标，要关注体系。
习惯的四个定律：让它明显、让它有吸引力、让它简单、让它令人满足。
"""

print("=== 第1步：提取核心观点 ===")
points = extract_chain.invoke({"text": article})
print(points)

print("\n=== 第2步：生成笔记（JSON） ===")
notes = note_chain.invoke({"points": points})
import json
print(json.dumps(notes, ensure_ascii=False, indent=2))

print("\n=== 第3步：翻译成英文 ===")
en_notes = translate_chain.invoke({"notes": str(notes)})
print(en_notes)

print("\n=== 一个调用完成 ===")
