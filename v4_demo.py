import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个擅长深度思考的助手，请仔细分析问题后再回答"),
    ("user", "{text}"),
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"text": "Python 的装饰器和生成器有什么关系？用类比的方式解释"})
print(result)
