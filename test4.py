from dotenv import load_dotenv

# Load API KEY 
load_dotenv()

from langchain_openai import ChatOpenAI

gptModel="gpt-4o"

llm=ChatOpenAI(
    temperature=0,
    model_name=gptModel
)

# langchain_core.prompts 모듈에서 PromptTemplate을 가져옵니다.
from langchain_core.prompts import PromptTemplate

# 사용할 템플릿 문자열을 정의합니다.
template = "{product}를 홍보하기 위한 좋은 문구를 추천해줘?"

# PromptTemplate 인스턴스를 생성합니다.
prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

# LCEL(LangChain Expression Language)
chain = prompt | llm
response = chain.invoke({"product": "카메라"})
print(response.content)