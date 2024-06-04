from dotenv import load_dotenv

# Load API KEY 
load_dotenv()

from langchain_openai import ChatOpenAI

gptModel="gpt-4o"

llm=ChatOpenAI(
    temperature=0,
    model_name=gptModel
)

prompt = "진희는 강아지를 키우고 있습니다. 진희가 키우고 있는 동물은?"
print(llm.invoke(prompt))