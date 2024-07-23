# 실행방법: 터미널에서 다음 명령어를 입력한다.
# streamlit run .\chat5_chat.py

import streamlit as st
from langchain_community.chat_models import ChatOpenAI

# Load API KEY
# .env 파일에 아래 줄을 입력하고 저장
# OPENAI_API_KEY=your_openai_api_key
# .env를 불러오기 위해 from dotenv import load_dotenv
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="뭐든지 질문하세요")
st.title("뭐든지 질문하세요")

def generate_response(input_text):  # llm이 답변 생성 함수
    llm = ChatOpenAI(temperature = 0,
                     model_name = 'gpt-4o',
                     )
    st.info(llm.predict(input_text))

with st.form('Question'):
    text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?')
    submitted = st.form_submit_button('보내기')
    generate_response(text)