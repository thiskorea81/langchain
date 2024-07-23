# streamlit run c:/Users/USER/Documents/git_thiskorea81/langchain/chat5_rag_chatbot.py

import streamlit as st 
from PyPDF2 import PdfReader
# from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings # 없어질 예정
from langchain_community.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
# from langchain.vectorstores import FAISS # 없어질 예정
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader # 없어질 예정
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF 문서에서 텍스트를 추출
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# 지정된 조건에 따라 주어진 텍스트를 더 작은 덩어리로 분할
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\\n", 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len)
    docs = text_splitter.split_text(text)
    return docs

# 주어진 텍스트 청크에 대한 임베딩을 생성하고 파이스(FAISS)를 사용하여 벡터 저장소를 생성
def get_vectorstore(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Load API KEY
# .env 파일에 아래 줄을 입력하고 저장
# OPENAI_API_KEY=your_openai_api_key
# .env를 불러오기 위해 from dotenv import load_dotenv
from dotenv import load_dotenv
import os
load_dotenv()

def get_conversation_chain(vectorstore):
    # ConversationBufferWindowMemory에 이전 대화 저장
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True)
    # ConversationRetrievalChain을 통해 랭체인 챗봇에 쿼리 전송
    model_name = "gpt-4o"
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, 
                       model_name=model_name),
        retriever = vectorstore.as_retriever(),
        get_chat_history = lambda h: h,
        memory=memory
    )
    return conversation_chain

user_uploads = st.file_uploader("파일을 업로드해주세요", accept_multiple_files=True)
if user_uploads is not None:
    if st.button("Upload"):
        with st.spinner("처리중.."):
            raw_text = get_pdf_text(user_uploads)  # PDF 텍스트 가져오기
            text_chunks = get_text_chunks(raw_text) # 텍스트에서 청크 검색
            vectorstore = get_vectorstore(text_chunks) # 파이스 벡터 저장소 만들기
            st.session_state.conversation = get_conversation_chain(vectorstore=vectorstore)  # 대화 체인 만들기

if user_query := st.chat_input("질문을 입력해주세요"):
    if 'conversation' in st.session_state:
        result = st.session_state.conversation({
            "question": user_query,
            "chat_history": st.session_state.get('chat_history', [])
        })
        response = result["answer"]
    else:
        response = "먼저 문서를 업로드해주세요"
    with st.chat_message("assistant"):
        st.write(response)