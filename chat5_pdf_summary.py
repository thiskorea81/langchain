from PyPDF2 import PdfFileReader
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
load_dotenv()

def process_text(text):
    #chracterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text=text)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    documents = FAISS.from_texts(chunks, embedding=embeddings)
    return documents

def main():
    st.title("pdf 요약하기")
    st.divider()
    
    pdf = st.file_uploader('pdf파일을 업로드해주세요', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        documents = process_text(text=text)
        query = "업로드된 PDF 파일의 내용을 약 3-5문장으로 요약해주세요."

        if query:
            docs = documents.similarity_search(query=query)
            llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
            chain = load_qa_chain(llm=llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader('--요약 결과--:')
            st.write(response)

if __name__ == '__main__':
    main()