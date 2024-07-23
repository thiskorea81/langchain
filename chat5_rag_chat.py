# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
documents = TextLoader('./AI.txt').load()

# print(documents)

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

print(docs)

# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Chromadb에 벡터 저장
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings)

print(db)

# Load API KEY
# .env 파일에 아래 줄을 입력하고 저장
# OPENAI_API_KEY=your_openai_api_key
# .env를 불러오기 위해 from dotenv import load_dotenv
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_community.chat_models import ChatOpenAI
model_name = "gpt-4o"
llm = ChatOpenAI(model_name=model_name)

#QnA chain
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
print(chain)

query = "AI란?"
print(query)
matching_docs = db.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
print(answer)