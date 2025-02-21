from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

loader = PyPDFLoader("PycharmProjects/SarkariSeva/Aadhaar_Enrolment__and__Update__-__English.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="qwen-2.5-32b"
)

prompt_template = """
You are a helpful assistant. Answer the user's question based on the context below. Keep the answer simple and precise.

Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": prompt}
)

query = "Steps to make an Aadhaar card"
response = qa_chain.invoke({"query": query})
print(response["result"])

