import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Load environment variables from .env file
load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))

# Load the embedding model
embedding = HuggingFaceEmbeddings()

# Load the Llama-3.3-70B model from Groq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


def process_document_to_chroma_db(file_name):
    # Load the PDF document using PyPDFLoader (no OCR / CV deps)
    loader = PyPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    # Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    # Store the document chunks in a Chroma vector database
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0


def answer_question(user_question):
    # Load the persistent Chroma vector database
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    # Create a retriever for document search
    retriever = vectordb.as_retriever()

    # Build a retrieval-augmented chain without langchain.chains
    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question. "
        "If you do not know the answer, say you do not know.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(user_question)

    return answer
