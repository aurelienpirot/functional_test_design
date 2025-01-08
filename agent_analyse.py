from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

def get_documents_from_pdf(url):
    loader = PyPDFLoader(url)
    docs = loader.load()

    spittler = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = spittler.split_documents(docs)
   
    return splitDocs

def create_db (docs):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorStore

def create_chain(vectorStore):
    #Instantiate Model
    model = ChatOpenAI(
        temperature=0.4,
        model = "gpt-3.5-turbo-1106",
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}
    """)

    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k":1})
    
    retriever_chain = create_retrieval_chain(
        retriever, 
        chain
    )

    return retriever_chain

docs = get_documents_from_pdf('Base_documentaires\\document1.pdf')
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
    "input": "What is XXXX?",
    })

#print(response)
print(response["context"])
print(response["answer"])
