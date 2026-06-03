from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import shutil

load_dotenv()

app = FastAPI()

# CORS — allows React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables — shared across all requests
vectorstore = None
chain = None

# ─── Helper function to build RAG chain ─────────────────

def build_chain(vs):
    prompt = PromptTemplate(
        template="""Use only the context below to answer.
If answer not in context say "This information is not available in the document."

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# ─── ROUTES ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running!"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, chain

    try:
        # Save uploaded file temporarily
        file_path = f"uploaded_{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Load PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

        # Store in ChromaDB
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        # Build RAG chain
        chain = build_chain(vectorstore)

        # Delete temp file — already stored in ChromaDB
        os.remove(file_path)

        return {
            "message": "PDF processed successfully!",
            "total_pages": len(pages),
            "total_chunks": len(chunks)
        }

    except Exception as e:
        return {"error": str(e)}


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    global chain

    # Check if PDF uploaded first
    if chain is None:
        return {"error": "Please upload a PDF first"}

    try:
        answer = chain.invoke(req.question)
        return {
            "question": req.question,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}