from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import shutil

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

vectorstore = None
chain = None


def build_chain(vs):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template="""Use only the context below to answer.
If answer not in context say 'Not found in document.'

Chat History: {chat_history}

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question", "chat_history"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chat_history = []

    def rag_with_memory(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        history_text = "\n".join(chat_history[-6:])

        rag_chain = prompt | llm | StrOutputParser()
        answer = rag_chain.invoke({
            "context": context,
            "question": question,
            "chat_history": history_text
        })

        chat_history.append(f"Human: {question}")
        chat_history.append(f"AI: {answer}")

        return answer

    return rag_with_memory


@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running!"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, chain

    try:
        file_path = f"uploaded_{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        chain = build_chain(vectorstore)
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

    if chain is None:
        return {"error": "Please upload a PDF first"}

    try:
        answer = chain(req.question)
        return {
            "question": req.question,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/ask-stream")
async def ask_stream(req: QuestionRequest):
    global vectorstore


    if vectorstore is None:
        return {"error":"Please Upload a PDF first"}


    async def generate():
        try:
            # Get relevant chunks
            retriever = vectorstore.as_retriever(
                search_kwargs={"k":3}
            )
            docs = retriever.invoke(req.question)
            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                streaming=True
            )

            prompt = f"""Use only the context below tp answer.
            If answer not in context say 'Not found in document.'
            Context = {context}

            Question: {req.question}

            Answer:"""

            # Stream tokens one by one

            async for chunk in llm.astream(prompt):
                token = chunk.content
                if token:
                    yield token

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }

    )


