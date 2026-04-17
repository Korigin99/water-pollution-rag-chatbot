"""
RAG System - Water Quality Pollution Emergency Response Manual
Windows + RTX 5090 + Ollama (Llama 3.1 8b)
"""

# ==============================================================================
# [CONFIG] Modify these variables for your environment
# ==============================================================================

PDF_PATH = "./\ub300\uaddc\ubaa8 \ud658\uacbd(\uc218\uc9c8)\uc624\uc5fc \uc704\uae30\ub300\uc751 \uc2e4\ubb34\ub9e4\ub274\uc5bc.pdf"

FAISS_INDEX_DIR = "./faiss_index"

# Ollama server URL
#   - Running on this machine : "http://127.0.0.1:11434"
#   - Connecting from outside : "http://192.168.0.111:11434"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

OLLAMA_MODEL = "llama3.1:8b"

# Korean-optimized embedding model
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"

USE_GPU = True

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 5

# ==============================================================================
# [MAIN CODE]
# ==============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def check_gpu():
    if USE_GPU and torch.cuda.is_available():
        device = "cuda"
        print(f"[GPU] {torch.cuda.get_device_name(0)} - CUDA {torch.version.cuda}")
    else:
        device = "cpu"
        if USE_GPU:
            print("[CPU] GPU not available - check torch CUDA installation")
            print(f"      torch.version.cuda = {torch.version.cuda}")
        else:
            print("[CPU] USE_GPU=False")
    return device


def load_pdf(pdf_path: str):
    print(f"[PDF] Loading: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[PDF] {len(documents)} pages loaded")
    return documents


def split_documents(documents):
    separators = [
        "\n\n", "\n", "\u3002", ". ", "! ", "? ",
        "\u2022 ", "- ", "\u25b6", "\u25c6", "\u25a0", "\u25a1", " ", "",
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"[SPLIT] {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vectorstore(chunks, device: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64 if device == "cuda" else 16,
        },
    )

    if os.path.exists(FAISS_INDEX_DIR):
        print(f"[FAISS] Loading existing index: {FAISS_INDEX_DIR}")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        print(f"[FAISS] Building new index ({len(chunks)} chunks)...")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        vectorstore.save_local(FAISS_INDEX_DIR)
        print(f"[FAISS] Index saved: {FAISS_INDEX_DIR}")

    return vectorstore


def build_rag_chain(vectorstore):
    prompt = PromptTemplate(
        template="""당신은 환경부 수질오염 위기대응 전문가 AI 어시스턴트입니다.
아래 참고 문서를 바탕으로 질문에 정확하고 구체적으로 답변하세요.
참고 문서에 없는 내용은 "해당 내용은 매뉴얼에 포함되어 있지 않습니다."라고 답하세요.

[참고 문서]
{context}

[질문]
{question}

[답변]
""",
        input_variables=["context", "question"],
    )

    print(f"[Ollama] Connecting: {OLLAMA_BASE_URL} / {OLLAMA_MODEL}")
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_predict=1024,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_K * 3},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    print("[RAG] Chain ready")
    return chain


def print_answer(result: dict):
    print("\n" + "=" * 60)
    print(result["result"])
    print("-" * 60)
    for i, doc in enumerate(result["source_documents"], 1):
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i}] p.{page} | {snippet}...")
    print("=" * 60 + "\n")


def main():
    print("=" * 60)
    print("  \uc218\uc9c8\uc624\uc5fc \uc704\uae30\ub300\uc751 \ub9e4\ub274\uc5bc RAG \uc2dc\uc2a4\ud15c")
    print("=" * 60)

    device = check_gpu()
    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks, device)
    chain = build_rag_chain(vectorstore)

    print("\n[READY] Type your question. Exit: q")
    while True:
        query = input("\n질문 > ").strip()
        if not query:
            continue
        if query.lower() in ("q", "quit", "exit", "\uc885\ub8cc"):
            print("Goodbye.")
            break
        result = chain.invoke({"query": query})
        print_answer(result)


if __name__ == "__main__":
    main()
