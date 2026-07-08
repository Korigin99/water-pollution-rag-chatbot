"""
Water Quality Pollution Emergency Response Manual - RAG Chatbot (Streamlit UI)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pdfplumber
import streamlit as st
import torch
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ==============================================================================
# [CONFIG]
# ==============================================================================

PDF_PATH = "./대규모 환경(수질)오염 위기대응 실무매뉴얼.pdf"
FAISS_INDEX_DIR = "./faiss_index"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:14b"
EMBEDDING_MODEL = "upskyy/bge-m3-korean"
USE_GPU = True
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
RETRIEVER_K = 12

# ==============================================================================
# [PDF LOADER] pdfplumber - better table/layout extraction than pypdf
# ==============================================================================

def load_pdf_pdfplumber(pdf_path: str):
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for table in page.extract_tables():
                for row in table:
                    row_text = " | ".join(cell.strip() if cell else "" for cell in row)
                    if row_text.strip():
                        text += "\n" + row_text
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page": i, "source": pdf_path}
                ))
    return docs


# ==============================================================================
# [RAG CORE] - cached so it only loads once per session
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_rag_chain():
    # GPU
    device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64 if device == "cuda" else 16},
    )

    # FAISS index
    if os.path.exists(FAISS_INDEX_DIR):
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    else:
        if not os.path.exists(PDF_PATH):
            st.error(f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
            st.stop()

        documents = load_pdf_pdfplumber(PDF_PATH)

        separators = [
            "\n\n", "\n", "。", ". ", "! ", "? ",
            "• ", "- ", "▶", "◆", "■", "□", " ", "",
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        vectorstore.save_local(FAISS_INDEX_DIR)

    # LLM
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.1,
        num_predict=1024,
        num_ctx=8192,
    )

    # Prompt
    prompt = PromptTemplate(
        template="""당신은 환경부 수질오염 위기대응 실무매뉴얼에만 근거하여 답변하는 AI입니다.
반드시 아래 규칙을 따르세요:
1. 오직 [참고 문서]에 있는 내용만 사용하여 답변하세요.
2. [참고 문서]에 없는 내용은 절대로 추측하거나 보완하지 마세요.
3. [참고 문서]에 답이 없으면 반드시 "해당 내용은 매뉴얼에서 찾을 수 없습니다."라고만 답하고 추가 설명을 하지 마세요.
4. 답변은 한국어로 작성하세요.
5. 매뉴얼에 명시된 표현을 그대로 사용하고, 임의로 해석하거나 바꾸지 마세요.
6. 답변의 각 항목 끝에 근거가 된 페이지 번호를 (p.OO) 형식으로 표기하세요.

[참고 문서]
{context}

[질문]
{question}

[답변]
""",
        input_variables=["context", "question"],
    )

    # Chain
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain, device


# ==============================================================================
# [UI]
# ==============================================================================

st.set_page_config(
    page_title="수질오염 위기대응 챗봇",
    page_icon="💧",
    layout="wide",
)

st.title("💧 수질오염 위기대응 매뉴얼 챗봇")
st.caption("대규모 환경(수질)오염 위기대응 실무매뉴얼 기반 AI 어시스턴트")

# RAG 로딩 (최초 1회)
with st.spinner("AI 모델을 준비하고 있습니다..."):
    chain, device = load_rag_chain()

st.success(f"준비 완료 | 모델: {OLLAMA_MODEL} | 임베딩: {'GPU' if device == 'cuda' else 'CPU'}")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"참조 문서 ({len(msg['sources'])}건)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}] p.{src['page']}**")
                    st.caption(src["snippet"])

# 입력창
if query := st.chat_input("수질오염 관련 질문을 입력하세요..."):
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # AI 응답
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            result = chain.invoke({"query": query})

        answer = result["result"]
        sources = [
            {
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:150].replace("\n", " "),
            }
            for doc in result["source_documents"]
        ]

        st.markdown(answer)
        with st.expander(f"참조 문서 ({len(sources)}건)"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**[{i}] p.{src['page']}**")
                st.caption(src["snippet"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
