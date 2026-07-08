from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./대규모 환경(수질)오염 위기대응 실무매뉴얼.pdf")
docs = loader.load()

with open("check_result.txt", "w", encoding="utf-8") as f:
    f.write(f"총 {len(docs)} 페이지\n\n")
    for doc in docs:
        content = doc.page_content
        page = doc.metadata["page"] + 1
        if "최초" in content and ("시간" in content or "이내" in content):
            f.write(f"=== p.{page} ===\n")
            f.write(content[:800])
            f.write("\n\n")

print("check_result.txt 저장 완료")
