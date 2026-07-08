from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./대규모 환경(수질)오염 위기대응 실무매뉴얼.pdf")
docs = loader.load()

with open("check_oil_result.txt", "w", encoding="utf-8") as f:
    for doc in docs:
        content = doc.page_content
        page = doc.metadata["page"] + 1
        if "유류" in content and ("오일" in content or "응급" in content or "차단" in content):
            f.write(f"=== p.{page} ===\n")
            f.write(content[:1500])
            f.write("\n\n")

print("check_oil_result.txt 저장 완료")
