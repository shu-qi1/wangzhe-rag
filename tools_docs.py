from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# 1. 读取 docs 文件夹里所有 TXT 文件
docs_folder = "docs"
all_documents = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(docs_folder, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        documents = loader.load()
        all_documents.extend(documents)
        print(f"已加载: {filename}")

print(f"\n总共加载了 {len(all_documents)} 个文档")

# 2. 分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # 每块最多 300 个字符
    chunk_overlap=50,     # 相邻块之间重叠 50 个字符
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

chunks = splitter.split_documents(all_documents)
print(f"分块后共 {len(chunks)} 个块")

# 3. 打印前 3 个块看看
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- 第 {i+1} 块 ---")
    print(chunk.page_content)
    print(f"来源: {chunk.metadata.get('source', '未知')}")