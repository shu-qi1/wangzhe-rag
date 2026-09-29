import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ========== 1. 加载并分块（和上一步一样）==========
docs_folder = "docs"
all_documents = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(docs_folder, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        documents = loader.load()
        all_documents.extend(documents)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = splitter.split_documents(all_documents)
print(f"分块后共 {len(chunks)} 个块")

# ========== 2. 向量化 + 存入 Chroma ==========
# 用智谱的 Embedding API（兼容 OpenAI 格式）
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY", "343c5a1e09dc45ab82b154df0a583129.QjGtXIauW1IkVhzA"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# 存入向量数据库
batch_size = 64
vectorstore = None

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    print(f"正在处理第 {i//batch_size + 1} 批，共 {len(batch)} 条")

    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    else:
        vectorstore.add_documents(batch)

print("向量化完成，已存入 chroma_db 文件夹")
print(f"数据库中共有 {vectorstore._collection.count()} 个向量")
