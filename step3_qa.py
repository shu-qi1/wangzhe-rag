import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ========== 1. 加载文档并分块 ==========
docs_folder = "docs"
all_documents = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(docs_folder, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        # 把文件名作为标题，加到每个文档内容开头
        for doc in docs:
            doc.page_content = f"【攻略标题：{filename}】\n{doc.page_content}"
        all_documents.extend(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = splitter.split_documents(all_documents)
print(f"分块后共 {len(chunks)} 个块")

# ========== 2. 连接向量数据库 ==========
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY", "343c5a1e09dc45ab82b154df0a583129.QjGtXIauW1IkVhzA"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ========== 3. 混合检索 ==========
# 向量检索器 k=5
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# BM25 k=5
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# 权重偏向 BM25
retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.2, 0.8]
)


# ========== 4. 配置大模型 ==========
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPU_API_KEY", "343c5a1e09dc45ab82b154df0a583129.QjGtXIauW1IkVhzA"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0
)

# ========== 5. Prompt ==========
prompt = ChatPromptTemplate.from_template("""
你是一个王者万象棋的攻略助手。请根据下面提供的攻略内容回答用户的问题。

重要：请先核对检索到的攻略是否和用户问的阵容一致。
如果用户问的是"三分吕布流"，而检索到的是"逐鹿嬴政流"，就说"攻略里没有找到相关内容"，不要用其他阵容的内容回答。

如果攻略里没有相关内容，就说"攻略里没有找到相关内容"，不要瞎编。

攻略内容：
{context}

用户问题：{question}

请用简洁清晰的中文回答，并标注答案来自哪篇攻略。
""")

# ========== 6. 组装 RAG 链 ==========
def format_docs(docs):
    return "\n\n".join([f"【来源：{d.metadata.get('source', '未知')}】\n{d.page_content}" for d in docs])

# ========== 阵容名列表 ==========
LINEUP_KEYWORDS = [
    "海陆空", "三分倒转流", "三分吕布流", "逐鹿嬴政流",
    "河洛花木兰流", "日落海整备流", "养猪流", "李信牺牲流"
]

def extract_lineup(question):
    """从问题里提取阵容名"""
    for lineup in LINEUP_KEYWORDS:
        if lineup in question:
            return lineup
    return None

def get_relevant_docs(question):
    """如果匹配到阵容名，只从对应攻略里检索；否则走原检索"""
    lineup = extract_lineup(question)

    if lineup:
        # 只从对应攻略里检索
        filtered_chunks = [c for c in chunks if lineup in c.metadata.get('source', '')]
        if filtered_chunks:
            temp_bm25 = BM25Retriever.from_documents(filtered_chunks)
            temp_bm25.k = 3
            return temp_bm25.invoke(question)

    # 没匹配到，走原来的混合检索
    return retriever.invoke(question)

rag_chain = (
    {"context": lambda q: format_docs(get_relevant_docs(q)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# ========== 7. 测试 ==========
if __name__ == "__main__":
    question = "三分倒转流阵容的核心卡是谁？"
    print(f"问题：{question}\n")
    answer = rag_chain.invoke(question)
    print(f"回答：{answer}")