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

# ========== 1. 从文件名自动提取阵容名 ==========
def get_lineup_names():
    """从攻略文件名自动提取阵容名"""
    lineups = []
    for filename in os.listdir("."):
        if filename.endswith(".txt") and "攻略" in filename:
            name = filename.replace(".txt", "").replace("攻略", "")
            name = name.split("（")[0].strip()
            if name:
                lineups.append(name)
    return lineups

LINEUP_KEYWORDS = get_lineup_names()
print(f"自动识别到的阵容：{LINEUP_KEYWORDS}")

# ========== 2. 加载文档并分块 ==========
all_documents = []

for filename in os.listdir("."):
    if filename.endswith(".txt"):
        loader = TextLoader(filename, encoding="utf-8")
        docs = loader.load()
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

# ========== 3. 连接向量数据库 ==========
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY", "343c5a1e09dc45ab82b154df0a583129.QjGtXIauW1IkVhzA"),
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ========== 4. 混合检索 ==========
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 6

retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.2, 0.8]
)

# ========== 5. 阵容过滤检索 ==========
def extract_lineup(question):
    # 先精确匹配
    for lineup in LINEUP_KEYWORDS:
        if lineup in question:
            return lineup
    # 再模糊匹配：去掉"流"字后匹配
    for lineup in LINEUP_KEYWORDS:
        short = lineup.replace("流", "")
        if short in question:
            return lineup
    return None

def get_relevant_docs(question):
    lineup = extract_lineup(question)
    if lineup:
        filtered_chunks = [c for c in chunks if lineup in c.metadata.get('source', '')]
        print(f"匹配到阵容：{lineup}，找到 {len(filtered_chunks)} 个块")
        if filtered_chunks:
            return filtered_chunks
    print("没匹配到阵容，走混合检索")
    return retriever.invoke(question)

# ========== 6. 配置大模型 ==========
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPU_API_KEY", "343c5a1e09dc45ab82b154df0a583129.QjGtXIauW1IkVhzA"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
    temperature=0
)

# ========== 7. Prompt ==========
prompt = ChatPromptTemplate.from_template("""
你是一个王者万象棋的攻略助手。请根据下面提供的攻略内容回答用户的问题。

要求：
1. 先核对检索到的攻略是否和用户问的阵容一致。如果不一致，就说"攻略里没有找到相关内容"。
2. 如果攻略里有具体的回合数、棋手、装备、英雄名称，必须完整列出，不要笼统概括。
3. 用户问"怎么运营"时，必须按前期、中期、后期分段回答，每一段都要写清楚第几回合、做什么、找什么牌。
4. 用户问"需要什么牌"时，必须列出所有核心牌和备选牌。
5. 用户问"棋手是谁"时，必须明确说出棋手名字。
6. 如果攻略里没有相关内容，就说"攻略里没有找到相关内容"，不要瞎编。

攻略内容：
{context}

用户问题：{question}

请用清晰的中文回答，并标注答案来自哪篇攻略。
""")

# ========== 8. 组装 RAG 链 ==========
def format_docs(docs):
    return "\n\n".join([f"【来源：{d.metadata.get('source', '未知')}】\n{d.page_content}" for d in docs])

rag_chain = (
    {"context": lambda q: format_docs(get_relevant_docs(q)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)