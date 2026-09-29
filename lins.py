from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key="343c5a1e09dc45ab82b154df0a583129.QjGtXIauW1IkVhzA",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

result = embeddings.embed_query("测试一下")
print(f"向量维度: {len(result)}")
print("API 连接成功")