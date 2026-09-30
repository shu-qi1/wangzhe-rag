# 王者万象棋 RAG 攻略助手

基于 RAG（检索增强生成）的王者万象棋攻略问答系统。用户输入问题，系统从攻略库中检索相关内容，由大模型生成回答，并标注来源。

## 功能

- 自然语言提问，如"海陆空怎么用""三分倒转流核心卡是谁"
- 混合检索（向量检索 + BM25 关键词检索）
- 阵容名过滤，避免相近阵容搞混
- 多轮对话记忆
- 流式输出，回答逐字显示
- 标注答案来源

## 技术栈

Python、LangChain、智谱 GLM-4-Flash、Chroma、Flask、HTML/CSS/JS

## 项目结构

- `server.py`：Flask 后端
- `step3_qa.py`：RAG 核心逻辑（检索 + 生成）
- `tools_load_docs.py`：加载文档并分块
- `tools_build_vector_db.py`：向量化并存入 Chroma
- `docs/`：攻略文档
- `static/`：前端样式和脚本
- `templates/`：HTML 页面

## 运行方式

1. 安装依赖

pip install langchain langchain-community langchain-openai langchain-classic langchain-text-splitters chromadb flask rank_bm25

2. 构建向量数据库

python tools_build_vector_db.py

3. 启动服务

python server.py

4. 浏览器打开 http://127.0.0.1:5000

## 使用示例

问：海陆空怎么用？

答：系统从"海陆空攻略"中检索相关内容，生成回答并标注来源。

## 说明

- 向量数据库（`chroma_db/`）未上传，首次运行需先执行 `tools_build_vector_db.py` 生成
- 支持阵容名过滤：问"海陆空"时，只从"海陆空攻略"里检索，避免相近阵容混淆
- 需要配置智谱 API Key，在代码中填入或设置环境变量 `ZHIPU_API_KEY`