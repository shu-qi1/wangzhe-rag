import os
from flask import Flask, request, jsonify, render_template, Response, send_from_directory

# ========== 自动构建向量库 ==========
if not os.path.exists("./chroma_db"):
    print("首次运行，正在构建向量库...")
    from tools_build_vector_db import build_vector_db
    build_vector_db()
    print("向量库构建完成")

from step3_qa import rag_chain

app = Flask(__name__, template_folder=".", static_folder=".")

# 保存对话历史
chat_history = []

@app.route("/style.css")
def style():
    return send_from_directory(".", "style.css")

@app.route("/script.js")
def script():
    return send_from_directory(".", "script.js")

@app.route("/background.png")
def background():
    return send_from_directory(".", "background.png")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

@app.route("/ask", methods=["POST"])
def ask():
    global chat_history
    data = request.get_json()
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"answer": "请输入问题"})

    if chat_history:
        last_q, last_a = chat_history[-1]
        enhanced_question = f"上一轮用户问：{last_q}\n上一轮回答：{last_a}\n\n现在用户说：{question}\n请结合上一轮的上下文回答。"
    else:
        enhanced_question = question

    def generate():
        full_answer = ""
        for chunk in rag_chain.stream(enhanced_question):
            full_answer += chunk
            yield chunk
        chat_history.append((question, full_answer))
        if len(chat_history) > 3:
            chat_history.pop(0)

    return Response(generate(), mimetype="text/plain")

@app.route("/reset", methods=["POST"])
def reset():
    global chat_history
    chat_history = []
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)