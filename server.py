import os
from flask import Flask, request, jsonify, render_template, Response, send_from_directory

# ========== 自动构建向量库 ==========
if not os.path.exists("./chroma_db"):
    print("首次运行，正在构建向量库...")
    from tools_build_vector_db import build_vector_db
    build_vector_db()
    print("向量库构建完成")

from step3_qa import rag_chain, LINEUP_KEYWORDS

app = Flask(__name__, template_folder=".", static_folder=".")

chat_history = []
current_lineup = None

def extract_lineup_from_text(text):
    for lineup in LINEUP_KEYWORDS:
        if lineup in text:
            return lineup
    for lineup in LINEUP_KEYWORDS:
        short = lineup.replace("流", "")
        if short in text:
            return lineup
    return None

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

@app.route("/ask", methods=["POST"])
def ask():
    global chat_history, current_lineup
    data = request.get_json()
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"answer": "请输入问题"})

    # 检查问题里有没有阵容名
    new_lineup = extract_lineup_from_text(question)
    if new_lineup:
        current_lineup = new_lineup

    # 如果当前有讨论的阵容，拼到问题前面
    if current_lineup:
        enhanced_question = f"【当前讨论阵容：{current_lineup}】{question}"
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
    global chat_history, current_lineup
    chat_history = []
    current_lineup = None
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)