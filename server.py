from flask import Flask, request, jsonify, render_template, Response
from step3_qa import rag_chain

app = Flask(__name__)

# 保存对话历史
chat_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global chat_history
    data = request.get_json()
    question = data.get("question", "")
    if not question.strip():
        return jsonify({"answer": "请输入问题"})

    # 如果有历史，把上一轮的问题和回答拼进问题里
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
        # 保存本轮对话
        chat_history.append((question, full_answer))
        # 只保留最近 3 轮
        if len(chat_history) > 3:
            chat_history.pop(0)

    return Response(generate(), mimetype="text/plain")

@app.route("/reset", methods=["POST"])
def reset():
    global chat_history
    chat_history = []
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)