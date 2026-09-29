import os
import gradio as gr
from step3_qa import rag_chain

# ========== 自定义 CSS ==========
custom_css = """
.gradio-container {
    background-image: url("/file=background.jpg") !important;
    background-size: cover;
    background-position: center;
}
/* 所有块级容器透明 */
.gradio-container .block, .gradio-container .form, .gradio-container .panel {
    background: transparent !important;
    border: none !important;
}
/* 标题 */
.main-title {
    text-align: center;
    color: #ffffff;
    font-size: 2.2em;
    font-weight: bold;
    margin-bottom: 0.2em;
    text-shadow: 0 0 20px rgba(0, 0, 0, 0.8);
}
.sub-title {
    text-align: center;
    color: #ffffff;
    font-size: 1em;
    margin-bottom: 1.5em;
    text-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
}
/* 卡片 */
.card {
    background: rgba(0, 0, 0, 0.5) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    backdrop-filter: blur(10px) !important;
}
/* 输入框和输出框 */
.answer-box textarea {
    background: rgba(0, 0, 0, 0.6) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 2px solid rgba(255, 255, 255, 0.2) !important;
    font-size: 15px !important;
    line-height: 1.8 !important;
    padding: 16px !important;
}
.answer-box label {
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: bold !important;
}
/* 提示区 */
.tips-box {
    background: rgba(0, 0, 0, 0.4) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-top: 12px !important;
}
.tips-box * {
    color: #ffffff !important;
    font-size: 14px !important;
    line-height: 2 !important;
}
/* 按钮 */
button.primary-btn {
    background: linear-gradient(90deg, #8a2be2, #4b0082) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    padding: 14px !important;
    margin-top: 12px !important;
}
button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 20px rgba(138, 43, 226, 0.8) !important;
}
"""

# ========== 处理函数 ==========
def answer_question(question):
    if not question.strip():
        return "请输入你的问题，比如：三分倒转流阵容的核心卡是谁？"
    result = rag_chain.invoke(question)
    return result

# ========== 界面 ==========
with gr.Blocks(css=custom_css, title="王者万象棋 攻略助手") as app:
    gr.HTML('<div class="main-title">王者万象棋 · 攻略助手</div>')
    gr.HTML('<div class="sub-title">基于 RAG 的智能问答系统 · 答案均来自收集的攻略</div>')

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                question_input = gr.Textbox(
                    label="你的问题",
                    placeholder="比如：三分倒转流阵容的核心卡是谁？",
                    lines=3,
                    elem_classes="answer-box"
                )
                submit_btn = gr.Button("提问", elem_classes="primary-btn")

                gr.Markdown("""
                **试试这些问题：**
                - 三分倒转流阵容的核心卡是谁？
                - 新手推荐什么阵容？
                - 曹操在阵容里是什么作用？
                """, elem_classes="tips-box")

        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                answer_output = gr.Textbox(
                    label="攻略助手回答",
                    lines=15,
                    elem_classes="answer-box"
                )

    submit_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)
    question_input.submit(fn=answer_question, inputs=question_input, outputs=answer_output)

if __name__ == "__main__":
    app.launch()