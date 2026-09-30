async function askQuestion() {
    const input = document.getElementById("question");
    const question = input.value.trim();
    if (!question) return;

    const history = document.getElementById("chat-history");
    history.innerHTML += `<div style="text-align:right; margin:8px 0; color:#1a3a8f;"><b>你：</b>${question}</div>`;
    input.value = "";

    const aiDiv = document.createElement("div");
    aiDiv.style.cssText = "text-align:left; margin:8px 0; color:#333;";
    aiDiv.innerHTML = "<b>AI：</b>";
    history.appendChild(aiDiv);
    history.scrollTop = history.scrollHeight;

    const response = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: question})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        const text = decoder.decode(value, {stream: true});
        aiDiv.innerHTML += text;
        history.scrollTop = history.scrollHeight;
    }
}

document.getElementById("question").addEventListener("keypress", function(e) {
    if (e.key === "Enter") askQuestion();
});

async function resetChat() {
    await fetch("/reset", {method: "POST"});
    document.getElementById("chat-history").innerHTML = `
        <div style="text-align:left; margin:8px 0; color:#333;">
            <b>AI：</b>你好！我是王者万象棋攻略助手。你可以问我：<br>
            • 三分倒转流阵容怎么运营？<br>
            • 新手推荐什么阵容？<br>
            • 曹操在阵容里是什么作用？<br>
            输入问题后按回车或点“发送”就行。
        </div>
    `;
}