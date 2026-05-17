(function () {
  const script = document.currentScript;
  const baseUrl = new URL(".", script ? script.src : window.location.href).origin;
  const iframeUrl = `${baseUrl}/chatbot.html`;

  if (document.getElementById("gutenberg-rag-chatbot-frame")) {
    return;
  }

  const button = document.createElement("button");
  button.id = "gutenberg-rag-chatbot-button";
  button.textContent = "RAG Chat";
  button.style.position = "fixed";
  button.style.right = "18px";
  button.style.bottom = "18px";
  button.style.zIndex = "2147483647";
  button.style.border = "0";
  button.style.borderRadius = "8px";
  button.style.padding = "12px 15px";
  button.style.background = "#116658";
  button.style.color = "white";
  button.style.font = "650 14px Inter, system-ui, -apple-system, Segoe UI, Arial, sans-serif";
  button.style.cursor = "pointer";
  button.style.boxShadow = "0 12px 34px rgba(17,102,88,.28)";

  const frame = document.createElement("iframe");
  frame.id = "gutenberg-rag-chatbot-frame";
  frame.src = iframeUrl;
  frame.title = "Gutenberg RAG Chatbot";
  frame.style.position = "fixed";
  frame.style.right = "18px";
  frame.style.bottom = "70px";
  frame.style.width = "400px";
  frame.style.height = "600px";
  frame.style.maxWidth = "calc(100vw - 36px)";
  frame.style.maxHeight = "calc(100vh - 92px)";
  frame.style.border = "1px solid #d9e0dd";
  frame.style.borderRadius = "12px";
  frame.style.background = "white";
  frame.style.boxShadow = "0 18px 46px rgba(22,32,29,.2)";
  frame.style.zIndex = "2147483647";
  frame.style.display = "none";

  button.addEventListener("click", () => {
    frame.style.display = frame.style.display === "none" ? "block" : "none";
  });

  document.body.appendChild(frame);
  document.body.appendChild(button);
})();
