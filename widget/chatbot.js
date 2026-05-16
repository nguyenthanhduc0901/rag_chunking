(function () {
  const script = document.currentScript;
  const baseUrl = new URL(".", script ? script.src : window.location.href).origin;
  const iframeUrl = `${baseUrl}/chatbot.html`;

  if (document.getElementById("gutenberg-rag-chatbot-frame")) {
    return;
  }

  const button = document.createElement("button");
  button.id = "gutenberg-rag-chatbot-button";
  button.textContent = "Chat";
  button.style.position = "fixed";
  button.style.right = "18px";
  button.style.bottom = "18px";
  button.style.zIndex = "2147483647";
  button.style.border = "0";
  button.style.borderRadius = "999px";
  button.style.padding = "11px 16px";
  button.style.background = "#176b5c";
  button.style.color = "white";
  button.style.font = "14px Arial, Helvetica, sans-serif";
  button.style.cursor = "pointer";
  button.style.boxShadow = "0 8px 26px rgba(0,0,0,.22)";

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
  frame.style.border = "1px solid #dadce0";
  frame.style.borderRadius = "10px";
  frame.style.background = "white";
  frame.style.boxShadow = "0 14px 40px rgba(0,0,0,.24)";
  frame.style.zIndex = "2147483647";
  frame.style.display = "none";

  button.addEventListener("click", () => {
    frame.style.display = frame.style.display === "none" ? "block" : "none";
  });

  document.body.appendChild(frame);
  document.body.appendChild(button);
})();
