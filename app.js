// Utility to escape HTML characters
function escapeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  // Minimal markdown parser:
  // 1. Extract triple-backtick code blocks (replace with placeholders).
  // 2. Replace newlines with <br>.
  // 3. Process inline code (wrapped in single backticks).
  // 4. Reinsert code blocks.
  function parseMarkdown(text) {
    const codeBlocks = [];
    text = text.replace(/```([\s\S]*?)```/g, (match, code) => {
      const placeholder = `@@CODEBLOCK${codeBlocks.length}@@`;
      codeBlocks.push('<pre class="code-snippet"><code>' + escapeHTML(code) + '</code></pre>');
      return placeholder;
    });
    
    // Replace newlines with <br>
    text = text.replace(/\n/g, '<br>');
    
    // Process inline code (wrapped in single backticks)
    text = text.replace(/`([^`]+?)`/g, (match, code) => '<code>' + escapeHTML(code) + '</code>');
    
    // Reinsert code blocks
    codeBlocks.forEach((block, i) => {
      text = text.replace(`@@CODEBLOCK${i}@@`, block);
    });
    
    return text;
  }
  
  // --- DOM references ---
  const chatBox = document.getElementById('chat-box');
  const userInput = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const fileInput = document.getElementById('file-input');
  const fileLabel = document.getElementById('file-label');
  const toggleModeBtn = document.getElementById('toggle-mode-btn');
  
  let conversationHistory = [];
  let diagramMode = false; 
  
  let conversationID = localStorage.getItem("conversationID") || generateConversationID();
  function generateConversationID() {
    const id = 'conv-' + Date.now() + '-' + Math.floor(Math.random() * 1000);
    localStorage.setItem("conversationID", id);
    return id;
  }
  
  function saveConversation() {
    localStorage.setItem("chatHistory1", JSON.stringify(conversationHistory));
  }
  
  function loadConversation() {
    const savedHistory = localStorage.getItem("chatHistory1");
    if (savedHistory) {
      conversationHistory = JSON.parse(savedHistory);
      conversationHistory.forEach(msg => appendMessage(msg.text, msg.sender));
    }
  }
  
  // Append a message to the chat after processing markdown
  function appendMessage(text, sender) {
    text = parseMarkdown(text);
    const messageDiv = document.createElement('div');
    messageDiv.className = sender === 'file' ? 'file-message' : 'message ' + sender;
    messageDiv.innerHTML = text;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
  
  function addMessage(text, sender) {
    conversationHistory.push({ sender, text });
    appendMessage(text, sender);
    saveConversation();
  }
  
  function updateModeUI() {
    if (diagramMode) {
      toggleModeBtn.textContent = "Switch to Chat Mode";
      fileLabel.style.display = "none";
    } else {
      toggleModeBtn.textContent = "Switch to Diagram Mode";
      fileLabel.style.display = "inline";
    }
  }
  updateModeUI();
  loadConversation();
  
  async function sendMessage() {
    const text = userInput.value.trim();
    const file = fileInput.files[0];
  
    if (text) {
      addMessage(text, "user");
    }
  
    if (diagramMode) {
      if (!text) return;
      try {
        const formData = new FormData();
        formData.append('instruction', text);
        const response = await fetch('http://localhost:8000/diagram_generate', {
          method: 'POST',
          body: formData
        });
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const linkHTML = `<a href="${url}" download="diagram.png"><img src="${url}" style="max-width:100%; border:1px solid #ccc;"/></a>`;
        addMessage("This is the diagram you asked for. Click the image to download it:<br/>" + linkHTML, "bot");
      } catch (error) {
        addMessage("Error: Unable to generate diagram.", "bot");
        console.error("Error:", error);
      }
    } else {
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        if (text) formData.append('instruction', text);
        try {
          const response = await fetch('http://localhost:8000/upload', {
            method: 'POST',
            body: formData
          });
          const data = await response.json();
          const previewHTML = `<strong>File: ${file.name}</strong><br/><pre>${data.result}</pre>`;
          addMessage(previewHTML, "file");
          fileInput.value = "";
        } catch (error) {
          addMessage("Error: Unable to process file.", "bot");
          console.error("Error:", error);
        }
      } else {
        if (!text) return;
        try {
          const payload = {
            conversation_id: conversationID,
            question: text
          };
          const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await response.json();
          addMessage(data.answer, "bot");
        } catch (error) {
          addMessage("Error: Unable to fetch response.", "bot");
          console.error("Error:", error);
        }
      }
    }
    userInput.value = "";
  }
  
  toggleModeBtn.addEventListener('click', () => {
    diagramMode = !diagramMode;
    updateModeUI();
  });
  
  sendBtn.addEventListener('click', sendMessage);
  userInput.addEventListener('keypress', event => {
    if (event.key === 'Enter') sendMessage();
  });
  