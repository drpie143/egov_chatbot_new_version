function makeSessionId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const state = {
  sessionId: localStorage.getItem("egov_session_id") || makeSessionId(),
  lastRequestId: null,
};

localStorage.setItem("egov_session_id", state.sessionId);

const els = {
  healthBadge: document.getElementById("health-badge"),
  clearChat: document.getElementById("clear-chat"),
  searchForm: document.getElementById("search-form"),
  searchInput: document.getElementById("search-input"),
  searchResults: document.getElementById("search-results"),
  searchCount: document.getElementById("search-count"),
  refreshStats: document.getElementById("refresh-stats"),
  likesCount: document.getElementById("likes-count"),
  dislikesCount: document.getElementById("dislikes-count"),
  popularList: document.getElementById("popular-list"),
  messages: document.getElementById("messages"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  sendButton: document.getElementById("send-button"),
  sourceTemplate: document.getElementById("source-template"),
};

function renderIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdown(value) {
  const escaped = escapeHtml(value || "");
  const linked = escaped.replace(
    /(https?:\/\/[^\s<]+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>',
  );
  return `<p>${linked.replaceAll("\n", "<br>")}</p>`;
}

function appendMessage(role, content, options = {}) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const message = document.createElement("article");
  message.className = "message";
  message.innerHTML = renderMarkdown(content);
  row.appendChild(message);

  if (role === "assistant" && options.sources?.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "source-list";
    options.sources.forEach((source) => {
      const card = els.sourceTemplate.content.firstElementChild.cloneNode(true);
      card.href = source.url || "#";
      card.querySelector(".source-title").textContent = source.title || "Nguồn";
      card.querySelector(".source-meta").textContent = source.agency || source.url || "";
      sourceList.appendChild(card);
    });
    message.appendChild(sourceList);
  }

  if (role === "assistant" && options.requestId) {
    const actions = document.createElement("div");
    actions.className = "feedback-actions";
    actions.innerHTML = `
      <button type="button" data-rating="like"><i data-lucide="thumbs-up"></i><span>Like</span></button>
      <button type="button" data-rating="dislike"><i data-lucide="thumbs-down"></i><span>Dislike</span></button>
    `;
    actions.addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-rating]");
      if (!button) return;
      [...actions.querySelectorAll("button")].forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      await sendFeedback(button.dataset.rating, options.requestId);
    });
    message.appendChild(actions);
  }

  els.messages.appendChild(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  renderIcons();
  return row;
}

function renderWelcome() {
  els.messages.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.innerHTML = `
    <h2>Hỏi đáp thủ tục hành chính</h2>
    <p>Sẵn sàng tra cứu dữ liệu dịch vụ công.</p>
  `;
  els.messages.appendChild(empty);
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    els.healthBadge.textContent = data.status === "ok" ? "Sẵn sàng" : "Thiếu tài nguyên";
    els.healthBadge.className = `status-badge ${data.status === "ok" ? "ok" : "degraded"}`;
  } catch {
    els.healthBadge.textContent = "Mất kết nối";
    els.healthBadge.className = "status-badge degraded";
  }
}

async function loadStats() {
  try {
    const [feedbackResp, popularResp] = await Promise.all([
      fetch("/stats/feedback"),
      fetch("/stats/popular?limit=8"),
    ]);
    const feedback = await feedbackResp.json();
    const popular = await popularResp.json();
    els.likesCount.textContent = feedback.feedback_summary?.likes ?? 0;
    els.dislikesCount.textContent = feedback.feedback_summary?.dislikes ?? 0;
    els.popularList.innerHTML = "";
    (popular.popular_procedures || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = `${item.name} (${item.total_queries})`;
      els.popularList.appendChild(li);
    });
  } catch {
    els.popularList.innerHTML = "<li>Chưa tải được thống kê</li>";
  }
}

async function runSearch(query) {
  els.searchResults.innerHTML = `<p class="loading">Đang tìm...</p>`;
  els.searchCount.textContent = "";
  try {
    const response = await fetch(`/search?q=${encodeURIComponent(query)}&limit=10`);
    const data = await response.json();
    const results = data.results || [];
    els.searchCount.textContent = `${results.length} kết quả`;
    if (!results.length) {
      els.searchResults.innerHTML = `<p class="loading">Không tìm thấy thủ tục phù hợp.</p>`;
      return;
    }
    els.searchResults.innerHTML = "";
    results.forEach((item) => {
      const link = document.createElement("a");
      link.className = "result-item";
      link.href = item.url || "#";
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.innerHTML = `
        <span class="result-title">${escapeHtml(item.title)}</span>
        <span class="result-meta">${escapeHtml(item.agency || item.url || "")}</span>
        <span class="result-snippet">${escapeHtml(item.snippet || "")}</span>
      `;
      els.searchResults.appendChild(link);
    });
  } catch {
    els.searchResults.innerHTML = `<p class="loading">Không thể tìm kiếm lúc này.</p>`;
  }
}

async function sendChat(question) {
  els.sendButton.disabled = true;
  const loading = appendMessage("assistant", "_Đang truy xuất dữ liệu và tạo câu trả lời..._");
  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: state.sessionId }),
    });
    const data = await response.json();
    loading.remove();
    if (!response.ok || data.error) {
      appendMessage("assistant", data.message || "Hệ thống đang gặp sự cố. Vui lòng thử lại sau.");
      return;
    }
    state.lastRequestId = data.request_id;
    appendMessage("assistant", data.answer, {
      sources: data.sources || [],
      requestId: data.request_id,
    });
    loadStats();
  } catch {
    loading.remove();
    appendMessage("assistant", "Không thể kết nối tới máy chủ. Vui lòng thử lại sau.");
  } finally {
    els.sendButton.disabled = false;
  }
}

async function sendFeedback(rating, requestId) {
  try {
    await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_id: requestId,
        session_id: state.sessionId,
        rating,
      }),
    });
    loadStats();
  } catch {
    // Feedback is non-blocking for the user flow.
  }
}

function autoresizeInput() {
  els.chatInput.style.height = "auto";
  els.chatInput.style.height = `${Math.min(144, els.chatInput.scrollHeight)}px`;
}

els.searchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = els.searchInput.value.trim();
  if (query) runSearch(query);
});

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = els.chatInput.value.trim();
  if (!question) return;
  if (els.messages.querySelector(".empty-state")) els.messages.innerHTML = "";
  appendMessage("user", question);
  els.chatInput.value = "";
  autoresizeInput();
  await sendChat(question);
});

els.chatInput.addEventListener("input", autoresizeInput);
els.chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.chatForm.requestSubmit();
  }
});

els.clearChat.addEventListener("click", async () => {
  await fetch("/clear_session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId }),
  }).catch(() => {});
  state.sessionId = makeSessionId();
  localStorage.setItem("egov_session_id", state.sessionId);
  renderWelcome();
});

els.refreshStats.addEventListener("click", loadStats);

renderIcons();
renderWelcome();
checkHealth();
loadStats();
