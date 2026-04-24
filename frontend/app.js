/**
 * FinDecide AI — Frontend Application
 * Chat UI, API integration, result rendering, panels.
 */
const App = (() => {
  // ── State ──────────────────────────────────────────────────────────────────
  let history = [];
  let stats = { queries: 0, preds: 0, approved: 0 };
  let predHistory = [];

  // ── Utility ────────────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);
  const fmt = n => "₹" + Math.round(n).toLocaleString("en-IN");
  const pct = n => Math.round(n * 10) / 10 + "%";

  function calcEMI(p, annualRate, months) {
    const r = annualRate / 12 / 100;
    if (r === 0) return p / months;
    return p * r * Math.pow(1 + r, months) / (Math.pow(1 + r, months) - 1);
  }

  function riskColor(s) {
    if (s <= 30) return "var(--green)";
    if (s <= 60) return "var(--amber)";
    return "var(--red)";
  }

  function riskBadge(s) {
    if (s <= 30) return { label: "LOW RISK", cls: "badge-green" };
    if (s <= 60) return { label: "MEDIUM RISK", cls: "badge-amber" };
    return { label: "HIGH RISK", cls: "badge-red" };
  }

  // ── API Calls ──────────────────────────────────────────────────────────────
  async function callChat(message) {
    const res = await fetch(`${CONFIG.API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async function callPredict(formData) {
    const res = await fetch(`${CONFIG.API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async function callEMI(principal, annual_rate, months) {
    const res = await fetch(`${CONFIG.API_URL}/emi`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ principal, annual_rate, months }),
    });
    return res.json();
  }

  async function fetchModelStats() {
    const res = await fetch(`${CONFIG.API_URL}/model/stats`);
    if (!res.ok) throw new Error("Stats not available");
    return res.json();
  }

  // ── Result Card Rendering ──────────────────────────────────────────────────
  function renderCard(data) {
    if (!data) return "";
    if (data.type === "loan_prediction") return renderLoanCard(data);
    if (data.type === "emi_calc")        return renderEMICard(data);
    if (data.type === "risk_score")      return renderRiskCard(data);
    return "";
  }

  function renderLoanCard(d) {
    const approved = d.decision === "APPROVED" || d.decision === "CONDITIONAL";
    const rb = riskBadge(d.risk_score);
    const factors = (d.factors || []).map(f => `
      <li class="explain-item">
        <span class="explain-icon">${f.impact === "positive" ? "✅" : f.impact === "negative" ? "❌" : "⚠️"}</span>
        <span class="factor-${f.impact}">${f.label}</span>
      </li>`).join("");

    return `<div class="result-card">
      <div class="result-card-header">
        <span class="result-card-title">Loan Decision Analysis</span>
        <span class="badge ${approved ? "badge-green" : "badge-red"}">${d.decision}</span>
      </div>
      <div class="result-body">
        <div class="metrics-grid">
          <div class="metric-box">
            <div class="metric-box-label">Approval Probability</div>
            <div class="metric-box-val" style="color:${approved ? "var(--green)" : "var(--red)"}">
              ${d.approval_probability}<span style="font-size:13px;color:var(--text3)">%</span>
            </div>
            <div class="metric-box-sub">ML model confidence</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">Risk Score</div>
            <div class="metric-box-val" style="color:${riskColor(d.risk_score)}">
              ${d.risk_score}<span style="font-size:13px;color:var(--text3)">/100</span>
            </div>
            <div class="metric-box-sub"><span class="badge ${rb.cls}" style="font-size:10px">${rb.label}</span></div>
          </div>
          ${d.emi ? `<div class="metric-box">
            <div class="metric-box-label">Monthly EMI</div>
            <div class="metric-box-val" style="font-size:17px;color:var(--blue)">${fmt(d.emi)}</div>
            <div class="metric-box-sub">at ${d.interest_rate}% p.a.</div>
          </div>` : ""}
          ${d.loan_amount ? `<div class="metric-box">
            <div class="metric-box-label">Loan Amount</div>
            <div class="metric-box-val" style="font-size:17px">${fmt(d.loan_amount)}</div>
            <div class="metric-box-sub">${d.tenure_months ? d.tenure_months + " months" : ""}</div>
          </div>` : ""}
        </div>
        <div class="risk-bar-wrap">
          <div class="risk-bar-label"><span>Risk Level</span><span>${d.risk_score}/100</span></div>
          <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:${d.risk_score}%;background:${riskColor(d.risk_score)}"></div>
          </div>
        </div>
        ${d.total_interest ? `<div class="interest-note">Total interest payable: <strong>${fmt(d.total_interest)}</strong></div>` : ""}
        <div class="sb-label" style="margin-bottom:8px">Key Factors (Explainable AI)</div>
        <ul class="explain-list">${factors}</ul>
        ${d.conditions ? `<div class="conditions-box">📋 ${d.conditions}</div>` : ""}
      </div>
    </div>`;
  }

  function renderEMICard(d) {
    const total = Math.round((d.emi || calcEMI(d.principal, d.rate, d.months)) * d.months);
    const interest = total - d.principal;
    const ipct = Math.round(interest / d.principal * 100);
    const emi = d.emi || calcEMI(d.principal, d.rate, d.months);
    return `<div class="result-card">
      <div class="result-card-header"><span class="result-card-title">EMI Breakdown</span><span class="badge badge-blue">CALCULATED</span></div>
      <div class="result-body">
        <div class="emi-hero">
          <div class="emi-hero-label">Monthly EMI</div>
          <div class="emi-hero-val">${fmt(emi)}</div>
        </div>
        <div class="emi-result">
          <div class="emi-row"><span>Principal</span><span>${fmt(d.principal)}</span></div>
          <div class="emi-row"><span>Interest Rate</span><span>${d.rate}% p.a.</span></div>
          <div class="emi-row"><span>Tenure</span><span>${d.months} months (${(d.months / 12).toFixed(1)} yrs)</span></div>
          <div class="emi-row"><span>Total Interest</span><span style="color:var(--amber)">${fmt(interest)}</span></div>
          <div class="emi-row"><span>Total Payment</span><span style="color:var(--blue);font-weight:600">${fmt(total)}</span></div>
          <div class="emi-row"><span>Interest % of Principal</span><span style="color:${ipct > 50 ? "var(--red)" : "var(--text)"}">${ipct}%</span></div>
        </div>
      </div>
    </div>`;
  }

  function renderRiskCard(d) {
    const rb = riskBadge(d.risk_score);
    const factors = (d.factors || []).map(f => `
      <li class="explain-item">
        <span class="explain-icon">${f.impact === "positive" ? "✅" : f.impact === "negative" ? "❌" : "⚠️"}</span>
        <span class="factor-${f.impact}">${f.label}</span>
      </li>`).join("");
    return `<div class="result-card">
      <div class="result-card-header"><span class="result-card-title">Risk Profile</span><span class="badge ${rb.cls}">${rb.label}</span></div>
      <div class="result-body">
        <div class="risk-hero">
          <div class="emi-hero-label">Risk Score</div>
          <div style="font-size:48px;font-weight:700;color:${riskColor(d.risk_score)}">${d.risk_score}</div>
          <div style="font-size:12px;color:var(--text3)">out of 100 (lower = safer)</div>
        </div>
        <div class="risk-bar-wrap">
          <div class="risk-bar-track" style="height:12px">
            <div class="risk-bar-fill" style="width:${d.risk_score}%;background:${riskColor(d.risk_score)}"></div>
          </div>
          <div class="risk-bar-legend"><span>Low</span><span>Medium</span><span>High</span></div>
        </div>
        <ul class="explain-list" style="margin-top:12px">${factors}</ul>
      </div>
    </div>`;
  }

  // ── Message Rendering ──────────────────────────────────────────────────────
  function addMessage(role, text, cardData = null) {
    const msgs = $("messages");
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    const isUser = role === "user";
    const formatted = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br>");
    div.innerHTML = `
      <div class="msg-avatar ${isUser ? "user-avatar" : "ai-avatar"}">${isUser ? "U" : "💠"}</div>
      <div class="msg-content">${formatted}${cardData ? renderCard(cardData) : ""}</div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function showTyping() {
    const msgs = $("messages");
    const div = document.createElement("div");
    div.className = "msg";
    div.id = "typing-indicator";
    div.innerHTML = `<div class="msg-avatar ai-avatar">💠</div>
      <div class="msg-content"><div class="typing"><span></span><span></span><span></span></div></div>`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function removeTyping() {
    const t = $("typing-indicator");
    if (t) t.remove();
  }

  // ── Stats ──────────────────────────────────────────────────────────────────
  function updateStats(cardData) {
    stats.queries++;
    $("stat-queries").textContent = stats.queries;
    if (!cardData) return;
    if (cardData.type === "loan_prediction" || cardData.type === "risk_score") {
      stats.preds++;
      $("stat-preds").textContent = stats.preds;
    }
    if (cardData.type === "loan_prediction") {
      if (cardData.decision === "APPROVED" || cardData.decision === "CONDITIONAL") stats.approved++;
      $("stat-rate").textContent = Math.round(stats.approved / stats.preds * 100) + "%";
      predHistory.unshift({
        decision: cardData.decision,
        prob: cardData.approval_probability,
        risk: cardData.risk_score,
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      });
      if (predHistory.length > 5) predHistory.pop();
      renderHistory();
    }
  }

  function renderHistory() {
    const el = $("hist-list");
    if (!predHistory.length) {
      el.innerHTML = '<div class="empty-state">No predictions yet</div>';
      return;
    }
    el.innerHTML = predHistory.map(h => {
      const ok = h.decision === "APPROVED" || h.decision === "CONDITIONAL";
      return `<div class="hist-item">
        <div>
          <div style="font-weight:600;font-size:12px">${h.time}</div>
          <div style="color:var(--text3);font-size:11px">Risk: ${h.risk}/100 · Prob: ${h.prob}%</div>
        </div>
        <span class="hist-status ${ok ? "status-ok" : "status-bad"}">${h.decision}</span>
      </div>`;
    }).join("");
  }

  // ── Main Send Flow ─────────────────────────────────────────────────────────
  async function sendMessage() {
    const input = $("user-input");
    const btn = $("send-btn");
    const msg = input.value.trim();
    if (!msg) return;
    input.value = "";
    input.style.height = "auto";
    btn.disabled = true;
    addMessage("user", msg);
    showTyping();
    try {
      const data = await callChat(msg);
      history.push({ role: "user", content: msg });
      history.push({ role: "assistant", content: data.reply });
      if (history.length > 40) history = history.slice(-40); // keep last 20 turns
      removeTyping();
      addMessage("assistant", data.reply, data.structured_data);
      updateStats(data.structured_data);
    } catch (e) {
      removeTyping();
      addMessage("assistant", `⚠️ **Error:** ${e.message}\n\nPlease check the backend is running and your API key is configured.`);
    }
    btn.disabled = false;
    input.focus();
  }

  function quickPrompt(text) {
    $("user-input").value = text;
    sendMessage();
  }

  // ── Panel ──────────────────────────────────────────────────────────────────
  function openPanel(html) {
    $("panel-content").innerHTML = html;
    $("panel-overlay").classList.add("open");
  }

  function closePanel() {
    $("panel-overlay").classList.remove("open");
  }

  function openLoanPanel() {
    openPanel(`
      <div class="panel-header">
        <span class="panel-title">Loan Application Form</span>
        <button class="close-btn" onclick="App.closePanel()">✕</button>
      </div>
      <div class="tab-row">
        <div class="tab active" onclick="App.switchTab(this,'personal')">Personal</div>
        <div class="tab" onclick="App.switchTab(this,'financial')">Financial</div>
        <div class="tab" onclick="App.switchTab(this,'loan')">Loan Details</div>
      </div>
      <div id="tab-personal">
        <div class="form-grid">
          <div class="form-group"><label class="form-label">Age</label><input class="form-input" id="f-age" type="number" placeholder="35" min="21" max="70"></div>
          <div class="form-group"><label class="form-label">Employment Type</label>
            <select class="form-select" id="f-emp">
              <option value="salaried">Salaried</option>
              <option value="self_employed">Self Employed</option>
              <option value="business">Business Owner</option>
              <option value="freelance">Freelancer</option>
              <option value="retired">Retired</option>
            </select></div>
          <div class="form-group full"><label class="form-label">Years Employed</label><input class="form-input" id="f-exp" type="number" placeholder="5" min="0" max="40"></div>
        </div>
      </div>
      <div id="tab-financial" style="display:none">
        <div class="form-grid">
          <div class="form-group"><label class="form-label">Monthly Income (₹)</label><input class="form-input" id="f-income" type="number" placeholder="75000"></div>
          <div class="form-group"><label class="form-label">Credit Score</label><input class="form-input" id="f-credit" type="number" placeholder="720" min="300" max="900"></div>
          <div class="form-group"><label class="form-label">Existing Monthly EMI (₹)</label><input class="form-input" id="f-debt" type="number" placeholder="8000"></div>
          <div class="form-group"><label class="form-label">Default History (count)</label><input class="form-input" id="f-default" type="number" placeholder="0" min="0" max="5"></div>
        </div>
      </div>
      <div id="tab-loan" style="display:none">
        <div class="form-grid">
          <div class="form-group"><label class="form-label">Loan Type</label>
            <select class="form-select" id="f-type">
              <option value="home">Home Loan</option>
              <option value="personal">Personal Loan</option>
              <option value="vehicle">Vehicle Loan</option>
              <option value="business">Business Loan</option>
              <option value="education">Education Loan</option>
            </select></div>
          <div class="form-group"><label class="form-label">Loan Amount (₹)</label><input class="form-input" id="f-amount" type="number" placeholder="1500000"></div>
          <div class="form-group"><label class="form-label">Tenure (months)</label><input class="form-input" id="f-tenure" type="number" placeholder="240" min="12" max="360"></div>
          <div class="form-group"><label class="form-label">Purpose</label><input class="form-input" id="f-purpose" type="text" placeholder="Home purchase"></div>
        </div>
      </div>
      <button class="submit-btn" onclick="App.submitLoanForm()">🔍 Analyze Application via ML + AI</button>
    `);
  }

  function switchTab(el, tab) {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    el.classList.add("active");
    ["personal", "financial", "loan"].forEach(t => {
      const el2 = document.getElementById("tab-" + t);
      if (el2) el2.style.display = t === tab ? "block" : "none";
    });
  }

  async function submitLoanForm() {
    const g = id => document.getElementById(id)?.value || "";
    const income = parseFloat(g("f-income"));
    const credit = parseInt(g("f-credit"));
    const amount = parseFloat(g("f-amount"));
    if (!income || !credit || !amount) { alert("Please fill Income, Credit Score and Loan Amount"); return; }

    const formData = {
      age:              parseInt(g("f-age")) || 35,
      income,
      employment_type:  g("f-emp") || "salaried",
      years_employed:   parseInt(g("f-exp")) || 5,
      credit_score:     credit,
      loan_amount:      amount,
      loan_term:        parseInt(g("f-tenure")) || 240,
      existing_debt:    parseFloat(g("f-debt")) || 0,
      default_history:  parseInt(g("f-default")) || 0,
      loan_type:        g("f-type") || "home",
    };
    closePanel();

    // Trigger ML prediction directly
    addMessage("user", `Submitting loan application via ML engine: Income ₹${income}, Credit ${credit}, Loan ₹${amount}`);
    showTyping();
    try {
      const pred = await callPredict(formData);
      removeTyping();
      const text = `Here is your **ML-powered loan decision** for the submitted application. The model analyzed ${Object.keys(formData).length} features against our 50,000-row training dataset.`;
      addMessage("assistant", text, { type: "loan_prediction", ...pred });
      updateStats({ type: "loan_prediction", ...pred });
      history.push({ role: "user", content: `Loan application: ${JSON.stringify(formData)}` });
      history.push({ role: "assistant", content: `Decision: ${pred.decision}, Probability: ${pred.approval_probability}%` });
    } catch (e) {
      removeTyping();
      addMessage("assistant", `⚠️ ML prediction error: ${e.message}`);
    }
  }

  function openEMIPanel() {
    openPanel(`
      <div class="panel-header">
        <span class="panel-title">Advanced EMI Calculator</span>
        <button class="close-btn" onclick="App.closePanel()">✕</button>
      </div>
      <div class="form-grid">
        <div class="form-group full"><label class="form-label">Principal Amount (₹)</label>
          <input class="form-input" id="e-principal" type="number" placeholder="1000000" oninput="App.liveEMI()"></div>
        <div class="form-group"><label class="form-label">Annual Rate (%)</label>
          <input class="form-input" id="e-rate" type="number" placeholder="8.5" step="0.1" oninput="App.liveEMI()"></div>
        <div class="form-group"><label class="form-label">Tenure (months)</label>
          <input class="form-input" id="e-months" type="number" placeholder="240" oninput="App.liveEMI()"></div>
      </div>
      <div id="emi-live-result"></div>
      <button class="submit-btn" style="margin-top:10px" onclick="App.sendEMIToChat()">💬 Send to AI for Deep Analysis</button>
    `);
  }

  function liveEMI() {
    const p = parseFloat(document.getElementById("e-principal")?.value || 0);
    const r = parseFloat(document.getElementById("e-rate")?.value || 0);
    const n = parseInt(document.getElementById("e-months")?.value || 0);
    const el = document.getElementById("emi-live-result");
    if (!el || !p || !r || !n) { if (el) el.innerHTML = ""; return; }
    const emi = calcEMI(p, r, n);
    const total = emi * n;
    const interest = total - p;
    const ipct = Math.round(interest / p * 100);
    el.innerHTML = `<div class="emi-result" style="margin-top:12px">
      <div class="emi-hero"><div class="emi-hero-label">Monthly EMI</div><div class="emi-hero-val">${fmt(emi)}</div></div>
      <div class="emi-row"><span>Total Interest</span><span style="color:var(--amber)">${fmt(interest)}</span></div>
      <div class="emi-row"><span>Total Payment</span><span style="color:var(--blue);font-weight:600">${fmt(total)}</span></div>
      <div class="emi-row"><span>Interest % of Principal</span><span style="color:${ipct > 50 ? "var(--red)" : "var(--text)"}">${ipct}%</span></div>
    </div>`;
  }

  function sendEMIToChat() {
    const p = document.getElementById("e-principal")?.value;
    const r = document.getElementById("e-rate")?.value;
    const n = document.getElementById("e-months")?.value;
    if (!p || !r || !n) { alert("Fill all fields first"); return; }
    closePanel();
    quickPrompt(`Analyze EMI for ₹${p} loan at ${r}% p.a. for ${n} months. Is this financially prudent? What percentage of a typical income would this consume?`);
  }

  async function showModelStats() {
    try {
      const s = await fetchModelStats();
      openPanel(`
        <div class="panel-header">
          <span class="panel-title">ML Model Performance</span>
          <button class="close-btn" onclick="App.closePanel()">✕</button>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
          ${[["Model", s.model_name], ["Accuracy", (s.accuracy * 100).toFixed(1) + "%"],
             ["Precision", (s.precision * 100).toFixed(1) + "%"], ["Recall", (s.recall * 100).toFixed(1) + "%"],
             ["F1 Score", (s.f1_score * 100).toFixed(1) + "%"], ["ROC-AUC", s.roc_auc.toFixed(4)],
             ["Training Samples", s.training_samples.toLocaleString("en-IN")], ["Features", s.feature_count]
          ].map(([k, v]) => `<div class="metric-box"><div class="metric-box-label">${k}</div><div class="metric-box-val" style="font-size:18px">${v}</div></div>`).join("")}
        </div>
        <div style="font-size:12px;color:var(--text3)">Trained at: ${new Date(s.trained_at).toLocaleString()}</div>
      `);
    } catch (e) {
      openPanel(`<div class="panel-header"><span class="panel-title">Model Stats</span><button class="close-btn" onclick="App.closePanel()">✕</button></div>
        <div style="color:var(--text2);padding:20px 0">Model stats unavailable. Ensure backend is running and model is trained.</div>`);
    }
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  function clearChat() {
    $("messages").innerHTML = "";
    history = [];
    stats = { queries: 0, preds: 0, approved: 0 };
    predHistory = [];
    $("stat-queries").textContent = "0";
    $("stat-preds").textContent = "0";
    $("stat-rate").textContent = "—";
    renderHistory();
    showWelcome();
  }

  function showWelcome() {
    const msgs = $("messages");
    const div = document.createElement("div");
    div.className = "msg";
    div.innerHTML = `<div class="msg-avatar ai-avatar">💠</div>
      <div class="msg-content">
        <p>Welcome to <strong>FinDecide AI</strong> — your intelligent financial decision assistant powered by Claude AI and a trained ML model on 50,000+ banking records.</p>
        <div class="result-card">
          <div class="result-body">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
              ${[["🏦","Loan Approval","ML-powered prediction"],["📊","Risk Scoring","0–100 profile"],
                 ["🔢","EMI Calculations","Full amortization"],["🔀","What-If Analysis","Scenario modeling"],
                 ["📚","Credit Education","Tips & strategies"],["🧠","Explainable AI","Transparent decisions"]
              ].map(([i,t,s])=>`<div class="welcome-feature"><div style="font-size:18px;margin-bottom:4px">${i}</div><div style="font-size:12px;font-weight:600">${t}</div><div style="font-size:11px;color:var(--text3)">${s}</div></div>`).join("")}
            </div>
            <div style="margin-top:12px;font-size:12px;color:var(--text3)">Try: <em>"Analyze home loan for ₹60,000 income, credit score 720, ₹20L amount"</em></div>
          </div>
        </div>
      </div>`;
    msgs.appendChild(div);
  }

  function init() {
    // Textarea auto-resize
    const input = $("user-input");
    input.addEventListener("input", function () {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 120) + "px";
    });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    // Close panel on overlay click
    $("panel-overlay").addEventListener("click", function (e) {
      if (e.target === this) closePanel();
    });

    showWelcome();
  }

  return {
    sendMessage, quickPrompt,
    openLoanPanel, openEMIPanel, closePanel,
    switchTab, submitLoanForm,
    liveEMI, sendEMIToChat,
    showModelStats, clearChat, init,
  };
})();

document.addEventListener("DOMContentLoaded", App.init);
