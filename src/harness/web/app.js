/* ============================================================================
   Harness Console — frontend
   ----------------------------------------------------------------------------
   Plain ES2020, no build step. The server exposes everything needed as REST +
   SSE, so this file is a pure client: swapping it out for a different frontend
   requires no backend change.

   State model:
     sessions  → list metadata from GET /api/sessions
     thread    → messages of the selected session (GET .../messages)
     trace     → audit events (GET .../trace, then live SSE)
     draft     → in-flight assistant preview, replaced by the final message
   ========================================================================= */
"use strict";

const MAX_TRACE = 400;

const state = {
  sessions: [],
  current: null,
  source: null,
  thread: [],
  trace: [],
  lastSeq: 0,
  usage: { tokens: 0, capacity: 0 },
  models: [],
  settings: null,
  running: false,
};

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ helpers */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fmtNum(n) {
  return typeof n === "number" ? n.toLocaleString("zh-CN") : String(n ?? "");
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

let bannerTimer = null;
function banner(text) {
  const node = $("banner");
  if (!text) {
    node.className = "banner";
    node.textContent = "";
    return;
  }
  node.textContent = text;
  node.className = "banner show";
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => banner(""), 6000);
}

/* --------------------------------------------------------------- connection */

async function checkHealth() {
  try {
    const data = await api("/api/health");
    const pulse = $("pulse");
    pulse.className = state.running ? "pulse busy" : data.ok ? "pulse ok" : "pulse err";
    $("health-text").textContent = data.ok ? `${data.sessions} 个会话` : "服务不可用";
    $("tool-count").textContent = data.ok
      ? `${data.tools} 个工具 · ${data.skills} 个技能`
      : "";
  } catch {
    $("pulse").className = "pulse err";
    $("health-text").textContent = "无法连接";
    $("tool-count").textContent = "";
  }
}

async function loadModels() {
  try {
    const data = await api("/api/models");
    state.models = data.models || [];
    const select = $("model-select");
    select.innerHTML = "";
    for (const name of state.models) {
      const opt = el("option", null, name);
      opt.value = name;
      if (name === data.default) opt.selected = true;
      select.appendChild(opt);
    }
  } catch { /* the header just stays empty */ }
}

/* ----------------------------------------------------------------- sessions */

async function loadSessions() {
  const data = await api("/api/sessions");
  state.sessions = data.sessions || [];
  renderSessions();
  $("session-count").textContent = String(state.sessions.length);
}

function renderSessions() {
  const box = $("session-items");
  box.innerHTML = "";
  if (!state.sessions.length) {
    box.appendChild(el("div", "empty", "还没有会话"));
    return;
  }
  // Newest first: the list is a work queue, not an archive.
  const ordered = state.sessions.slice().reverse();
  for (const s of ordered) {
    const row = el("div", "sess" + (s.session_id === state.current ? " active" : ""));
    row.onclick = () => selectSession(s.session_id);

    const dot = el("div", "dot");
    if (s.state === "running") dot.classList.add("running");
    else if (s.state === "error") dot.classList.add("error");

    const body = el("div", "body");
    const id = el("div", "id", s.session_id);
    const sub = el("div", "sub", `${s.message_count} 条 · ${s.state}`);
    body.append(id, sub);

    const kill = el("button", "kill", "×");
    kill.title = "删除会话";
    kill.onclick = async (ev) => {
      ev.stopPropagation();
      if (!confirm(`删除会话 ${s.session_id}？此操作不可撤销。`)) return;
      try {
        await api(`/api/sessions/${s.session_id}`, { method: "DELETE" });
        if (state.current === s.session_id) {
          state.current = null;
          closeStream();
          resetChat();
          $("input").disabled = true;
          $("btn-send").disabled = true;
        }
        await loadSessions();
      } catch (e) {
        banner(`删除失败：${e.message}`);
      }
    };

    row.append(dot, body, kill);
    box.appendChild(row);
  }
}

async function createSession() {
  banner("");
  const model = $("model-select").value || null;
  try {
    const s = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ model }),
    });
    await loadSessions();
    await selectSession(s.session_id);
    $("input").focus();
  } catch (e) {
    banner(`新建会话失败：${e.message}`);
  }
}

async function selectSession(id) {
  if (state.current === id) return;
  state.current = id;
  state.trace = [];
  state.lastSeq = 0;
  state.thread = [];
  state.usage = { tokens: 0, capacity: 0 };
  closeStream();
  renderSessions();
  resetChat();
  renderTrace();
  renderContext();
  updateUsage(0, 0);
  $("input").disabled = false;
  updateSendState("idle");

  await Promise.all([loadThread(id), loadTrace(id)]);
  openStream(id);
}

async function loadThread(id) {
  try {
    const data = await api(`/api/sessions/${id}/messages`);
    state.thread = data.messages || [];
    state.usage.capacity = data.capacity || 0;
    renderMessages(state.thread);
    renderContext();
    updateUsage(data.tokens || 0, data.capacity || 0);
  } catch (e) {
    banner(`加载历史失败：${e.message}`);
  }
}

async function loadTrace(id) {
  try {
    const data = await api(`/api/sessions/${id}/trace`);
    state.trace = (data.events || []).map((e) => ({
      kind: e.kind,
      payload: e.payload || {},
      ts: e.ts,
      seq: e.seq,
    }));
    if (state.trace.length > MAX_TRACE) state.trace = state.trace.slice(-MAX_TRACE);
    state.lastSeq = state.trace.reduce((max, t) => Math.max(max, t.seq || 0), 0);
    renderTrace();
  } catch { /* trace is diagnostic; failure must not block the chat */ }
}

/* --------------------------------------------------------------------- chat */

function resetChat() {
  const stream = $("stream");
  stream.innerHTML = "";
  const ph = el("div", "placeholder");
  ph.id = "placeholder";
  ph.append(
    el("div", "big", "开始对话"),
    el("div", "hint", "输入消息并发送。工具调用与下钻过程会显示在右侧轨迹面板。")
  );
  stream.appendChild(ph);
}

function threadEl() {
  let box = $("stream").querySelector(".thread");
  if (!box) {
    const ph = $("placeholder");
    if (ph) ph.remove();
    box = el("div", "thread");
    $("stream").appendChild(box);
  }
  return box;
}

function renderMessages(messages) {
  const stream = $("stream");
  stream.innerHTML = "";
  const visible = messages.filter((m) => m.role !== "system");
  if (!visible.length) {
    resetChat();
    return;
  }
  const box = el("div", "thread");
  for (const m of visible) {
    if (m.role === "user") box.appendChild(turnNode("user", "你", m.content, m.ts));
    else if (m.role === "assistant" && m.content)
      box.appendChild(turnNode("assistant", "AI", m.content, m.ts, "final"));
  }
  stream.appendChild(box);
  scrollBottom();
}

function turnNode(role, label, text, ts, tag) {
  const wrap = el("div", `turn ${role}`);
  const who = el("div", "who", label);
  const bubble = el("div", "bubble");
  const body = el("div", "text", text);
  const meta = el("div", "meta");
  if (tag) meta.appendChild(el("span", `tag ${tag}`, tag));
  if (ts) meta.appendChild(el("span", null, fmtTime(ts)));
  bubble.append(body, meta);
  wrap.append(who, bubble);
  return wrap;
}

function appendUser(text, ts) {
  threadEl().appendChild(turnNode("user", "你", text, ts));
  state.thread.push({ role: "user", content: text, ts });
  renderContext();
  scrollBottom();
}

function appendAssistant(text, final, ts) {
  if (!text) return;
  threadEl().appendChild(
    turnNode("assistant", "AI", text, ts, final ? "final" : "reasoning")
  );
  scrollBottom();
}

/* live preview of the model's streaming output */
function appendDelta(text) {
  if (!text) return;
  let node = $("draft-text");
  if (!node) {
    const wrap = el("div", "turn assistant");
    wrap.id = "draft";
    const who = el("div", "who", "AI");
    const bubble = el("div", "bubble");
    node = el("div", "text");
    node.id = "draft-text";
    const meta = el("div", "meta");
    meta.appendChild(el("span", "tag streaming", "streaming"));
    bubble.append(node, meta);
    wrap.append(who, bubble);
    threadEl().appendChild(wrap);
  }
  node.textContent += text;
  scrollBottom();
}

function finishDelta() {
  const node = $("draft");
  if (node) node.remove();
}

function showThinking() {
  hideThinking();
  const wrap = el("div", "turn assistant");
  wrap.id = "thinking-row";
  const who = el("div", "who", "AI");
  const bubble = el("div", "bubble");
  const dots = el("div", "thinking");
  dots.append(el("i"), el("i"), el("i"));
  bubble.appendChild(dots);
  wrap.append(who, bubble);
  threadEl().appendChild(wrap);
  scrollBottom();
}

function hideThinking() {
  const node = $("thinking-row");
  if (node) node.remove();
}

function scrollBottom() {
  const s = $("stream");
  s.scrollTop = s.scrollHeight;
}

/* ---------------------------------------------------------------- composer */

function updateSendState(stateName) {
  state.running = stateName === "running";
  $("btn-send").disabled = state.running || !state.current;
  $("btn-stop").disabled = !state.running;
  $("pulse").className = state.running ? "pulse busy" : "pulse ok";
}

async function send(channel) {
  const input = $("input");
  const text = input.value.trim();
  if (!text || !state.current) return;

  if (channel === "chat") appendUser(text, Date.now() / 1000);
  input.value = "";
  autoGrow();

  try {
    await api(`/api/sessions/${state.current}/messages`, {
      method: "POST",
      body: JSON.stringify({ text, channel }),
    });
    if (channel === "chat") {
      updateSendState("running");
      showThinking();
    }
  } catch (e) {
    banner(`发送失败：${e.message}`);
  }
}

async function cancel() {
  if (!state.current) return;
  try {
    await api(`/api/sessions/${state.current}/cancel`, { method: "POST" });
  } catch (e) {
    banner(`中止失败：${e.message}`);
  }
}

function autoGrow() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

/* --------------------------------------------------------------------- SSE */

function closeStream() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
}

function openStream(id) {
  closeStream();
  // Resume after the last stored event so replay cannot duplicate the backlog
  // the trace panel just loaded.
  const src = new EventSource(`/api/sessions/${id}/events?after_seq=${state.lastSeq || 0}`);
  state.source = src;

  src.onopen = () => { $("pulse").className = "pulse ok"; };
  src.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    handleEvent(data);
  };
  src.onerror = () => { $("pulse").className = "pulse err"; };
}

function handleEvent(ev) {
  const { kind, payload = {}, ts, seq } = ev;
  if (typeof seq === "number") state.lastSeq = Math.max(state.lastSeq, seq);
  pushTrace(kind, payload, ts, seq);

  switch (kind) {
    case "delta":
      hideThinking();
      appendDelta(payload.text || "");
      break;

    case "assistant":
      hideThinking();
      finishDelta();
      if (payload.content) {
        appendAssistant(payload.content, payload.final, ts);
        state.thread.push({
          role: "assistant",
          content: payload.content,
          ts,
        });
        renderContext();
      }
      break;

    case "tool":
      hideThinking();
      // Tool activity lives only in the trace panel; the conversation stays
      // readable prose.
      if (state.thread.length) {
        state.thread.push({
          role: "tool",
          content: payload.content || payload.preview || "",
          name: payload.path,
          ts,
        });
        renderContext();
      }
      break;

    case "inject":
      if (!payload.echo) appendUser(payload.text, ts);
      break;

    case "compress":
      banner(`上下文已压缩（${payload.action}）`);
      break;

    case "usage":
      updateUsage(payload.tokens || 0, payload.capacity || 0);
      break;

    case "state":
      updateSendState(payload.state);
      if (payload.state === "running") showThinking();
      if (["done", "aborted", "error"].includes(payload.state)) {
        hideThinking();
        finishDelta();
        loadSessions().catch(() => {});
        if (state.current) loadThread(state.current).catch(() => {});
        checkHealth();
      }
      break;

    case "error":
      hideThinking();
      finishDelta();
      banner(payload.message || "推理出错");
      break;
  }
}

/* ------------------------------------------------------------------- trace */

function pushTrace(kind, payload, ts, seq) {
  // Streaming fragments are a rendering detail, not audit events: hundreds per
  // reply would bury the tool activity this panel exists to show.
  if (kind === "delta") return;
  if (seq != null && state.trace.some((t) => t.seq === seq)) return;
  state.trace.push({ kind, payload, ts: ts || Date.now() / 1000, seq });
  if (state.trace.length > MAX_TRACE) state.trace.shift();
  renderTrace();
}

function summarize(item) {
  const p = item.payload || {};
  switch (item.kind) {
    case "assistant":
      return (p.content || "").slice(0, 120) || `发起 ${(p.tool_calls || []).length} 个工具调用`;
    case "tool":
      return (p.preview || p.content || "").slice(0, 120);
    case "route":
      return p.branch ? `分支 ${p.branch}（置信度 ${p.confidence}）` : `回退全量（置信度 ${p.confidence}）`;
    case "compress":
      return `上下文压缩：${p.action}`;
    case "usage":
      return `${fmtNum(p.tokens)} / ${fmtNum(p.capacity)} tokens`;
    case "state":
      return p.detail ? `${p.state} — ${p.detail}` : p.state;
    case "inject":
      return `[${p.channel}] ${(p.text || "").slice(0, 80)}`;
    case "error":
      return p.message || "未知错误";
    default:
      return "";
  }
}

function renderTrace() {
  const box = $("trace-list");
  $("trace-count").textContent = String(state.trace.length);
  if (!state.trace.length) {
    box.innerHTML = "";
    const empty = el("div", "empty");
    empty.append("暂无事件", el("br"), "工具调用与树状下钻会显示在这里");
    box.appendChild(empty);
    return;
  }

  const frag = document.createDocumentFragment();
  for (const item of state.trace.slice().reverse()) {
    const row = el("div", "trace-row");
    const top = el("div", "top");
    const kind = el("span", "kind " + item.kind, item.kind);
    const when = el("span", "when", fmtTime(item.ts));
    top.append(kind);

    if (item.kind === "tool") {
      top.appendChild(el("span", "path", item.payload.path || ""));
    }
    top.appendChild(when);
    row.appendChild(top);

    if (item.kind === "tool") {
      const args = (item.payload.arguments || "").trim();
      if (args) row.appendChild(el("pre", "args", args));
      const body = (item.payload.content || item.payload.preview || "").trim();
      if (body) row.appendChild(el("pre", null, body));
    } else {
      const sum = summarize(item);
      if (sum) row.appendChild(el("div", "sum", sum));
    }
    frag.appendChild(row);
  }
  box.innerHTML = "";
  box.appendChild(frag);
}

/* ----------------------------------------------------------------- context */

function renderContext() {
  const box = $("context-list");
  const visible = state.thread.filter((m) => m.role !== "system");
  $("ctx-count").textContent = String(visible.length);
  if (!visible.length) {
    box.innerHTML = "";
    box.appendChild(el("div", "empty", "暂无消息"));
    return;
  }

  const frag = document.createDocumentFragment();
  for (const m of visible) {
    const row = el("div", "ctx-row " + m.role);
    const label = m.role === "tool" ? "tool" : m.role === "user" ? "user" : "ai";
    row.appendChild(el("div", "role", label));
    const content = el("div", "content" + (m.role === "tool" ? " mono" : ""));
    content.textContent =
      m.role === "tool" ? `${m.name || ""}\n${m.content || ""}` : m.content || "";
    row.appendChild(content);
    frag.appendChild(row);
  }
  box.innerHTML = "";
  box.appendChild(frag);
}

function updateUsage(tokens, capacity) {
  const cap = capacity || state.usage.capacity || 0;
  state.usage = { tokens, capacity: cap };
  const ratio = cap > 0 ? tokens / cap : 0;
  $("usage-text").textContent = cap > 0
    ? `${fmtNum(tokens)} / ${fmtNum(cap)}  (${Math.round(ratio * 100)}%)`
    : "—";
  const meter = $("usage-meter");
  meter.className = "meter" + (ratio >= 0.85 ? " hot" : ratio >= 0.5 ? " warn" : "");
  meter.querySelector("i").style.width = `${Math.min(100, ratio * 100)}%`;
}

/* ---------------------------------------------------------------- settings */

async function openSettings() {
  $("settings-overlay").classList.add("show");
  $("settings-msg").className = "msg";
  $("settings-msg").textContent = "标有「重启生效」的项目需重启服务";
  try {
    const data = await api("/api/settings");
    state.settings = data.groups || {};
    $("settings-file").textContent = data.file || "";
    renderSettings();
  } catch (e) {
    $("settings-body").innerHTML = "";
    $("settings-body").appendChild(el("div", "empty", `加载失败：${e.message}`));
  }
}

function closeSettings() {
  $("settings-overlay").classList.remove("show");
}

function renderSettings() {
  const box = $("settings-body");
  box.innerHTML = "";
  const groups = state.settings || {};

  for (const groupName of Object.keys(groups)) {
    const section = el("div", "settings-group");
    section.appendChild(el("h3", null, groupName));

    for (const item of groups[groupName]) {
      const row = el("div", "setting");

      const label = el("div", "label");
      label.appendChild(el("div", "k", item.name));
      const note = el("div", "note");
      note.appendChild(el("span", null, `默认 ${item.default}`));
      if (item.overridden) note.appendChild(el("span", "badge changed", "已修改"));
      if (item.restart_required) note.appendChild(el("span", "badge restart", "重启生效"));
      label.appendChild(note);

      const input = el("input");
      input.type = item.type === "str" ? "text" : "number";
      if (item.type === "float") input.step = "0.01";
      input.value = item.value;
      input.dataset.key = item.key;
      input.dataset.default = String(item.default);
      input.dataset.original = String(item.value);
      input.dataset.type = item.type;
      if (item.overridden) input.classList.add("changed");
      input.oninput = () => {
        const changed = String(input.value) !== String(item.default);
        input.classList.toggle("changed", changed);
        input.classList.remove("invalid");
      };

      const revert = el("button", "revert", "恢复");
      revert.onclick = () => {
        input.value = item.default;
        input.classList.remove("changed");
        input.classList.remove("invalid");
      };

      row.append(label, input, revert);
      section.appendChild(row);
    }
    box.appendChild(section);
  }
}

async function saveSettings() {
  const inputs = [...$("settings-body").querySelectorAll("input")];
  const values = {};
  let bad = 0;

  for (const input of inputs) {
    const type = input.dataset.type;
    const raw = input.value.trim();
    if (raw === "") continue;
    let value;
    if (type === "int") value = Number.parseInt(raw, 10);
    else if (type === "float") value = Number.parseFloat(raw);
    else value = raw;

    if (typeof value === "number" && !Number.isFinite(value)) {
      input.classList.add("invalid");
      bad += 1;
      continue;
    }

    // Submit only what the user actually touched. Comparing against the default
    // would resend every untouched field, and the response would then report
    // restarts for settings the user never edited.
    if (input.dataset.original === undefined) continue;
    if (String(value) === input.dataset.original) continue;
    values[input.dataset.key] = value;
  }

  if (bad) {
    setSettingsMsg(`有 ${bad} 项不是有效数值`, "err");
    return;
  }

  const msg = $("settings-msg");
  try {
    const data = Object.keys(values).length
      ? await api("/api/settings", {
          method: "PATCH",
          body: JSON.stringify({ values }),
        })
      : null;

    state.settings = data ? data.groups : state.settings;
    renderSettings();

    if (!data) {
      setSettingsMsg("没有改动。", "ok");
    } else if (data.restart_required.length) {
      setSettingsMsg(
        `已保存。以下项需重启服务才生效：${data.restart_required.join(", ")}`,
        "ok"
      );
    } else {
      setSettingsMsg("已保存，立即生效。", "ok");
    }
  } catch (e) {
    setSettingsMsg(`保存失败：${e.message}`, "err");
  }
}

async function resetSettings() {
  if (!confirm("将所有超参数恢复为默认值？")) return;
  try {
    const data = await api("/api/settings", { method: "DELETE" });
    state.settings = data.groups;
    renderSettings();
    setSettingsMsg("已全部恢复默认。", "ok");
  } catch (e) {
    setSettingsMsg(`恢复失败：${e.message}`, "err");
  }
}

function setSettingsMsg(text, cls) {
  const msg = $("settings-msg");
  msg.textContent = text;
  msg.className = `msg ${cls || ""}`;
}

/* -------------------------------------------------------------------- boot */

function bindTabs() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.onclick = () => {
      for (const t of document.querySelectorAll(".tab")) t.classList.remove("active");
      for (const p of document.querySelectorAll(".pane")) p.classList.remove("active");
      tab.classList.add("active");
      $(`pane-${tab.dataset.pane}`).classList.add("active");
    };
  }
}

function bindComposer() {
  const input = $("input");

  input.oninput = autoGrow;
  input.onkeydown = (ev) => {
    if (ev.key !== "Enter") return;
    if (ev.ctrlKey || ev.metaKey) {
      ev.preventDefault();
      send(ev.shiftKey ? "control" : "chat");
    }
  };

  $("btn-send").onclick = () => send("chat");
  $("btn-stop").onclick = cancel;
  $("btn-new").onclick = createSession;
  $("btn-settings").onclick = openSettings;
  $("settings-close").onclick = closeSettings;
  $("settings-save").onclick = saveSettings;
  $("settings-reset").onclick = resetSettings;

  $("settings-overlay").onclick = (ev) => {
    if (ev.target === $("settings-overlay")) closeSettings();
  };

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && $("settings-overlay").classList.contains("show")) {
      closeSettings();
    }
  });
}

async function boot() {
  bindTabs();
  bindComposer();
  await Promise.all([checkHealth(), loadModels(), loadSessions()]);
  setInterval(checkHealth, 15000);
}

boot().catch((e) => banner(`初始化失败：${e.message}`));
