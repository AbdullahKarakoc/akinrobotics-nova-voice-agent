"use strict";

// ---------------------------------------------------------------- elements
const $ = (id) => document.getElementById(id);
const micBtn = $("mic"), ring = $("ring"), statusEl = $("status"), connEl = $("conn");
const messagesEl = $("messages"), composer = $("composer"), textInput = $("text");
const NOVA_MARK = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M10 22V10l12 12V10" /></svg>';
const ICONS = {
  document: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5M9 13h6M9 17h4" /></svg>',
  tool: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z" /></svg>',
};
const LLM_STATUS = {
  starting: "model başlatılıyor…",
  downloading: "model indiriliyor (ilk açılış)…",
  loading: "model belleğe yükleniyor…",
};

const STATUS = {
  idle: "Başlamak için mikrofona bir kez dokunun",
  listening: "Dinliyorum — konuşabilirsiniz (tekrar basmanıza gerek yok)",
  hearing: "Sizi duyuyorum…",
  thinking: "Düşünüyorum…",
  speaking: "Konuşuyor — konuşarak veya düğmeye basarak sözünü kesebilirsiniz",
};

// ------------------------------------------------------------------- state
let ws = null;
let audioCtx = null;
// The source node is kept in a global on purpose: some browsers garbage-collect an
// unreferenced MediaStreamAudioSourceNode, which silently stops the mic stream.
let micStream = null, micSource = null, micNode = null;
let micOn = false;
let playhead = 0;
const playing = new Set();
let reportedPlaying = false;
let droppedTurn = 0; // audio of this turn (and older) is discarded after an interrupt
let currentAgentMsg = null;
let currentTurn = 0;
const history = [];
let online = false, llmName = "", llmStatus = "ready", healthTimer = null;

// --------------------------------------------------------------- websocket
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    setConn(true);
    pollHealth();
  };
  ws.onclose = () => {
    setConn(false);
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onEvent(JSON.parse(ev.data)) : onAudio(ev.data));
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function setConn(isOnline) {
  online = isOnline;
  const loading = online && llmStatus !== "ready";
  connEl.dataset.state = !online ? "offline" : loading ? "loading" : "online";
  connEl.textContent = !online
    ? "bağlantı yok"
    : loading
      ? LLM_STATUS[llmStatus] || "model hazırlanıyor…"
      : `bağlı${llmName ? ` · ${llmName.split("/").pop()}` : ""}`;
  connEl.title = llmName;
}

// On first start the LLM may still be downloading: show it instead of letting questions fail.
async function pollHealth() {
  clearTimeout(healthTimer);
  try {
    const health = await (await fetch("/health")).json();
    llmStatus = health.llm_status || (health.llm_ready ? "ready" : "loading");
    llmName = llmName || health.llm_model || "";
  } catch {
    llmStatus = "loading";
  }
  setConn(online);
  if (llmStatus !== "ready") healthTimer = setTimeout(pollHealth, 3000);
}

function onEvent(ev) {
  switch (ev.type) {
    case "ready":
      llmName = ev.llm || llmName;
      setConn(true);
      break;
    case "vad":
      if (ev.state === "speech_start") setState("hearing");
      else setState("thinking");
      break;
    case "transcript":
      if (ev.empty) {
        setState(micOn ? "listening" : "idle");
        break;
      }
      currentTurn = ev.turn_id;
      addUserMessage(ev.text, ev.language);
      currentAgentMsg = addAgentMessage();
      setState("thinking");
      break;
    case "sentence":
      if (currentAgentMsg && ev.turn_id === currentTurn) appendAgentText(currentAgentMsg, ev.text);
      break;
    case "answer":
      if (currentAgentMsg && ev.turn_id === currentTurn) finishAgentMessage(currentAgentMsg, ev);
      if (!playing.size) setState(micOn ? "listening" : "idle");
      break;
    case "metrics":
      showMetrics(ev);
      break;
    case "interrupt":
      droppedTurn = Math.max(droppedTurn, ev.turn_id);
      stopPlayback();
      if (currentAgentMsg && ev.reason === "barge_in") currentAgentMsg.classList.add("interrupted");
      break;
    case "reset":
      showWelcome();
      break;
    case "error":
      if (currentAgentMsg) appendAgentText(currentAgentMsg, `⚠ ${ev.message}`);
      break;
  }
}

// ------------------------------------------------------------------- audio
async function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new AudioContext({ latencyHint: "interactive" });
    await audioCtx.audioWorklet.addModule("/static/pcm-worklet.js");
  }
  if (audioCtx.state === "suspended") await audioCtx.resume();
}

// "Off" pauses sending instead of releasing the device: re-acquiring the mic (new getUserMedia
// + new graph nodes) is where browsers' echo cancellation / audio graph can come back silent.
function setMicActive(on) {
  micOn = on;
  send({ type: "mic", on }); // the server resets its voice-activity state on every switch
  $("mic-off").hidden = !on;
  setState(on ? "listening" : "idle");
}

async function startMic() {
  await ensureAudio();
  if (micStream && micStream.getAudioTracks()[0]?.readyState === "live") {
    setMicActive(true); // resume a paused mic
    return;
  }
  micStream = await navigator.mediaDevices.getUserMedia({
    // Echo cancellation stays on (barge-in). Browser noise suppression is off: it smears consonants,
    // and Whisper is trained on noisy audio and copes better with the raw signal.
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true },
  });
  micSource = audioCtx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(audioCtx, "pcm-capture");
  micNode.port.onmessage = ({ data }) => {
    if (!micOn) return; // paused
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(data.pcm);
    ring.style.setProperty("--level", (1 + Math.min(data.rms * 6, 0.35)).toFixed(3));
  };
  const sink = audioCtx.createGain();
  sink.gain.value = 0; // keep the worklet pulled by the graph without playing the mic back
  micSource.connect(micNode).connect(sink).connect(audioCtx.destination);
  // e.g. the headset was unplugged: say so instead of silently listening to nothing
  micStream.getAudioTracks()[0].onended = () => {
    stopMic();
    statusEl.textContent = "Mikrofon bağlantısı kesildi — tekrar başlatmak için dokunun";
  };
  setMicActive(true);
}

// Fully releases the device; only used when the device itself went away.
function stopMic() {
  micStream?.getTracks().forEach((t) => t.stop());
  micSource?.disconnect();
  micNode?.disconnect();
  micStream = micSource = micNode = null;
  setMicActive(false);
}

function onAudio(buffer) {
  const header = new DataView(buffer, 0, 8);
  const turnId = header.getUint32(0, true);
  const sampleRate = header.getUint32(4, true);
  if (turnId <= droppedTurn || !audioCtx) return;
  const pcm = new Int16Array(buffer, 8);
  const audioBuffer = audioCtx.createBuffer(1, pcm.length, sampleRate);
  const channel = audioBuffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
  const node = audioCtx.createBufferSource();
  node.buffer = audioBuffer;
  node.connect(audioCtx.destination);
  const startAt = Math.max(audioCtx.currentTime + 0.03, playhead);
  node.start(startAt);
  playhead = startAt + audioBuffer.duration;
  playing.add(node);
  node.onended = () => {
    playing.delete(node);
    if (!playing.size) {
      reportPlaying(false);
      setState(micOn ? "listening" : "idle");
    }
  };
  reportPlaying(true);
  setState("speaking");
}

function stopPlayback() {
  playing.forEach((n) => {
    n.onended = null;
    try { n.stop(); } catch {}
  });
  playing.clear();
  playhead = 0;
  reportPlaying(false);
}

function reportPlaying(value) {
  if (value !== reportedPlaying) {
    reportedPlaying = value;
    send({ type: "playback", playing: value });
  }
}

function setState(state) {
  micBtn.dataset.state = micOn || state !== "listening" ? state : "idle";
  document.body.dataset.state = micBtn.dataset.state; // drives the halo and the step indicator
  statusEl.textContent = STATUS[micBtn.dataset.state];
}

// -------------------------------------------------------------------- chat
function showWelcome() {
  messagesEl.replaceChildren($("welcome").content.cloneNode(true));
}

function clearEmpty() {
  messagesEl.querySelector(".empty")?.remove();
}

function scrollToEnd(li) {
  li.scrollIntoView({ block: "end", behavior: "smooth" });
}

function addUserMessage(text, language) {
  clearEmpty();
  const li = document.createElement("li");
  li.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (language) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = language.toUpperCase();
    bubble.append(badge);
  }
  bubble.append(document.createTextNode(text));
  li.append(bubble);
  messagesEl.append(li);
  scrollToEnd(li);
}

function addAgentMessage() {
  const li = document.createElement("li");
  li.className = "msg agent";
  li.innerHTML =
    `<span class="avatar">${NOVA_MARK}</span>` +
    '<div class="bubble"><div class="body pending"><span class="dots"><i></i><i></i><i></i></span></div><div class="meta"></div></div>';
  messagesEl.append(li);
  scrollToEnd(li);
  return li;
}

function appendAgentText(li, text) {
  const body = li.querySelector(".body");
  if (body.classList.contains("pending")) {
    body.classList.remove("pending");
    body.textContent = text;
  } else {
    body.textContent += ` ${text}`;
  }
  scrollToEnd(li);
}

function finishAgentMessage(li, ev) {
  const body = li.querySelector(".body");
  body.classList.remove("pending");
  body.textContent = ev.text || "…";
  const meta = li.querySelector(".meta");
  const seen = new Set();
  for (const s of ev.sources || []) {
    const key = `${s.type}:${s.id}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const chip = document.createElement("span");
    chip.className = `chip ${s.type}`;
    chip.innerHTML = ICONS[s.type] || "";
    chip.append(document.createTextNode(s.id));
    chip.title = s.type === "tool" ? `Araç · ${JSON.stringify(s.args)}` : `Doküman · ${s.title} (skor ${s.score})`;
    meta.append(chip);
  }
  if (ev.ungrounded_numbers?.length) {
    const warn = document.createElement("span");
    warn.className = "warn";
    warn.textContent = `⚠ kaynakta olmayan sayı: ${ev.ungrounded_numbers.join(", ")}`;
    meta.append(warn);
  }
  scrollToEnd(li);
}

// ----------------------------------------------------------------- metrics
const SEGMENTS = [
  ["endpoint_ms", "Konuşma sonu (VAD)", "--c-vad"],
  ["stt_ms", "STT", "--c-stt"],
  ["retrieval_ms", "Bilgi arama", "--c-rag"],
  ["llm_ms", "LLM ilk cümle", "--c-llm"],
  ["tts_first_ms", "TTS ilk cümle", "--c-tts"],
];

function showMetrics(m) {
  if (m.first_audio_ms == null) return;
  const seg = { ...m };
  const known = ["endpoint_ms", "stt_ms", "retrieval_ms", "tts_first_ms"].reduce((a, k) => a + (m[k] || 0), 0);
  seg.llm_ms = Math.max(0, m.first_audio_ms - known);
  history.push(m.first_audio_ms);

  $("latency").hidden = false;
  $("lat-total").textContent = `${(m.first_audio_ms / 1000).toFixed(2)} sn`;
  const bar = $("lat-bar"), legend = $("lat-legend");
  bar.innerHTML = legend.innerHTML = "";
  for (const [key, label, color] of SEGMENTS) {
    const v = seg[key];
    if (v == null) continue;
    const span = document.createElement("span");
    span.style.width = `${(100 * v) / m.first_audio_ms}%`;
    span.style.background = `var(${color})`;
    bar.append(span);
    const dt = document.createElement("dt");
    dt.textContent = label;
    dt.style.setProperty("--dot", `var(${color})`);
    const dd = document.createElement("dd");
    dd.textContent = `${Math.round(v)} ms`;
    legend.append(dt, dd);
  }
  const extras = [];
  if (m.stt_rtf != null) extras.push(`STT RTF ${m.stt_rtf.toFixed(2)}`);
  if (m.tts_rtf != null) extras.push(`TTS RTF ${m.tts_rtf.toFixed(2)}`);
  if (m.llm_tokens_per_s != null) extras.push(`${m.llm_tokens_per_s.toFixed(1)} token/sn`);
  if (m.tools?.length) extras.push(`araç: ${m.tools.join(", ")}`);
  const avg = history.reduce((a, b) => a + b, 0) / history.length;
  $("lat-avg").textContent = `${extras.join(" · ")}${extras.length ? " · " : ""}oturum ort. ${(avg / 1000).toFixed(2)} sn (${history.length} tur)`;
}

// ------------------------------------------------------------------ wiring
// The mic stays open for the whole conversation (turns are detected by voice activity).
// Pressing the button again never closes it — people used to push-to-talk press it before
// every question. While the agent talks or thinks, a press interrupts it instead.
micBtn.addEventListener("click", async () => {
  try {
    if (!micOn) return await startMic();
    if (playing.size || ["thinking", "speaking"].includes(micBtn.dataset.state)) {
      stopPlayback();
      send({ type: "interrupt" });
      setState("listening");
    } else {
      statusEl.textContent = "Mikrofon zaten açık — konuşmanız yeterli";
    }
  } catch (err) {
    statusEl.textContent = `Mikrofon açılamadı: ${err.message}`;
  }
});

$("mic-off").addEventListener("click", () => {
  stopPlayback();
  send({ type: "interrupt" });
  setMicActive(false); // pause: no audio is sent until the mic button is pressed again
});

async function sendText(text) {
  if (!text) return;
  await ensureAudio();
  stopPlayback();
  send({ type: "text", text });
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  textInput.value = "";
  await sendText(text);
});

// Example questions in the welcome card are sent like typed ones.
messagesEl.addEventListener("click", (e) => {
  const button = e.target.closest(".suggestions button");
  if (button) sendText(button.dataset.q);
});

$("reset").addEventListener("click", () => {
  stopPlayback();
  send({ type: "reset" });
});

showWelcome();
connect();
