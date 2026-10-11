"use strict";

// ---------------------------------------------------------------- elements
const $ = (id) => document.getElementById(id);
const micBtn = $("mic"), ring = $("ring"), statusEl = $("status"), connEl = $("conn");
const messagesEl = $("messages"), composer = $("composer"), textInput = $("text");

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

// --------------------------------------------------------------- websocket
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => setConn(true);
  ws.onclose = () => {
    setConn(false);
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onEvent(JSON.parse(ev.data)) : onAudio(ev.data));
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function setConn(online, label) {
  connEl.dataset.state = online ? "online" : "offline";
  connEl.textContent = label || (online ? "bağlı" : "bağlantı yok");
}

function onEvent(ev) {
  switch (ev.type) {
    case "ready":
      setConn(true, `bağlı · ${ev.llm}`);
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
      messagesEl.innerHTML = "";
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
  statusEl.textContent = STATUS[micBtn.dataset.state];
}

// -------------------------------------------------------------------- chat
function clearEmpty() {
  messagesEl.querySelector(".empty")?.remove();
}

function addUserMessage(text, language) {
  clearEmpty();
  const li = document.createElement("li");
  li.className = "msg user";
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = (language || "").toUpperCase();
  li.append(badge, document.createTextNode(text));
  messagesEl.append(li);
  li.scrollIntoView({ block: "end", behavior: "smooth" });
}

function addAgentMessage() {
  const li = document.createElement("li");
  li.className = "msg agent";
  li.innerHTML = '<div class="body">…</div><div class="meta"></div>';
  messagesEl.append(li);
  return li;
}

function appendAgentText(li, text) {
  const body = li.querySelector(".body");
  body.textContent = (body.textContent === "…" ? "" : body.textContent + " ") + text;
  li.scrollIntoView({ block: "end", behavior: "smooth" });
}

function finishAgentMessage(li, ev) {
  li.querySelector(".body").textContent = ev.text || "…";
  const meta = li.querySelector(".meta");
  const seen = new Set();
  for (const s of ev.sources || []) {
    const label = s.type === "tool" ? `🔧 ${s.id}` : `📄 ${s.id}`;
    if (seen.has(label)) continue;
    seen.add(label);
    const chip = document.createElement("span");
    chip.className = `chip ${s.type}`;
    chip.textContent = label;
    chip.title = s.type === "tool" ? JSON.stringify(s.args) : `${s.title} (skor ${s.score})`;
    meta.append(chip);
  }
  if (ev.ungrounded_numbers?.length) {
    const warn = document.createElement("span");
    warn.className = "warn";
    warn.textContent = `⚠ kaynakta olmayan sayı: ${ev.ungrounded_numbers.join(", ")}`;
    meta.append(warn);
  }
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

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;
  await ensureAudio();
  stopPlayback();
  send({ type: "text", text });
  textInput.value = "";
});

$("reset").addEventListener("click", () => {
  stopPlayback();
  send({ type: "reset" });
});

connect();
