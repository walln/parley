"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  checkpoint: $("checkpoint"),
  mic: $("mic"),
  level: $("level"),
  start: $("start"),
  stop: $("stop"),
  shutdown: $("shutdown"),
  status: $("status"),
  elapsed: $("elapsed"),
  rtt: $("rtt"),
  underrun: $("underrun"),
  transcript: $("transcript"),
  error: $("error"),
  saved: $("saved"),
  player: $("player"),
  dlRecording: $("dl-recording"),
  dlInput: $("dl-input"),
  dlDetails: $("dl-details"),
};

const IN_RATE = 16000;
const OUT_RATE = 24000;

// idle -> preparing -> connecting -> live -> stopping -> idle; "off" once the GPU is shut down.
let state = "idle";
let ws = null;
let audio = null;
let sent = 0;
let limit = 0;
let startedAt = 0;
let timer = null;

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = "status " + (cls || "");
}

function setControls() {
  const open = ws && ws.readyState === WebSocket.OPEN;
  els.start.disabled = !(open && state === "idle");
  els.stop.disabled = !(state === "connecting" || state === "live");
  els.shutdown.disabled = !(open && state === "idle");
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

async function openAudio() {
  const capture = new AudioContext({ sampleRate: IN_RATE });
  const playback = new AudioContext({ sampleRate: OUT_RATE });
  if (capture.sampleRate !== IN_RATE || playback.sampleRate !== OUT_RATE) {
    throw new Error(`this browser won't run audio at ${IN_RATE} and ${OUT_RATE} Hz`);
  }
  await Promise.all([
    capture.audioWorklet.addModule("capture-worklet.js"),
    playback.audioWorklet.addModule("playback-worklet.js"),
  ]);
  // The model trained on voices near 0.05 RMS over digital silence and never heard itself on the
  // user channel, so the browser levels the microphone, suppresses room noise and cancels the
  // model's playback.
  const mic = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const [track] = mic.getAudioTracks();
  const echo = track.getSettings().echoCancellation ? "echo cancellation on" : "echo cancellation off";
  els.mic.textContent = `${track.label || "unnamed microphone"} · ${echo}`;
  track.onmute = () => setStatus(`the microphone went quiet — the system muted ${track.label}`, "err");
  track.onunmute = () => setStatus("live — speak now", "ready");
  track.onended = () => fail(`the microphone stopped: ${track.label}`);
  await Promise.all([capture.resume(), playback.resume()]);
  const captureNode = new AudioWorkletNode(capture, "capture");
  const playbackNode = new AudioWorkletNode(playback, "playback");
  playbackNode.connect(playback.destination);
  playbackNode.port.onmessage = (event) => {
    if (event.data.type === "overflow") fail("playback fell behind the model");
    if (event.data.type === "drained") els.underrun.textContent = String(event.data.underruns);
  };
  captureNode.port.onmessage = (event) => {
    if (state !== "live") return;
    showLevel(new Int16Array(event.data));
    ws.send(event.data);
    sent += 1;
    if (sent >= limit) stopTalking();
  };
  const source = capture.createMediaStreamSource(mic);
  return { capture, playback, mic, source, captureNode, playbackNode };
}

// Two seconds of exact zeros means the device stopped delivering audio, not a quiet room.
const SILENT_FRAMES = 25;
let silentFrames = 0;

function showLevel(pcm) {
  let peak = 0;
  for (let i = 0; i < pcm.length; i += 1) peak = Math.max(peak, Math.abs(pcm[i]));
  els.level.value = peak / 32768;
  silentFrames = peak === 0 ? silentFrames + 1 : 0;
  if (silentFrames === SILENT_FRAMES) {
    setStatus(`the microphone is sending silence — check the input device (${els.mic.textContent})`, "err");
  } else if (silentFrames === 0 && els.status.classList.contains("err")) {
    setStatus("live — speak now", "ready");
  }
}

function stopMicrophone() {
  if (!audio) return;
  for (const track of audio.mic.getTracks()) track.stop();
  try {
    audio.source.disconnect();
    audio.captureNode.disconnect();
  } catch (e) {}
  clearInterval(timer);
}

function closeAudio() {
  if (!audio) return;
  stopMicrophone();
  audio.capture.close().catch(() => {});
  audio.playback.close().catch(() => {});
  audio = null;
}

function fail(message) {
  if (state === "live" || state === "connecting") send({ type: "stop" });
  closeAudio();
  state = "idle";
  els.error.textContent = message;
  setStatus(message, "err");
  setControls();
}

function stopTalking() {
  if (state !== "live" && state !== "connecting") return;
  state = "stopping";
  stopMicrophone();
  send({ type: "stop" });
  setStatus("finishing the model's last frames…", "busy");
  setControls();
}

function onReady(msg) {
  if (state !== "connecting" || !audio) return;
  limit = msg.frames;
  sent = 0;
  audio.source.connect(audio.captureNode);
  audio.captureNode.connect(audio.capture.destination);
  state = "live";
  startedAt = performance.now();
  timer = setInterval(() => {
    els.elapsed.textContent = `${((performance.now() - startedAt) / 1000).toFixed(1)} s`;
  }, 100);
  setStatus("live — speak now", "ready");
  setControls();
}

function onSaved(msg) {
  els.saved.classList.remove("hidden");
  els.player.src = msg.recording;
  els.dlRecording.href = msg.recording;
  els.dlInput.href = msg.input;
  els.dlDetails.href = msg.details;
  if (audio) audio.playbackNode.port.postMessage({ type: "drain" });
  setTimeout(closeAudio, 2500);
  if (state !== "idle") setStatus("conversation saved — start another when you like", "ready");
  state = "idle";
  setControls();
}

function onMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    if (audio) audio.playbackNode.port.postMessage({ type: "pcm", buffer: event.data }, [event.data]);
    return;
  }
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case "hello":
      els.checkpoint.textContent = msg.checkpoint;
      setStatus("ready — press Start talking", "ready");
      break;
    case "ready":
      onReady(msg);
      break;
    case "frame":
      els.rtt.textContent = `${msg.round_trip_ms.toFixed(0)} ms`;
      if (msg.text) els.transcript.textContent += msg.text;
      break;
    case "saved":
      onSaved(msg);
      break;
    case "error":
      fail(msg.message);
      break;
    case "shut-down":
      state = "off";
      setStatus("the GPU is shutting down — you can close this page", "busy");
      break;
  }
  setControls();
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = setControls;
  ws.onmessage = onMessage;
  ws.onclose = () => {
    closeAudio();
    state = "off";
    setStatus("the bridge stopped — run just talk again", "err");
    setControls();
  };
}

els.start.addEventListener("click", async () => {
  if (state !== "idle") return;
  state = "preparing";
  els.error.textContent = "";
  els.transcript.textContent = "";
  els.saved.classList.add("hidden");
  els.elapsed.textContent = "0.0 s";
  els.rtt.textContent = "—";
  els.underrun.textContent = "0";
  setStatus("waiting for the microphone…", "busy");
  setControls();
  try {
    audio = await openAudio();
  } catch (err) {
    fail(err && err.name === "NotAllowedError" ? "microphone permission denied" : `audio setup failed: ${err && err.message}`);
    return;
  }
  state = "connecting";
  setStatus("opening a session on the GPU…", "busy");
  send({ type: "start" });
  setControls();
});

els.stop.addEventListener("click", stopTalking);

els.shutdown.addEventListener("click", () => {
  if (state !== "idle") return;
  send({ type: "shutdown" });
  setStatus("asking the GPU to shut down…", "busy");
});

window.addEventListener("pagehide", () => {
  closeAudio();
  if (ws) ws.close();
});

connect();
