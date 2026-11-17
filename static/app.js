(() => {
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const resetBtn = document.getElementById("resetBtn");
  const statusEl = document.getElementById("status");
  const stateLabel = document.getElementById("stateLabel");
  const orb = document.getElementById("orb");
  const transcript = document.getElementById("transcript");

  let pc = null;
  let ws = null;
  let localStream = null;
  let remoteAudio = null;

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function setState(state) {
    stateLabel.textContent = state;
    orb.className = `orb ${state || "idle"}`;
  }

  function addLine(role, text) {
    const li = document.createElement("li");
    li.className = role;
    li.innerHTML = `<span class="who">${role}</span>${escapeHtml(text)}`;
    transcript.prepend(li);
  }

  function escapeHtml(s) {
    return s
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws`;
  }

  async function start() {
    startBtn.disabled = true;
    setStatus("Requesting microphone…");

    localStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    ws = new WebSocket(wsUrl());
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = () => reject(new Error("WebSocket failed"));
    });

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      handleServerMessage(msg);
    };
    ws.onclose = () => setStatus("Disconnected");

    pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    localStream.getTracks().forEach((t) => pc.addTrack(t, localStream));

    remoteAudio = new Audio();
    remoteAudio.autoplay = true;
    pc.ontrack = (ev) => {
      remoteAudio.srcObject = ev.streams[0] || new MediaStream([ev.track]);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Wait briefly for ICE gathering (simple LAN / Jetson use).
    await waitIceComplete(pc, 1500);

    ws.send(
      JSON.stringify({
        type: "offer",
        sdp: pc.localDescription.sdp,
        sdpType: pc.localDescription.type,
      })
    );

    stopBtn.disabled = false;
    resetBtn.disabled = false;
    setState("idle");
    setStatus("Connected — start talking");
  }

  function handleServerMessage(msg) {
    switch (msg.type) {
      case "answer":
        pc.setRemoteDescription({ type: msg.sdpType || "answer", sdp: msg.sdp });
        break;
      case "state":
        setState(msg.state);
        if (msg.barge_in) setStatus("Interrupted — listening");
        break;
      case "status":
        setStatus(msg.message);
        break;
      case "user_transcript":
        addLine("user", msg.text);
        break;
      case "assistant_text":
        // Show streaming sentence chunks; skip the final combined duplicate.
        if (msg.partial === false) break;
        addLine("assistant", msg.text);
        break;
      case "error":
        setStatus(`Error: ${msg.message}`);
        break;
      case "pc_state":
        if (msg.state === "connected") setStatus("WebRTC connected");
        break;
      default:
        break;
    }
  }

  function waitIceComplete(peer, timeoutMs) {
    if (peer.iceGatheringState === "complete") return Promise.resolve();
    return new Promise((resolve) => {
      const t = setTimeout(resolve, timeoutMs);
      peer.onicegatheringstatechange = () => {
        if (peer.iceGatheringState === "complete") {
          clearTimeout(t);
          resolve();
        }
      };
    });
  }

  async function stop() {
    stopBtn.disabled = true;
    resetBtn.disabled = true;
    startBtn.disabled = false;
    setState("idle");
    setStatus("Stopped");

    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    ws = null;

    if (pc) {
      pc.getSenders().forEach((s) => s.track && s.track.stop());
      pc.close();
      pc = null;
    }
    if (localStream) {
      localStream.getTracks().forEach((t) => t.stop());
      localStream = null;
    }
    if (remoteAudio) {
      remoteAudio.srcObject = null;
      remoteAudio = null;
    }
  }

  function resetChat() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "reset" }));
    }
    transcript.innerHTML = "";
  }

  startBtn.addEventListener("click", () => {
    start().catch((err) => {
      console.error(err);
      setStatus(err.message || String(err));
      startBtn.disabled = false;
    });
  });
  stopBtn.addEventListener("click", () => stop());
  resetBtn.addEventListener("click", () => resetChat());
})();
