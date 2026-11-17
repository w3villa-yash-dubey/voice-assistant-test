(() => {
  const talkBtn = document.getElementById("talkBtn");
  const endBtn = document.getElementById("endBtn");
  const statusEl = document.getElementById("status");

  /** @type {import('livekit-client').Room | null} */
  let room = null;
  let connecting = false;

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function setMode(mode) {
    document.body.classList.remove("is-connected", "is-listening", "is-speaking");
    if (mode) document.body.classList.add(mode);
  }

  function attachRemoteAudio(track, participant) {
    if (track.kind !== "audio") return;
    const el = track.attach();
    el.autoplay = true;
    el.playsInline = true;
    el.dataset.participant = participant.identity;
    document.body.appendChild(el);
  }

  function clearRemoteAudio() {
    document.querySelectorAll("audio").forEach((el) => {
      el.remove();
    });
  }

  async function start() {
    if (connecting || room) return;
    connecting = true;
    talkBtn.disabled = true;
    setStatus("Connecting…");
    setMode(null);

    try {
      const LivekitClient = window.LivekitClient;
      if (!LivekitClient) {
        throw new Error("LiveKit client failed to load");
      }

      const res = await fetch("/api/token", { method: "POST" });
      if (!res.ok) {
        throw new Error(`Token request failed (${res.status})`);
      }
      const { token, url } = await res.json();
      if (!token || !url) {
        throw new Error("Token response missing url/token");
      }

      room = new LivekitClient.Room({
        adaptiveStream: true,
        dynacast: true,
        audioCaptureDefaults: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, _pub, participant) => {
        attachRemoteAudio(track, participant);
        if (track.kind === "audio") {
          setMode("is-speaking");
          setStatus("Assistant speaking");
        }
      });

      room.on(LivekitClient.RoomEvent.TrackUnsubscribed, (track) => {
        track.detach().forEach((el) => el.remove());
      });

      room.on(LivekitClient.RoomEvent.ActiveSpeakersChanged, (speakers) => {
        const local = room?.localParticipant?.identity;
        const speaking = speakers.some((p) => p.identity !== local);
        const youTalk = speakers.some((p) => p.identity === local);
        if (speaking) {
          setMode("is-speaking");
          setStatus("Assistant speaking");
        } else if (youTalk) {
          setMode("is-listening");
          setStatus("Listening…");
        } else if (room) {
          setMode("is-connected");
          setStatus("Connected — say something");
        }
      });

      room.on(LivekitClient.RoomEvent.Disconnected, () => {
        cleanupUi();
        setStatus("Disconnected");
      });

      await room.connect(url, token);
      await room.localParticipant.setMicrophoneEnabled(true);

      setMode("is-connected");
      setStatus("Connected — say something");
      talkBtn.hidden = true;
      endBtn.hidden = false;
    } catch (err) {
      console.error(err);
      await stop();
      setStatus(err?.message || "Could not connect");
      talkBtn.disabled = false;
      talkBtn.hidden = false;
      endBtn.hidden = true;
    } finally {
      connecting = false;
    }
  }

  function cleanupUi() {
    clearRemoteAudio();
    setMode(null);
    talkBtn.disabled = false;
    talkBtn.hidden = false;
    endBtn.hidden = true;
    room = null;
  }

  async function stop() {
    const current = room;
    room = null;
    if (current) {
      try {
        await current.disconnect();
      } catch (_) {
        /* ignore */
      }
    }
    cleanupUi();
    setStatus("Ready when you are");
  }

  talkBtn.addEventListener("click", start);
  endBtn.addEventListener("click", stop);
})();
