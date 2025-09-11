console.log("Background script loaded.");

let capturedStream = null;
let recorderA = null;
let recorderB = null;
let activeRecorder = null;   // the one currently recording
let nextRecorder = null;     // overlap recorder
let rotateTimer = null;
let sessionId = null;
let chunkIndex = 0;
let isStopping = false;
let finalSent = false;

const CHUNK_MS = 12 * 60 * 1000; // 12 minutes
const OVERLAP_MS = 1000;         // 1 second overlap

// pick supported mimeType
function getRecorderOptions() {
  let options = { mimeType: "audio/webm;codecs=opus" };
  if (!MediaRecorder.isTypeSupported(options.mimeType)) {
    options = { mimeType: "audio/webm" };
  }
  return options;
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "startCapture") {
    startCapture();
  } else if (message.action === "stopCapture") {
    stopCapture();
  }
});

function createRecorder(stream) {
  const options = getRecorderOptions();
  const mr = new MediaRecorder(stream, options);

  // capture reference to this recorder in the closure
  mr._isActiveRecorder = false; // we'll set this for the 'activeRecorder' when started

  mr.ondataavailable = (event) => {
    if (!event.data || event.data.size === 0) return;

    // If we're stopping: allow final chunk only from the recorder that was active when stop called
    if (isStopping) {
      // Only the recorder that matches activeRecorder at stop time should send final chunk
      if (!finalSent && mr === activeRecorder) {
        finalSent = true;
        console.log("➡️ Sending FINAL chunk via /merge (from activeRecorder)");
        sendFinalChunk(event.data);
      } else {
        console.log("Ignored final chunk from non-active recorder.");
      }
      return;
    }

    // Normal periodic chunk
    sendPeriodicChunk(event.data);
  };

  mr.onstop = () => {
    console.log("Recorder stopped.");
  };

  return mr;
}

async function sendPeriodicChunk(blob) {
  const filename = `chunk_${sessionId}_${chunkIndex}.webm`;
  const form = new FormData();
  form.append("audio", blob, filename);
  form.append("session_id", sessionId);
  form.append("chunk_index", String(chunkIndex));

  const currentIndex = chunkIndex;
  chunkIndex += 1;

  try {
    const resp = await fetch("http://127.0.0.1:5000/summarize_audio", {
      method: "POST",
      body: form,
    });
    if (!resp.ok) {
      console.error(`[${sessionId}] chunk ${currentIndex} upload failed:`, resp.status, await resp.text());
    } else {
      console.log(`[${sessionId}] chunk ${currentIndex} uploaded`);
    }
  } catch (err) {
    console.error(`[${sessionId}] chunk ${currentIndex} upload error:`, err);
  }
}

async function sendFinalChunk(blob) {
  const filename = `final_${sessionId}.webm`;
  const form = new FormData();
  form.append("audio", blob, filename);
  form.append("session_id", sessionId);

  try {
    const resp = await fetch("http://127.0.0.1:5000/merge", {
      method: "POST",
      body: form,
    });
    if (!resp.ok) {
      console.error(`[${sessionId}] final merge failed:`, resp.status, await resp.text());
    } else {
      console.log(`[${sessionId}] final merge done:`, await resp.json());
    }
  } catch (err) {
    console.error(`[${sessionId}] final merge error:`, err);
  }
}

function rotateOnce() {
  if (!capturedStream) {
    console.warn("rotateOnce called but no stream present");
    return;
  }

  nextRecorder = createRecorder(capturedStream);
  nextRecorder.start();
  console.log("Started next recorder for overlap.");

  setTimeout(() => {
    if (activeRecorder && activeRecorder.state === "recording") {
      activeRecorder.stop();
      console.log("Stopped previous recorder after overlap.");
    }
    activeRecorder = nextRecorder;
    nextRecorder = null;
  }, OVERLAP_MS);
}

function startCapture() {
  if (capturedStream) {
    console.warn("Already capturing.");
    return;
  }

  sessionId = crypto.randomUUID();
  chunkIndex = 0;
  isStopping = false;
  finalSent = false;
  console.log("Starting capture session:", sessionId);

  chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
    if (chrome.runtime.lastError || !stream) {
      console.error("tabCapture failed:", chrome.runtime.lastError?.message);
      return;
    }

    capturedStream = stream;

    const audioElement = new Audio();
    audioElement.srcObject = capturedStream;
    audioElement.play();

    activeRecorder = createRecorder(capturedStream);
    activeRecorder.start();
    console.log("Initial recorder started.");

    const firstDelay = CHUNK_MS - OVERLAP_MS;
    setTimeout(() => {
      rotateOnce();
      rotateTimer = setInterval(rotateOnce, CHUNK_MS);
    }, firstDelay);

    console.log(`Chunking every ${CHUNK_MS / 60000} minutes with ${OVERLAP_MS}ms overlap.`);
  });
}

function stopCapture() {
  if (!activeRecorder || activeRecorder.state !== "recording") {
    console.warn("Stop requested but no active recording.");
    return;
  }

  console.log("Stop requested. Flushing final chunk...");
  isStopping = true;
  finalSent = false;

  if (rotateTimer) {
    clearInterval(rotateTimer);
    rotateTimer = null;
  }

  try {
    if (nextRecorder && nextRecorder.state === "recording") {
      nextRecorder.stop();
      console.log("Stopped overlapping recorder during stop.");
    }
  } catch (e) { console.warn("Error stopping nextRecorder:", e); }

  try { activeRecorder.requestData(); } catch (e) {}
  setTimeout(() => {
    try { activeRecorder.stop(); } catch (e) {}
  }, 250);

  setTimeout(() => {
    if (capturedStream) {
      capturedStream.getTracks().forEach(t => t.stop());
      capturedStream = null;
    }
  }, 1500);
}