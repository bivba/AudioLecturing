console.log("Background script loaded.");

// We need to store these in the top-level scope so they can be accessed
// by both the 'start' and 'stop' message handlers.
let mediaRecorder;
let audioChunks = [];
let capturedStream; // To keep a reference to the stream for stopping it

// Listen for messages sent from the popup (popup.js)
chrome.runtime.onMessage.addListener(async (message) => {
  if (message.action === 'startCapture') {
    console.log("Received 'startCapture' message. Starting capture...");
    
    try {
      // First, get the active tab. This part can use await.
      chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
  if (!tabs || tabs.length === 0) {
    console.error("No active tab found.");
    return;
  }

  const activeTab = tabs[0];

  // Now use chrome.tabCapture here
  chrome.tabCapture.capture(
    { audio: true, video: false },
    function(stream) {
      if (chrome.runtime.lastError) {
        console.error("Error during capture:", chrome.runtime.lastError.message);
        return;
      }

      capturedStream = stream;

      mediaRecorder = new MediaRecorder(capturedStream);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
          console.log("Collected an audio chunk.");
        }
      };

      mediaRecorder.onstop = () => {
        console.log("MediaRecorder stopped. Processing audio chunks...");
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        sendAudioToServer(audioBlob);
        audioChunks = [];
      };

      mediaRecorder.start();
      console.log("MediaRecorder started.");
            }
        );
    });

    } catch (error) {
      console.error("Error starting tab capture:", error);
    }

  } else if (message.action === 'stopCapture') {
    console.log("Received 'stopCapture' message. Stopping capture...");

    // Check if the MediaRecorder is initialized and is currently recording
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop(); // This will trigger the 'onstop' event handler we defined above

      // It's good practice to stop the tracks on the stream to release resources
      // and remove the blue "capturing" icon from the browser tab.
      capturedStream.getTracks().forEach(track => track.stop());
      console.log("Capture stream stopped.");
    } else {
      console.warn("Stop was called, but MediaRecorder was not recording.");
    }
  }
});


/**
 * Sends the captured audio Blob to the Python backend for processing.
 * @param {Blob} audioBlob The audio file to send.
 */
async function sendAudioToServer(audioBlob) {
  console.log("Preparing to send audio to server...");

  // FormData is the standard way to send files via a web request
  const formData = new FormData();
  formData.append('audio', audioBlob, 'captured_audio.webm');

  try {
    // Make the POST request to your running Flask server
    const response = await fetch('http://127.0.0.1:5000/summarize_audio', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const result = await response.json();
      console.log("SUCCESS: Received summary from server:", result.text.text);
      // TODO: Display this summary to the user in the popup
    } else {
      const errorResult = await response.json();
      console.error("ERROR: Server responded with an error:", response.status, errorResult.error);
    }
  } catch (error) {
    console.error("CRITICAL ERROR: Could not connect to the server.", error);
    // TODO: Display an error message to the user in the popup
  }
}