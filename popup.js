// Add event listeners to the buttons in the popup

document.getElementById('startCapture').addEventListener('click', async () => {
  // First, ask backend to start screen recording
  try {
    const resp = await fetch('http://127.0.0.1:5000/start_recording', { method: 'POST' });
    if (!resp.ok) {
      console.error('Failed to start server-side screen recording:', resp.status, await resp.text());
    } else {
      console.log('Server-side screen recording started');
    }
  } catch (err) {
    console.error('Error contacting backend to start recording:', err);
  } finally {
    // Then start capturing audio from the active tab
    chrome.runtime.sendMessage({ action: 'startCapture' });
  }
});

document.getElementById('stopCapture').addEventListener('click', () => {
  // Send a message to the background script to stop capturing
  chrome.runtime.sendMessage({ action: 'stopCapture' });
});