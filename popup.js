// Add event listeners to the buttons in the popup

document.getElementById('startCapture').addEventListener('click', () => {
  // Send a message to the background script to start capturing
  chrome.runtime.sendMessage({ action: 'startCapture' });
});

document.getElementById('stopCapture').addEventListener('click', () => {
  // Send a message to the background script to stop capturing
  chrome.runtime.sendMessage({ action: 'stopCapture' });
});