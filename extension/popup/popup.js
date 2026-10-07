/**
 * SentimentLens Popup Controller
 * Manages API URL configuration, health testing, and cache reset.
 */

document.addEventListener("DOMContentLoaded", async () => {
  const apiUrlInput = document.getElementById("apiUrlInput");
  const saveBtn = document.getElementById("saveBtn");
  const resetLocalBtn = document.getElementById("resetLocalBtn");
  const statusBadge = document.getElementById("statusBadge");
  const testResult = document.getElementById("testResult");
  const openSidePanelBtn = document.getElementById("openSidePanelBtn");
  const clearCacheBtn = document.getElementById("clearCacheBtn");

  // Load current configuration
  const config = await getStoredConfig();
  apiUrlInput.value = config.apiBaseUrl || "http://localhost:8000";

  // Check health and update badge
  async function testConnection(url) {
    statusBadge.className = "status-badge";
    statusBadge.innerText = "Testing...";

    testResult.style.display = "block";
    testResult.className = "test-result";
    testResult.innerText = "Pinging /health endpoint...";

    const clean = url.trim().replace(/\/+$/, "");

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const res = await fetch(`${clean}/health`, {
        method: "GET",
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        statusBadge.className = "status-badge connected";
        statusBadge.innerText = "Connected";
        testResult.className = "test-result success";
        testResult.innerText = `✓ Connected successfully to ${clean}`;
        return true;
      } else {
        throw new Error(`HTTP ${res.status}`);
      }
    } catch (err) {
      statusBadge.className = "status-badge disconnected";
      statusBadge.innerText = "Offline";
      testResult.className = "test-result error";
      testResult.innerText = `✗ Cannot reach server (${err.message})`;
      return false;
    }
  }

  // Initial test
  testConnection(apiUrlInput.value);

  // Save & Test
  saveBtn.addEventListener("click", async () => {
    const newUrl = apiUrlInput.value.trim();
    if (!newUrl) return;

    await saveStoredConfig({ apiBaseUrl: newUrl });
    testConnection(newUrl);
  });

  // Reset to Localhost
  resetLocalBtn.addEventListener("click", async () => {
    const defaultUrl = "http://localhost:8000";
    apiUrlInput.value = defaultUrl;
    await saveStoredConfig({ apiBaseUrl: defaultUrl });
    testConnection(defaultUrl);
  });

  // Open Side Panel
  openSidePanelBtn.addEventListener("click", async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && chrome.sidePanel && chrome.sidePanel.open) {
      chrome.sidePanel.open({ tabId: tab.id });
      window.close();
    }
  });

  // Clear Tweet Cache
  clearCacheBtn.addEventListener("click", () => {
    chrome.runtime.sendMessage({ type: "CLEAR_MEMORY_CACHE" }, () => {
      clearCacheBtn.innerText = "Cache Cleared!";
      setTimeout(() => {
        clearCacheBtn.innerText = "Clear Tweet Cache";
      }, 1500);
    });
  });
});
