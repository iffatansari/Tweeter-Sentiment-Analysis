/**
 * SentimentLens Background Service Worker (Manifest V3)
 * Proxies API requests to bypass X.com Content Security Policy (CSP).
 */

importScripts("config.js");

// Cache analyzed tweets in memory to avoid redundant calls during session
const memoryCache = new Map();

// Helper to normalize base URL
function cleanUrl(url) {
  return (url || "").trim().replace(/\/+$/, "");
}

// Perform network prediction
async function fetchPrediction(text) {
  const config = await getStoredConfig();
  const baseUrl = cleanUrl(config.apiBaseUrl);

  const endpoint = `${baseUrl}/predict`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API returned HTTP ${response.status}: ${errorText}`);
  }

  return await response.json();
}

// Perform SHAP explainability fetch
async function fetchExplain(text) {
  const config = await getStoredConfig();
  const baseUrl = cleanUrl(config.apiBaseUrl);

  const endpoint = `${baseUrl}/explain`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Explain API returned HTTP ${response.status}: ${errorText}`);
  }

  return await response.json();
}

// Perform health check
async function fetchHealth() {
  const config = await getStoredConfig();
  const baseUrl = cleanUrl(config.apiBaseUrl);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 6000);

  try {
    const response = await fetch(`${baseUrl}/health`, {
      method: "GET",
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return { ok: response.ok, status: response.status };
  } catch (err) {
    clearTimeout(timeoutId);
    return { ok: false, error: err.message };
  }
}

// Message Listener from content scripts and side panel
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ANALYZE_TWEET") {
    const { tweetId, text } = message;

    // Check memory cache
    if (memoryCache.has(tweetId)) {
      sendResponse({ success: true, data: memoryCache.get(tweetId), cached: true });
      return true;
    }

    fetchPrediction(text)
      .then((data) => {
        memoryCache.set(tweetId, data);
        sendResponse({ success: true, data, cached: false });
      })
      .catch((error) => {
        sendResponse({ success: false, error: error.message });
      });

    return true; // Keep channel open for async response
  }

  if (message.type === "CHECK_HEALTH") {
    fetchHealth()
      .then((status) => sendResponse(status))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "OPEN_SIDE_PANEL") {
    const tabId = sender.tab ? sender.tab.id : null;
    if (tabId && chrome.sidePanel && chrome.sidePanel.open) {
      chrome.sidePanel.open({ tabId }).catch((err) => {
        console.warn("Could not open side panel:", err);
      });
    }
    // Store selected tweet data for the side panel to consume
    chrome.storage.local.set({ activeTweetAnalysis: message.payload }, () => {
      // Broadcast to any already open side panel
      chrome.runtime.sendMessage({
        type: "ACTIVE_ANALYSIS_UPDATED",
        payload: message.payload,
      }).catch(() => {
        // Suppress error if side panel is not open yet
      });
    });

    sendResponse({ success: true });
    return true;
  }

  if (message.type === "GET_ACTIVE_ANALYSIS") {
    chrome.storage.local.get(["activeTweetAnalysis"], (res) => {
      sendResponse(res.activeTweetAnalysis || null);
    });
    return true;
  }

  if (message.type === "EXPLAIN_TWEET") {
    const { text } = message;
    fetchExplain(text)
      .then((data) => sendResponse({ success: true, data }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (message.type === "CLEAR_MEMORY_CACHE") {
    memoryCache.clear();
    sendResponse({ success: true });
    return true;
  }
});
