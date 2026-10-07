/**
 * SentimentLens Configuration
 * Single source of truth for API URLs and performance throttles.
 */

const DEFAULT_CONFIG = {
  // Default backend URL (changeable anytime from the extension popup)
  apiBaseUrl: "http://localhost:8000",
  // Auto analyze visible tweets as the user scrolls
  autoAnalyze: true,
  // Concurrency limit for local CPU safety
  maxConcurrent: 2,
  // Debounce time for DOM mutations (ms)
  mutationDebounceMs: 350,
  // Cache retention time (1 hour)
  cacheDurationMs: 60 * 60 * 1000,
};

// Retrieve configuration merged with chrome.storage.local
async function getStoredConfig() {
  return new Promise((resolve) => {
    if (typeof chrome === "undefined" || !chrome.storage) {
      resolve(DEFAULT_CONFIG);
      return;
    }
    chrome.storage.local.get(["sentimentConfig"], (result) => {
      if (result && result.sentimentConfig) {
        resolve({ ...DEFAULT_CONFIG, ...result.sentimentConfig });
      } else {
        resolve(DEFAULT_CONFIG);
      }
    });
  });
}

// Save configuration updates
async function saveStoredConfig(newConfig) {
  return new Promise((resolve) => {
    chrome.storage.local.get(["sentimentConfig"], (result) => {
      const merged = { ...(result.sentimentConfig || DEFAULT_CONFIG), ...newConfig };
      chrome.storage.local.set({ sentimentConfig: merged }, () => {
        resolve(merged);
      });
    });
  });
}

// Make accessible to worker / window
if (typeof module !== "undefined" && module.exports) {
  module.exports = { DEFAULT_CONFIG, getStoredConfig, saveStoredConfig };
}
