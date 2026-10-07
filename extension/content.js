/**
 * SentimentLens Content Script
 * Observes Twitter/X feed, extracts tweet text, requests analysis via background worker,
 * and renders non-intrusive sentiment/sarcasm badges.
 */

(() => {
  // Local in-memory cache to prevent duplicate requests across scrolling
  const cache = new Map();
  // Set of tweet IDs currently in flight
  const pendingRequests = new Set();
  // Queue of visible tweet elements waiting for analysis
  const analysisQueue = [];
  let activeWorkers = 0;
  const MAX_CONCURRENT = 2;

  // Simple string hash for tweets without accessible status ID
  function hashText(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash |= 0;
    }
    return "h_" + Math.abs(hash);
  }

  // Extract tweet ID from permalink or fall back to hash
  function getTweetId(article, text) {
    const timeLink = article.querySelector('a[href*="/status/"]');
    if (timeLink) {
      const match = timeLink.href.match(/\/status\/(\d+)/);
      if (match && match[1]) {
        return match[1];
      }
    }
    return hashText(text);
  }

  // Extract clean text from tweetText element
  // Extract clean text from tweetText element, preserving Twitter's <img> emojis
  function extractTweetText(article) {
    const textEl = article.querySelector('[data-testid="tweetText"]');
    if (!textEl) return "";

    // Clone element to prevent mutating Twitter/X page DOM
    const clone = textEl.cloneNode(true);

    // Replace Twitter <img> emojis with their alt unicode text (e.g. <img alt="👀"> -> "👀")
    const emojiImgs = clone.querySelectorAll("img[alt]");
    emojiImgs.forEach((img) => {
      const alt = img.getAttribute("alt") || "";
      const textNode = document.createTextNode(alt);
      if (img.parentNode) {
        img.parentNode.replaceChild(textNode, img);
      }
    });

    return clone.innerText.trim();
  }

  // Create UI badge elements
  function renderBadge(article, tweetId, text, analysisData) {
    // Avoid duplicate containers
    let container = article.querySelector(".sentimentlens-container");
    if (!container) {
      container = document.createElement("div");
      container.className = "sentimentlens-container";

      // Insert directly after tweet text, or before action group
      const tweetTextEl = article.querySelector('[data-testid="tweetText"]');
      if (tweetTextEl && tweetTextEl.parentNode) {
        tweetTextEl.parentNode.insertBefore(container, tweetTextEl.nextSibling);
      } else {
        const actionGroup = article.querySelector('div[role="group"]');
        if (actionGroup && actionGroup.parentNode) {
          actionGroup.parentNode.insertBefore(container, actionGroup);
        } else {
          article.appendChild(container);
        }
      }
    }

    container.innerHTML = ""; // Clear existing

    if (!analysisData) {
      // Loading State
      const badge = document.createElement("div");
      badge.className = "sentimentlens-badge loading";
      badge.innerHTML = `<span class="sentimentlens-spinner"></span> Analyzing...`;
      container.appendChild(badge);
      return;
    }

    if (analysisData.error) {
      // Graceful offline/error state
      const errorBadge = document.createElement("div");
      errorBadge.className = "sentimentlens-badge error";
      errorBadge.title = analysisData.error;
      errorBadge.innerText = "SentimentLens offline";
      container.appendChild(errorBadge);
      return;
    }

    const final = analysisData.final || {};
    const label = (final.label || "neutral").toLowerCase();
    const confidencePct = Math.round((final.confidence || 0) * 100);
    const sarcasm = analysisData.sarcasm || {};
    const isSarcastic = Boolean(sarcasm.is_sarcastic);

    // 1. Main Sentiment Badge
    const badge = document.createElement("div");
    badge.className = `sentimentlens-badge ${label}`;
    badge.title = "Click to inspect detailed metrics in Side Panel";
    badge.innerHTML = `
      <span class="sentimentlens-dot"></span>
      <span class="sentimentlens-label">${capitalize(label)}</span>
      <span class="sentimentlens-sep">·</span>
      <span class="sentimentlens-conf">${confidencePct}%</span>
    `;

    // 2. Sarcasm Pill (if detected)
    let sarcasmPill = null;
    if (isSarcastic) {
      const sarcasmPct = Math.round((sarcasm.sarcasm_probability || 0) * 100);
      sarcasmPill = document.createElement("span");
      sarcasmPill.className = "sentimentlens-sarcasm-pill";
      sarcasmPill.title = `Sarcasm probability: ${sarcasmPct}%`;
      sarcasmPill.innerHTML = `
        <span class="sentimentlens-sarcasm-dot"></span>
        <span>Sarcasm Detected</span>
      `;
    }

    // 3. Inspect Link
    const inspectHint = document.createElement("span");
    inspectHint.className = "sentimentlens-inspect-hint";
    inspectHint.innerHTML = "Details &rarr;";

    // Click handler to open Chrome Side Panel
    const openDetails = (e) => {
      e.stopPropagation();
      e.preventDefault();
      chrome.runtime.sendMessage({
        type: "OPEN_SIDE_PANEL",
        payload: {
          tweetId,
          text,
          analysis: analysisData,
          timestamp: Date.now(),
        },
      });
    };

    badge.addEventListener("click", openDetails);
    if (sarcasmPill) sarcasmPill.addEventListener("click", openDetails);
    inspectHint.addEventListener("click", openDetails);

    container.appendChild(badge);
    if (sarcasmPill) container.appendChild(sarcasmPill);
    container.appendChild(inspectHint);
  }

  function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  // Worker Queue to throttle requests to the local ML server
  async function processQueue() {
    if (activeWorkers >= MAX_CONCURRENT || analysisQueue.length === 0) {
      return;
    }

    const { article, tweetId, text } = analysisQueue.shift();

    // Check if element is still in DOM
    if (!document.body.contains(article)) {
      processQueue();
      return;
    }

    // If already cached, render immediately
    if (cache.has(tweetId)) {
      renderBadge(article, tweetId, text, cache.get(tweetId));
      processQueue();
      return;
    }

    activeWorkers++;
    pendingRequests.add(tweetId);
    renderBadge(article, tweetId, text, null); // Show loading spinner

    try {
      const response = await new Promise((resolve) => {
        chrome.runtime.sendMessage(
          { type: "ANALYZE_TWEET", tweetId, text },
          (res) => resolve(res)
        );
      });

      if (response && response.success) {
        cache.set(tweetId, response.data);
        renderBadge(article, tweetId, text, response.data);
      } else {
        const errorMsg = response ? response.error : "Backend unavailable";
        cache.set(tweetId, { error: errorMsg });
        renderBadge(article, tweetId, text, { error: errorMsg });
      }
    } catch (err) {
      renderBadge(article, tweetId, text, { error: err.message });
    } finally {
      activeWorkers--;
      pendingRequests.delete(tweetId);
      processQueue();
    }
  }

  // IntersectionObserver to only queue tweets that actually appear on screen
  const visibilityObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const article = entry.target;
          visibilityObserver.unobserve(article);

          const text = extractTweetText(article);
          if (!text) return;

          const tweetId = getTweetId(article, text);

          // If already cached, render right away
          if (cache.has(tweetId)) {
            renderBadge(article, tweetId, text, cache.get(tweetId));
            return;
          }

          if (!pendingRequests.has(tweetId)) {
            analysisQueue.push({ article, tweetId, text });
            processQueue();
          }
        }
      });
    },
    { rootMargin: "200px 0px" } // Pre-fetch slightly before scrolling into view
  );

  // Scan document for new tweet articles
  function scanTweets() {
    const articles = document.querySelectorAll('article[data-testid="tweet"]');
    articles.forEach((article) => {
      if (article.dataset.sentimentlensProcessed) {
        // Already registered, but if React re-rendered content, re-attach badge if needed
        const text = extractTweetText(article);
        if (text) {
          const tweetId = getTweetId(article, text);
          if (cache.has(tweetId) && !article.querySelector(".sentimentlens-container")) {
            renderBadge(article, tweetId, text, cache.get(tweetId));
          }
        }
        return;
      }

      article.dataset.sentimentlensProcessed = "true";
      visibilityObserver.observe(article);
    });
  }

  // Debounced MutationObserver to detect new tweets during infinite scrolling
  let debounceTimer = null;
  const domObserver = new MutationObserver(() => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(scanTweets, 300);
  });

  // Start observing when document is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      scanTweets();
      domObserver.observe(document.body, { childList: true, subtree: true });
    });
  } else {
    scanTweets();
    domObserver.observe(document.body, { childList: true, subtree: true });
  }

  console.log("%c[SentimentLens]%c In-feed Twitter observer active.", "color: #3b82f6; font-weight: bold;", "");
})();
