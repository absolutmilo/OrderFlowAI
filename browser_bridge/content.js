(() => {
  const quotePattern = /\bBTCUSD\s+\$?([\d,]+(?:\.\d+)?)(?:\s|$)/i;

  function readVisibleQuote() {
    const match = document.title.match(quotePattern);
    if (!match) return;

    const price = Number(match[1].replace(/,/g, ''));
    if (!Number.isFinite(price) || price <= 0) return;

    chrome.runtime.sendMessage({
      type: 'aggr_btcusd_quote',
      price,
      observed_at_ms: Date.now(),
    });
  }

  readVisibleQuote();
  window.setInterval(readVisibleQuote, 1000);
  new MutationObserver(readVisibleQuote).observe(document.head, {
    childList: true,
    subtree: true,
    characterData: true,
  });
})();
