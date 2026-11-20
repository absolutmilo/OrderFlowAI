let lastErrorAt = 0;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (
    message?.type !== 'aggr_btcusd_quote'
    || !sender.url
    || new URL(sender.url).origin !== 'https://btcusd.aggr.trade'
    || !Number.isFinite(message.price)
    || !Number.isSafeInteger(message.observed_at_ms)
  ) {
    return false;
  }

  fetch('http://127.0.0.1:8000/api/reference-price', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      price: message.price,
      observed_at_ms: message.observed_at_ms,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Local dashboard returned HTTP ${response.status}`);
      }
      sendResponse({ ok: true });
    })
    .catch((error) => {
      if (Date.now() - lastErrorAt >= 10000) {
        console.error('AGGR quote bridge could not reach OrderFlowAI:', error);
        lastErrorAt = Date.now();
      }
      sendResponse({ ok: false });
    });

  return true;
});
