"""
OrderFlowAI - Application Runner.
Launches the FastAPI backend and opens the live Order Flow & Sonification Dashboard.
"""

import sys
import os
import webbrowser
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import CONFIG


def main():
    print("=" * 65)
    print("  ⚡ OrderFlowAI - Microestructura & Sonificación en Tiempo Real")
    print("=" * 65)
    active_feeds = []
    if CONFIG.exchange.enable_binance_spot:
        active_feeds.append("Binance-Spot")
    if CONFIG.exchange.enable_bybit_linear:
        active_feeds.append("Bybit-Linear")
    if CONFIG.exchange.enable_binance_futures:
        active_feeds.append("Binance-Futures")
    feeds_str = " + ".join(active_feeds) or getattr(CONFIG.exchange, "name", "Active")

    print(f"[*] Exchange Feeds: {feeds_str} ({CONFIG.exchange.symbol})")
    print(f"[*] Dashboard URL: http://{CONFIG.server.host}:{CONFIG.server.port}")
    print("=" * 65)

    # Open browser automatically
    try:
        webbrowser.open(f"http://{CONFIG.server.host}:{CONFIG.server.port}")
    except Exception:
        pass

    # Launch Uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=CONFIG.server.host,
        port=CONFIG.server.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
