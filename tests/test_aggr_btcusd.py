from unittest.mock import Mock
from src.connectors.aggr_btcusd import AggrBtcusdConnector


def test_parse_binance_dstream_message():
    sample_payload = {
        "e": "aggTrade",
        "E": 1731324567900,
        "s": "BTCUSD_PERP",
        "a": 1234567,
        "p": "96540.50",
        "q": "10",
        "f": 100,
        "l": 105,
        "T": 1731324567890,
        "m": True,
    }
    parsed = AggrBtcusdConnector.parse_binance_dstream_message(sample_payload)
    assert parsed is not None
    price, ts = parsed
    assert price == 96540.50
    assert ts == 1731324567890


def test_parse_binance_dstream_message_invalid():
    assert AggrBtcusdConnector.parse_binance_dstream_message({}) is None
    assert AggrBtcusdConnector.parse_binance_dstream_message({"e": "kline"}) is None
    assert AggrBtcusdConnector.parse_binance_dstream_message({"e": "aggTrade", "p": "not_a_number"}) is None
    assert AggrBtcusdConnector.parse_binance_dstream_message("not_a_dict") is None


def test_parse_bybit_inverse_message():
    sample_payload = {
        "topic": "publicTrade.BTCUSD",
        "type": "snapshot",
        "ts": 1731324567900,
        "data": [
            {
                "T": 1731324567850,
                "s": "BTCUSD",
                "S": "Buy",
                "v": "100",
                "p": "96542.00",
                "L": "PlusTick",
                "i": "trade-id-1",
                "BT": False,
            },
            {
                "T": 1731324567860,
                "s": "BTCUSD",
                "S": "Sell",
                "v": "50",
                "p": "96541.50",
                "L": "MinusTick",
                "i": "trade-id-2",
                "BT": False,
            },
        ],
    }
    trades = AggrBtcusdConnector.parse_bybit_inverse_message(sample_payload)
    assert len(trades) == 2
    assert trades[0] == (96542.00, 1731324567850)
    assert trades[1] == (96541.50, 1731324567860)


def test_parse_bybit_inverse_message_other_topic():
    sample_payload = {
        "topic": "publicTrade.ETHUSD",
        "data": [{"p": "3200.00", "T": 1731324567850}],
    }
    trades = AggrBtcusdConnector.parse_bybit_inverse_message(sample_payload)
    assert trades == []
    assert AggrBtcusdConnector.parse_bybit_inverse_message({}) == []


def test_connector_initialization_and_urls():
    conn = AggrBtcusdConnector(preferred_source="auto")
    assert not conn.is_running
    assert conn.active_source is None
    assert "dstream.binance.com" in conn.BINANCE_DSTREAM_URL
    assert "stream.bybit.com" in conn.BYBIT_INVERSE_URL


def test_emit_price_sync_callback():
    received = []

    def callback(price, ts, source):
        received.append((price, ts, source))

    conn = AggrBtcusdConnector(on_price_callback=callback)
    coro = conn._emit_price(97100.5, 1731324567890, "Binance-CoinM")
    try:
        coro.send(None)
    except StopIteration:
        pass

    assert len(received) == 1
    assert received[0] == (97100.5, 1731324567890, "Binance-CoinM")
