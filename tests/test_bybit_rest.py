import asyncio
from unittest.mock import AsyncMock, patch

from src.connectors import bybit_rest


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def json(self):
        return self.payload

    async def text(self):
        return ""


class FakeSession:
    def __init__(self, payload=None):
        self.payload = payload or {"retCode": 0, "result": {"list": []}}
        self.params = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def get(self, _url, params):
        self.params = params
        return FakeResponse(self.payload)


class SequenceResponse(FakeResponse):
    def __init__(self, owner):
        self.owner = owner

    async def json(self):
        return self.owner.payloads.pop(0)


class SequenceSession(FakeSession):
    def __init__(self, payloads):
        super().__init__()
        self.payloads = payloads

    def get(self, _url, params):
        self.params = params
        return SequenceResponse(self)


def test_fetch_bybit_klines_parses_and_sorts_exchange_rows():
    session = FakeSession({
        "retCode": 0,
        "result": {
            "list": [
                ["2000", "2", "3", "1", "2.5", "20", "50"],
                ["1000", "1", "2", "0.5", "1.5", "10", "15"],
            ]
        },
    })

    with patch.object(bybit_rest.aiohttp, "ClientSession", return_value=session):
        candles = asyncio.run(
            bybit_rest.fetch_bybit_klines(symbol="btcusdt", interval="5", limit=2)
        )

    assert [candle["time_ms"] for candle in candles] == [1000, 2000]
    assert candles[0]["close"] == 1.5
    assert session.params == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "interval": "5",
        "limit": 2,
    }


def test_paginated_bybit_klines_moves_back_from_oldest_candle():
    session = FakeSession()
    first_page = [
        {
            "time_ms": float(timestamp),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
        for timestamp in range(1000, 2000)
    ]
    second_page = [{**first_page[0], "time_ms": 0.0}]
    fetch_page = AsyncMock(side_effect=[first_page, second_page])

    with patch.object(bybit_rest.aiohttp, "ClientSession", return_value=session):
        with patch.object(bybit_rest, "_fetch_klines_with_retries", fetch_page):
            with patch.object(bybit_rest.asyncio, "sleep", new_callable=AsyncMock):
                candles = asyncio.run(
                    bybit_rest.fetch_bybit_klines_paginated(total_bars=1001)
                )

    assert len(candles) == 1001
    assert candles[0]["time_ms"] == 0
    assert candles[-1]["time_ms"] == 1999
    assert fetch_page.await_args_list[1].kwargs["end_time_ms"] == 999


def test_bybit_temporarily_rejecting_request_retries_with_backoff():
    session = SequenceSession([
        {"retCode": 10016, "retMsg": "svc error: Get kline failed"},
        {
            "retCode": 0,
            "result": {"list": [["1000", "1", "2", "0.5", "1.5", "10", "15"]]},
        },
    ])
    with patch.object(bybit_rest.asyncio, "sleep", new_callable=AsyncMock) as sleep:
        candles = asyncio.run(
            bybit_rest._fetch_klines_with_retries(
                session, "BTCUSDT", "1", 1, None, None
            )
        )

    assert candles[0]["close"] == 1.5
    sleep.assert_awaited_once_with(1.0)


def test_non_retryable_bybit_error_is_returned_immediately():
    session = SequenceSession([{"retCode": 10001, "retMsg": "invalid parameter"}])
    with patch.object(bybit_rest.asyncio, "sleep", new_callable=AsyncMock) as sleep:
        try:
            asyncio.run(
                bybit_rest._fetch_klines_with_retries(
                    session, "BTCUSDT", "1", 1, None, None
                )
            )
        except bybit_rest.BybitAPIError as error:
            assert error.retryable is False
            assert "10001" in str(error)
        else:
            raise AssertionError("Expected BybitAPIError for invalid parameters.")

    sleep.assert_not_awaited()
