"""
deriv_client.py
Thin async wrapper around Deriv's WebSocket API (v3).
Docs: https://developers.deriv.com/docs/websockets

NOTE: this cannot be tested from Claude's sandbox (no network route to
ws.derivws.com here) -- run it on your own machine. Usage examples are
at the bottom under `if __name__ == "__main__":`.
"""

import asyncio
import itertools
import json

import websockets

import config


class DerivClient:
    def __init__(self, ws_url: str = None, api_token: str = None):
        self.ws_url = ws_url or config.WS_URL
        self.api_token = api_token or config.API_TOKEN
        self.ws = None
        self._req_id = itertools.count(1)
        self._authorized = False

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10)
        return self

    async def reconnect(self):
        """
        Closes the current (likely dead) connection if possible, opens a
        fresh one, and re-authorizes. Used by backtest.py's retry logic
        when a tick/candle pull fails due to a dropped connection.
        """
        try:
            if self.ws is not None:
                await self.ws.close()
        except Exception:
            pass  # already dead, nothing to clean up
        await self.connect()
        if self.api_token:
            await self.authorize()

    async def close(self):
        if self.ws is not None:
            await self.ws.close()

    async def _send(self, payload: dict) -> dict:
        req_id = next(self._req_id)
        payload = {**payload, "req_id": req_id}
        await self.ws.send(json.dumps(payload))
        # Read until we get the response matching this req_id (skips
        # unrelated subscription pushes that may interleave).
        while True:
            raw = await self.ws.recv()
            data = json.loads(raw)
            if data.get("req_id") == req_id:
                if "error" in data:
                    raise DerivAPIError(data["error"])
                return data

    async def authorize(self):
        resp = await self._send({"authorize": self.api_token})
        self._authorized = True
        return resp["authorize"]

    async def get_balance(self):
        resp = await self._send({"balance": 1})
        return resp["balance"]

    async def get_website_status(self):
        """
        Returns Deriv's website_status payload, which includes an
        api_call_limits object -- use this to pace paginated tick pulls
        safely instead of guessing a delay.
        """
        resp = await self._send({"website_status": 1})
        return resp["website_status"]

    async def get_tick_history(self, symbol: str = None, count: int = 5000, start: int = 1, end: str = "latest"):
        """
        Raw tick data (not OHLC). Returns list of {epoch, price}, oldest -> newest.
        Deriv caps a single request at ~5000 ticks -- for longer history,
        page backwards using `end` = the earliest epoch from the previous
        batch (see get_tick_history_paginated below).
        """
        symbol = symbol or config.SYMBOL
        resp = await self._send({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": end,
            "start": start,
            "style": "ticks",
        })
        history = resp.get("history", {})
        prices = history.get("prices", [])
        times = history.get("times", [])
        return [{"epoch": t, "price": p} for t, p in zip(times, prices)]

    async def get_tick_history_paginated(self, symbol: str = None, total_count: int = 20000,
                                          batch_size: int = 5000, pace_delay: float = 0.3):
        """
        Pages backwards through tick history to assemble more than one
        request's worth. Returns oldest -> newest, deduplicated by epoch.
        pace_delay: seconds to wait between requests (conservative default
        to stay well under Deriv's per-second rate limits -- call
        get_website_status() first if you want to tune this against your
        account's actual api_call_limits).
        """
        symbol = symbol or config.SYMBOL
        all_ticks = []
        end = "latest"
        seen_epochs = set()

        while len(all_ticks) < total_count:
            batch = await self.get_tick_history(symbol, count=min(batch_size, total_count - len(all_ticks)), end=end)
            if not batch:
                break
            new_batch = [t for t in batch if t["epoch"] not in seen_epochs]
            if not new_batch:
                break
            for t in new_batch:
                seen_epochs.add(t["epoch"])
            all_ticks = new_batch + all_ticks  # prepend since we're paging backwards
            end = batch[0]["epoch"] - 1  # next page ends just before this batch's earliest tick
            if len(batch) < batch_size:
                break  # hit the start of available history
            if pace_delay:
                await asyncio.sleep(pace_delay)

        return all_ticks

    async def subscribe_ticks(self, symbol: str = None):
        """
        Async generator yielding each new live tick as it arrives:
        {epoch, quote}.
        """
        symbol = symbol or config.SYMBOL
        req_id = next(self._req_id)
        await self.ws.send(json.dumps({
            "ticks": symbol,
            "subscribe": 1,
            "req_id": req_id,
        }))
        while True:
            raw = await self.ws.recv()
            data = json.loads(raw)
            if "error" in data:
                raise DerivAPIError(data["error"])
            if data.get("msg_type") == "tick":
                yield data["tick"]

    async def get_candle_history_paginated(self, symbol: str = None, granularity: int = None,
                                            total_count: int = 8064, batch_size: int = 5000,
                                            pace_delay: float = 0.3, end="latest"):
        """
        Pages backwards through candle history for spans longer than one
        request's cap. Returns oldest -> newest, deduplicated by epoch.
        `end` can be 'latest' (default) or a specific epoch (int) to
        start paging backward from -- used when extending an existing
        cache with OLDER history instead of pulling the most recent span.
        """
        symbol = symbol or config.SYMBOL
        if granularity is None:
            granularity, _ = config.resolve_granularity_and_tf_cal()

        all_candles = []
        seen_epochs = set()

        while len(all_candles) < total_count:
            batch = await self.get_candle_history(
                symbol, granularity, count=min(batch_size, total_count - len(all_candles)), end=end)
            if not batch:
                break
            new_batch = [c for c in batch if c["epoch"] not in seen_epochs]
            if not new_batch:
                break
            for c in new_batch:
                seen_epochs.add(c["epoch"])
            all_candles = new_batch + all_candles
            end = batch[0]["epoch"] - 1  # next page ends just before this batch's earliest candle
            if len(batch) < batch_size:
                break  # hit the start of available history
            if pace_delay:
                await asyncio.sleep(pace_delay)

        all_candles.sort(key=lambda c: c["epoch"])
        return all_candles

    async def get_candle_history(self, symbol: str = None, granularity: int = None, count: int = None,
                                  end="latest"):
        """
        Returns a list of dicts: {epoch, open, high, low, close, volume(optional)}
        oldest -> newest. `end` can be 'latest' or a specific epoch (int)
        to page backward from -- used by get_candle_history_paginated.
        """
        symbol = symbol or config.SYMBOL
        if granularity is None:
            granularity, _ = config.resolve_granularity_and_tf_cal()
        count = count or config.CANDLE_HISTORY_COUNT

        resp = await self._send({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": end,
            "start": 1,
            "style": "candles",
            "granularity": granularity,
        })
        candles = resp.get("candles", [])
        # Deriv candles don't include tick volume for synthetic indices by
        # default; fall back to 1.0 so downstream volume-weighted
        # indicators (ZIO, VPMO) still run -- see note in README below.
        for c in candles:
            c.setdefault("volume", 1.0)
        return candles

    async def subscribe_candles(self, symbol: str = None, granularity: int = None):
        """
        Async generator yielding each new completed (or forming) candle as
        it arrives. Caller is responsible for deciding whether to act on
        forming vs. closed candles (see strategy.py docstring on live
        tick-level evaluation).
        """
        symbol = symbol or config.SYMBOL
        if granularity is None:
            granularity, _ = config.resolve_granularity_and_tf_cal()
        req_id = next(self._req_id)
        await self.ws.send(json.dumps({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": granularity,
            "subscribe": 1,
            "req_id": req_id,
        }))
        while True:
            raw = await self.ws.recv()
            data = json.loads(raw)
            if "error" in data:
                raise DerivAPIError(data["error"])
            if data.get("msg_type") == "ohlc":
                ohlc = data["ohlc"]
                ohlc.setdefault("volume", 1.0)
                yield ohlc

    async def get_proposal(self, contract_type: str, symbol: str = None, amount: float = None,
                            duration: int = None, duration_unit: str = None, basis: str = "stake"):
        """
        contract_type: 'CALL' or 'PUT'
        Returns the proposal (includes id needed to buy at that price).
        """
        symbol = symbol or config.SYMBOL
        amount = amount if amount is not None else config.DEFAULT_STAKE
        duration = duration or config.CONTRACT_DURATION
        duration_unit = duration_unit or config.CONTRACT_DURATION_UNIT

        resp = await self._send({
            "proposal": 1,
            "amount": amount,
            "basis": basis,
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol,
        })
        return resp["proposal"]

    async def buy_contract(self, contract_type: str, symbol: str = None, amount: float = None,
                            duration: int = None, duration_unit: str = None):
        """
        Fetches a fresh proposal then buys it at the quoted price.
        Returns the buy confirmation (includes contract_id for tracking).
        """
        if not self._authorized:
            raise RuntimeError("Call authorize() before placing trades.")
        proposal = await self.get_proposal(contract_type, symbol, amount, duration, duration_unit)
        resp = await self._send({
            "buy": proposal["id"],
            "price": proposal["ask_price"],
        })
        return resp["buy"]

    async def get_contract_status(self, contract_id: int):
        resp = await self._send({"proposal_open_contract": 1, "contract_id": contract_id})
        return resp["proposal_open_contract"]


class DerivAPIError(Exception):
    def __init__(self, error: dict):
        self.code = error.get("code")
        self.message = error.get("message")
        super().__init__(f"[{self.code}] {self.message}")


# ─── quick manual test / example usage — run this file directly ──────────────

async def _demo():
    client = DerivClient()
    await client.connect()

    auth = await client.authorize()
    print(f"Authorized as: {auth.get('loginid')}  (is_virtual={auth.get('is_virtual')})")

    balance = await client.get_balance()
    print(f"Balance: {balance['balance']} {balance['currency']}")

    candles = await client.get_candle_history(count=20)
    print(f"Pulled {len(candles)} candles. Last candle: {candles[-1]}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(_demo())
