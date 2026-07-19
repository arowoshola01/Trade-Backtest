"""
network_retry.py
Shared connection-retry logic used by both backtest.py (Pass 2 scoring)
and cluster_trajectory.py (trajectory capture) -- extracted to its own
module so neither has to import the other just to get call_with_reconnect.
"""

import asyncio

import websockets.exceptions

from deriv_client import DerivClient, DerivAPIError

TICK_PACE_DELAY = 0.3  # seconds between per-bar/cluster tick pulls, conservative default
RECONNECT_MAX_ATTEMPTS = 5
RECONNECT_BACKOFF_BASE = 2   # seconds; exponential: 2, 4, 8, 16, 32 -- for dropped connections
RATE_LIMIT_BACKOFF_BASE = 3  # seconds; exponential: 3, 6, 12, 24, 48 -- for Deriv's "RateLimit" error


class BacktestConnectionError(Exception):
    """Raised when a network call fails even after exhausting reconnect attempts."""
    pass


async def call_with_reconnect(client: DerivClient, label: str, description: str,
                               coro_func, *args, max_attempts: int = RECONNECT_MAX_ATTEMPTS, **kwargs):
    """
    Calls coro_func(*args, **kwargs) (an awaitable client method).

    Two distinct failure modes get retried with backoff, up to
    max_attempts each:
      - Dropped connection (ConnectionClosed / OSError): waits, then
        reconnects the client's WebSocket before retrying.
      - Deriv's "RateLimit" API error: waits (no reconnect needed, the
        socket itself is fine, just the request rate), then retries the
        same request.

    Any OTHER DerivAPIError (bad params, no data for that range, etc.) is
    NOT retried -- retrying wouldn't fix it, so it propagates immediately
    for the caller to treat as a genuine per-bar outcome.
    """
    last_exc = None
    for attempt in range(max_attempts + 1):
        try:
            return await coro_func(*args, **kwargs)
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            last_exc = e
            if attempt < max_attempts:
                wait = RECONNECT_BACKOFF_BASE * (2 ** attempt)
                print(f"\n[{label}] {description}: connection issue ({e}), "
                      f"reconnecting (attempt {attempt + 1}/{max_attempts}) in {wait}s...")
                await asyncio.sleep(wait)
                try:
                    await client.reconnect()
                except Exception as reconnect_exc:
                    # Reconnect itself failed (often a transient Deriv
                    # WrongResponse on the fresh handshake/authorize).
                    # Don't fall through to calling coro_func on a dead
                    # socket -- that would burn the rest of the retry
                    # budget instantly with non-retryable DerivAPIErrors.
                    # Falling off the end of this except block already
                    # proceeds to the next for-loop attempt naturally.
                    print(f"[{label}] reconnect attempt failed: {reconnect_exc} "
                          f"-- will retry reconnect on next attempt")
            else:
                raise BacktestConnectionError(
                    f"{label}: {description} failed after {max_attempts} reconnect attempts") from e
        except DerivAPIError as e:
            # RateLimit and WrongResponse are both transient server-side
            # issues -- the request itself is valid, Deriv just can't
            # service it right now. Retry with backoff. Any OTHER code
            # (bad params, no data for range, etc.) is a genuine
            # rejection -- retrying won't help, so propagate immediately.
            if e.code not in ("RateLimit", "WrongResponse"):
                raise
            last_exc = e
            if attempt < max_attempts:
                wait = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                reason = "rate limited" if e.code == "RateLimit" else "server error (WrongResponse)"
                print(f"\n[{label}] {description}: {reason}, waiting {wait}s "
                      f"(attempt {attempt + 1}/{max_attempts})...")
                await asyncio.sleep(wait)
                # WrongResponse often leaves the socket in a bad state --
                # force a fresh reconnect before retrying the request.
                if e.code == "WrongResponse":
                    try:
                        await client.reconnect()
                    except Exception as reconnect_exc:
                        # Falling off the end of this except block already
                        # proceeds to the next for-loop attempt naturally.
                        print(f"[{label}] reconnect after WrongResponse failed: {reconnect_exc} "
                              f"-- will retry on next attempt")
            else:
                raise BacktestConnectionError(
                    f"{label}: {description} failed after {max_attempts} retries ({e.code})") from e
    raise BacktestConnectionError(f"{label}: {description} failed") from last_exc
