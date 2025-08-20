import aiohttp
from loguru import logger

async def _fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None):
    async with session.get(url, params=params, timeout=30) as resp:
        resp.raise_for_status()
        return await resp.json()

async def fetch_top_markets(vs_currency: str = "usd", per_page: int = 50):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": vs_currency, "order": "market_cap_desc", "per_page": per_page, "page": 1, "sparkline": "false"}
    headers = {"Accept": "application/json"}
    async with aiohttp.ClientSession(headers=headers) as session:
        data = await _fetch_json(session, url, params)
        logger.info(f"Fetched {len(data)} markets from CoinGecko.")
        return data
