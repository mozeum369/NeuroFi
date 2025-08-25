import aiohttp
from loguru import logger

async def _fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None):
    async with session.get(url, params=params, timeout=30) as resp:
        resp.raise_for_status()
        return await resp.json()

async def fetch_top_markets(vs_currency: str = "usd", per_page: int = 50):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": "false"
    }
    headers = {"Accept": "application/json"}
    async with aiohttp.ClientSession(headers=headers) as session:
        data = await _fetch_json(session, url, params)
        logger.info(f"Fetched {len(data)} markets from CoinGecko.")
        return data

async def get_market_conditions():
    data = await fetch_top_markets()

    # Volatility: average of absolute 24h price changes
    volatility = sum(abs(asset.get("price_change_percentage_24h", 0)) for asset in data) / len(data)

    # Whale activity: volume concentration in top 5 assets
    top_volumes = sum(asset.get("total_volume", 0) for asset in data[:5])
    total_volume = sum(asset.get("total_volume", 0) for asset in data)
    whale_activity = top_volumes / total_volume if total_volume else 0

    # Sentiment score: placeholder (to be replaced with NLP analysis)
    sentiment_score = 0.5

    return {
        "volatility": volatility,
        "sentiment_score": sentiment_score,
        "whale_activity": whale_activity
    }

