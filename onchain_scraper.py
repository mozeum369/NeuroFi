import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

import requests

# Import utility functions from data_utils
from data_utils import log_message, safe_fetch_json, save_data_snapshot

# =========================
# Config & Output
# =========================

OUTPUT_DIR = Path("ai_core/onchain_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ETH_RPC = os.getenv("ETH_RPC", "https://cloudflare-eth.com")
ARBITRUM_RPC = os.getenv("ARBITRUM_RPC", "https://arb1.arbitrum.io/rpc")
SOLANA_RPC = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com")

DEFAULT_BLOCK_LOOKBACK = int(os.getenv("BLOCK_LOOKBACK", "2000"))
DEFAULT_CHUNK_SIZE = int(os.getenv("BLOCK_CHUNK_SIZE", "1000"))

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

# =========================
# Utilities
# =========================

def _now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _rpc_call(url: str, method: str, params: list) -> Any:
    headers = {"Content-Type": "application/json"}
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"RPC error from {url}: {data['error']}")
        log_message(f"RPC call to {url} method {method} succeeded.")
        return data.get("result")
    except Exception as e:
        log_message(f"RPC call to {url} method {method} failed: {e}", level='error')
        return None

def _hex_to_int(hex_str: str) -> int:
    return int(hex_str, 16)

def _pad_address_topic(address: str) -> str:
    addr = address.lower()
    if not addr.startswith("0x") or len(addr) != 42:
        raise ValueError(f"Invalid EVM address: {address}")
    return "0x" + ("0" * 24) + addr[2:]

def _looks_like_solana_address(addr: str) -> bool:
    return (not addr.startswith("0x")) and (30 <= len(addr) <= 50)

def _chunk_ranges(start: int, end: int, step: int) -> Iterable[tuple[int, int]]:
    cur = start
    while cur <= end:
        b = min(cur + step - 1, end)
        yield cur, b
        cur = b + 1

# =========================
# EVM (Ethereum / Arbitrum)
# =========================

def _evm_latest_block(rpc_url: str) -> int:
    latest_hex = _rpc_call(rpc_url, "eth_blockNumber", [])
    return _hex_to_int(latest_hex) if latest_hex else 0

def _evm_fetch_logs_with_chunking(rpc_url: str, base_filter: Dict[str, Any], from_block: int, to_block: int, initial_chunk: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    chunk = max(1, initial_chunk)
    floor = 64
    a = from_block
    while a <= to_block:
        b = min(a + chunk - 1, to_block)
        params = [dict(base_filter, **{"fromBlock": hex(a), "toBlock": hex(b)})]
        try:
            logs = _rpc_call(rpc_url, "eth_getLogs", params) or []
            results.extend(logs)
            a = b + 1
        except Exception as e:
            log_message(f"{rpc_url} getLogs chunk {a}-{b} failed: {e}", level='warning')
            if chunk <= floor:
                if chunk == 1:
                    raise
                chunk = 1
            else:
                chunk = max(floor, chunk // 2)
    return results

def _evm_get_erc20_transfers_for_address(rpc_url: str, address: str, lookback_blocks: int, contracts: Optional[List[str]] = None, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[Dict[str, Any]]:
    latest = _evm_latest_block(rpc_url)
    from_block = max(latest - lookback_blocks, 0)
    addr_topic = _pad_address_topic(address)
    all_logs: List[Dict[str, Any]] = []

    def run_filters(contract_list: Optional[List[str]]):
        nonlocal all_logs
        address_field = contract_list if contract_list else None

        base_from = {"topics": [TRANSFER_TOPIC, addr_topic]}
        if address_field:
            base_from["address"] = address_field
        logs_from = _evm_fetch_logs_with_chunking(rpc_url, base_from, from_block, latest, chunk_size)
        all_logs.extend(logs_from)

        base_to = {"topics": [TRANSFER_TOPIC, None, addr_topic]}
        if address_field:
            base_to["address"] = address_field
        logs_to = _evm_fetch_logs_with_chunking(rpc_url, base_to, from_block, latest, chunk_size)
        all_logs.extend(logs_to)

    if contracts and len(contracts) > 0:
        BATCH = 8
        for i in range(0, len(contracts), BATCH):
            run_filters(contracts[i:i+BATCH])
    else:
        run_filters(None)

    transfers = []
    for log in all_logs:
        topics = log.get("topics", [])
        data_hex = log.get("data", "0x0")
        try:
            value = _hex_to_int(data_hex) if data_hex and data_hex != "0x" else 0
        except Exception:
            value = 0
        from_addr = "0x" + topics[1][-40:] if len(topics) >= 3 and topics[1].startswith("0x") else None
        to_addr = "0x" + topics[2][-40:] if len(topics) >= 3 and topics[2].startswith("0x") else None
        transfers.append({
            "network": rpc_url,
            "blockNumber": _hex_to_int(log.get("blockNumber", "0x0")) if log.get("blockNumber") else None,
            "txHash": log.get("transactionHash"),
            "logIndex": _hex_to_int(log.get("logIndex", "0x0")) if log.get("logIndex") else None,
            "contract": log.get("address"),
            "from": from_addr,
            "to": to_addr,
            "value": value
        })

    transfers.sort(key=lambda x: (x["blockNumber"] or 0, x["logIndex"] or 0))
    return transfers

def fetch_evm_onchain_for_address(address: str, lookback_blocks: int, contracts: Optional[List[str]] = None) -> Dict[str, Any]:
    if not address.startswith("0x"):
        return {}
    out: Dict[str, Any] = {"ethereum": {}, "arbitrum": {}}
    try:
        eth_transfers = _evm_get_erc20_transfers_for_address(ETH_RPC, address, lookback_blocks, contracts)
        out["ethereum"] = {"erc20_transfers": eth_transfers}
    except Exception as e:
        log_message(f"Ethereum RPC failed for {address}: {e}", level='warning')
    try:
        arb_transfers = _evm_get_erc20_transfers_for_address(ARBITRUM_RPC, address, lookback_blocks, contracts)
        out["arbitrum"] = {"erc20_transfers": arb_transfers}
    except Exception as e:
        log_message(f"Arbitrum RPC failed for {address}: {e}", level='warning')
    return out

# =========================
# Solana
# =========================

def fetch_solana_activity(address: str, limit: int = 25) -> Dict[str, Any]:
    if not _looks_like_solana_address(address):
        return {}
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [address, {"limit": limit}]
    }
    try:
        resp = requests.post(SOLANA_RPC, headers={"Content-Type": "application/json"}, json=payload, timeout=25)
        resp.raise_for_status()
        result = resp.json().get("result", [])
        signatures = [{
            "signature": r.get("signature"),
            "slot": r.get("slot"),
            "err": r.get("err"),
            "blockTime": r.get("blockTime")
        } for r in result]
        log_message(f"Fetched Solana activity for {address}")
        return {"signatures": signatures}
    except Exception as e:
        log_message(f"Solana RPC failed for {address}: {e}", level='warning')
        return {}

# =========================
# Save & Orchestrate
# =========================

def run_onchain_scraper_rpc(addresses: List[str], lookback_blocks: int = DEFAULT_BLOCK_LOOKBACK, chains: Optional[List[str]] = None, contracts: Optional[List[str]] = None) -> Dict[str, Any]:
    chosen = set(chains or ["ethereum", "arbitrum", "solana"])
    aggregate: Dict[str, Any] = {}
    for addr in addresses:
        record: Dict[str, Any] = {}
        if {"ethereum", "arbitrum"} & chosen and addr.startswith("0x"):
            evm_data = fetch_evm_onchain_for_address(addr, lookback_blocks, contracts)
            if evm_data:
                record.update(evm_data)
                save_data_snapshot(evm_data, prefix=f"evm_{addr}")
        if "solana" in chosen and _looks_like_solana_address(addr):
            sol_data = fetch_solana_activity(addr)
            if sol_data:
                record["solana"] = sol_data
                save_data_snapshot(sol_data, prefix=f"solana_{addr}")
        aggregate[addr] = record
    return aggregate

# =========================
# Agent Entry Point
# =========================

def gather_onchain_data_for_goal(goal: Any) -> Dict[str, Any]:
    addresses: List[str] = []
    lookback_blocks = DEFAULT_BLOCK_LOOKBACK
    chains: Optional[List[str]] = None
    contracts: Optional[List[str]] = None
    if isinstance(goal, dict):
        if isinstance(goal.get("addresses"), list):
            addresses = goal["addresses"]
        if isinstance(goal.get("lookback_blocks"), int):
            lookback_blocks = goal["lookback_blocks"]
        if isinstance(goal.get("chains"), list):
            chains = goal["chains"]
        if isinstance(goal.get("contracts"), list):
            contracts = goal["contracts"]
    elif isinstance(goal, str):
        addresses = [goal]
    else:
        log_message("gather_onchain_data_for_goal: No addresses provided in goal.", level='info')
        return {}
    if not addresses:
        log_message("gather_onchain_data_for_goal: Empty address list.", level='info')
        return {}
    result = run_onchain_scraper_rpc(addresses=addresses, lookback_blocks=lookback_blocks, chains=chains, contracts=contracts)
    return {"status": "success", "data": result, "count": len(addresses)}

# =========================
# CLI Test
# =========================

if __name__ == "__main__":
    test_addresses = [
        "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        "11111111111111111111111111111111"
    ]
    out = run_onchain_scraper_rpc(test_addresses, lookback_blocks=1500)
    print(json.dumps(out, indent=2)[:2000] + "\n...\n") 
