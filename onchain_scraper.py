# onchain_scraper.py
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

import requests

# =========================
# Config & Output
# =========================

OUTPUT_DIR = Path("ai_core/onchain_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Public RPC defaults (override via env if desired)
ETH_RPC = os.getenv("ETH_RPC", "https://cloudflare-eth.com")
ARBITRUM_RPC = os.getenv("ARBITRUM_RPC", "https://arb1.arbitrum.io/rpc")
SOLANA_RPC = os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com")

# Default lookback and chunk size
DEFAULT_BLOCK_LOOKBACK = int(os.getenv("BLOCK_LOOKBACK", "2000"))  # ~8–10 mins on Ethereum
DEFAULT_CHUNK_SIZE = int(os.getenv("BLOCK_CHUNK_SIZE", "1000"))    # chunk requests to avoid RPC size errors

# ERC-20 Transfer topic signature
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


# =========================
# Utilities
# =========================

def _now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _rpc_call(url: str, method: str, params: list) -> Any:
    headers = {"Content-Type": "application/json"}
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    resp = requests.post(url, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"RPC error from {url}: {data['error']}")
    return data.get("result")


def _hex_to_int(hex_str: str) -> int:
    return int(hex_str, 16)


def _pad_address_topic(address: str) -> str:
    """
    Convert 0xabc... address to 32-byte topic form:
    0x000000000000000000000000<lower_40_hex>
    """
    addr = address.lower()
    if not addr.startswith("0x") or len(addr) != 42:
        raise ValueError(f"Invalid EVM address: {address}")
    return "0x" + ("0" * 24) + addr[2:]


def _save_json(data: Any, source: str, address: str) -> Path:
    filename = OUTPUT_DIR / f"{source}_{address}_{_now_stamp()}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[SAVE] {source} -> {filename}")
    return filename


def _looks_like_solana_address(addr: str) -> bool:
    # Naive heuristic: Base58-ish, no 0x prefix, length typically 32–44
    return (not addr.startswith("0x")) and (30 <= len(addr) <= 50)


def _chunk_ranges(start: int, end: int, step: int) -> Iterable[tuple[int, int]]:
    """
    Yield inclusive [a, b] chunk ranges.
    """
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
    return _hex_to_int(latest_hex)


def _evm_fetch_logs_with_chunking(
    rpc_url: str,
    base_filter: Dict[str, Any],
    from_block: int,
    to_block: int,
    initial_chunk: int
) -> List[Dict[str, Any]]:
    """
    Query eth_getLogs in chunks. If an RPC node rejects due to response size,
    automatically halves chunk size down to a small floor.
    """
    results: List[Dict[str, Any]] = []
    chunk = max(1, initial_chunk)
    floor = 64  # don't go below this unless absolutely necessary

    # We may dynamically adjust chunk size if we hit errors
    a = from_block
    while a <= to_block:
        b = min(a + chunk - 1, to_block)
        params = [dict(base_filter, **{"fromBlock": hex(a), "toBlock": hex(b)})]
        try:
            logs = _rpc_call(rpc_url, "eth_getLogs", params) or []
            results.extend(logs)
            a = b + 1  # advance window
        except Exception as e:
            # Reduce chunk size and retry the same window
            if chunk <= floor:
                # One last attempt with single-block if really necessary
                if chunk == 1:
                    raise
                chunk = 1
            else:
                chunk = max(floor, chunk // 2)
            print(f"[WARN] {rpc_url} getLogs chunk {a}-{b} failed ({e}); reducing chunk to {chunk} and retrying...")
            # Do not advance 'a' here; retry with smaller chunk

    return results


def _evm_get_erc20_transfers_for_address(
    rpc_url: str,
    address: str,
    lookback_blocks: int,
    contracts: Optional[List[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> List[Dict[str, Any]]:
    """
    Query ERC‑20 Transfer logs involving `address` over the last `lookback_blocks`.
    If `contracts` are provided, we request per-contract (much cheaper & more reliable).
    Otherwise, we do a broad topic-only scan (heavier)—keep lookback small.
    """
    latest = _evm_latest_block(rpc_url)
    from_block = max(latest - lookback_blocks, 0)

    addr_topic = _pad_address_topic(address)

    all_logs: List[Dict[str, Any]] = []

# --- Discovery: contracts & wallets (pure RPC, bounded) ---

def bootstrap_contracts_via_receipts(
    chain: str = "ethereum",
    blocks: int = 10,          # sample this many latest blocks
    max_tx: int = 400,         # cap total receipts to inspect
    top_k: int = 15            # return top K token contracts by transfer-log count
) -> List[str]:
    """
    Pure-RPC bootstrap: scan a small slice of recent blocks, fetch receipts,
    extract ERC-20 Transfer logs, and return the most active token contracts.
    This avoids broad eth_getLogs queries that public nodes reject.
    """
    rpcs = ETH_RPCS if chain == "ethereum" else ARBITRUM_RPCS
    latest = _evm_latest_block(rpcs)

    contract_counts: Dict[str, int] = {}
    tx_seen = 0
    start_block = max(0, latest - blocks + 1)

    for bn in range(latest, start_block - 1, -1):
        # get full block (with txs)
        block, used = _rpc_call_any(rpcs, "eth_getBlockByNumber", [hex(bn), True])
        txs = block.get("transactions", [])
        if not txs:
            continue

        for tx in txs:
            if tx_seen >= max_tx:
                break
            tx_hash = tx.get("hash")
            if not tx_hash:
                continue
            # get receipt for precise logs; cheap per-tx call
            try:
                receipt, used2 = _rpc_call_any(rpcs, "eth_getTransactionReceipt", [tx_hash])
                for lg in receipt.get("logs", []):
                    topics = lg.get("topics", [])
                    if topics and topics[0].lower() == TRANSFER_TOPIC:
                        caddr = lg.get("address")
                        if caddr:
                            contract_counts[caddr] = contract_counts.get(caddr, 0) + 1
            except Exception as e:
                # skip problematic tx; keep moving
                print(f"[WARN] receipt fetch failed {tx_hash} on {used2 if 'used2' in locals() else used}: {e}")
            tx_seen += 1

        if tx_seen >= max_tx:
            break

    top = sorted(contract_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    contracts = [c for c, _ in top]
    print(f"[DISCOVERY] {chain}: sampled blocks {start_block}..{latest}, tx={tx_seen}, top_contracts={len(contracts)}")
    return contracts


def discover_active_wallets_from_contracts(
    chain: str = "ethereum",
    contracts: List[str] = None,
    lookback_blocks: int = 900,
    top_k_wallets: int = 50
) -> List[str]:
    """
    For the given token contracts, scan ERC-20 Transfer logs over a short window,
    collect 'from' and 'to' addresses, and return the most active wallets.
    Uses address-filtered eth_getLogs (accepted by public nodes).
    """
    if not contracts:
        return []

    rpcs = ETH_RPCS if chain == "ethereum" else ARBITRUM_RPCS
    latest = _evm_latest_block(rpcs)
    from_block = max(latest - lookback_blocks, 0)
    addr_counts: Dict[str, int] = {}

    # query per chunk of contracts to be gentle on public RPCs
    BATCH = 8
    for i in range(0, len(contracts), BATCH):
        batch = contracts[i:i+BATCH]
        base = {
            "address": batch,               # CONTRACT FILTER HERE
            "topics": [TRANSFER_TOPIC],     # any ERC-20 Transfer
        }
        try:
            logs = _evm_fetch_logs_with_chunking(
                rpcs=rpcs,
                base_filter=base,
                from_block=from_block,
                to_block=latest,
                initial_chunk=DEFAULT_CHUNK_SIZE,
            )
        except Exception as e:
            print(f"[WARN] {chain} discover wallets failed for batch {i//BATCH}: {e}")
            continue

        for lg in logs:
            t = lg.get("topics", [])
            if len(t) >= 3:
                frm = "0x" + t[1][-40:] if t[1].startswith("0x") else None
                to  = "0x" + t[2][-40:] if t[2].startswith("0x") else None
                if frm:
                    addr_counts[frm] = addr_counts.get(frm, 0) + 1
                if to:
                    addr_counts[to] = addr_counts.get(to, 0) + 1

    # Helper to run 'from' and 'to' topic filters (optionally per-contract)

    def run_filters(contract_list: Optional[List[str]]):
        nonlocal all_logs
        address_field = contract_list if contract_list else None

        # As 'from'
        base_from = {
            "topics": [TRANSFER_TOPIC, addr_topic],
        }
        if address_field:
            base_from["address"] = address_field

        logs_from = _evm_fetch_logs_with_chunking(
            rpc_url=rpc_url,
            base_filter=base_from,
            from_block=from_block,
            to_block=latest,
            initial_chunk=chunk_size,
        )
        all_logs.extend(logs_from)

        # As 'to'
        base_to = {
            "topics": [TRANSFER_TOPIC, None, addr_topic],
        }
        if address_field:
            base_to["address"] = address_field

        logs_to = _evm_fetch_logs_with_chunking(
            rpc_url=rpc_url,
            base_filter=base_to,
            from_block=from_block,
            to_block=latest,
            initial_chunk=chunk_size,
        )
        all_logs.extend(logs_to)

    if contracts and len(contracts) > 0:
        # Run per-contract in batches to avoid nodes rejecting multi-address overloads
        BATCH = 8
        for i in range(0, len(contracts), BATCH):
            run_filters(contracts[i:i+BATCH])
    else:
        # Broad scan (heavier)
        run_filters(None)

    # Normalize
    transfers = []
    for log in all_logs:
        topics = log.get("topics", [])
        data_hex = log.get("data", "0x0")

        # Value is in 'data' for standard ERC‑20 Transfer(address,address,uint256)
        try:
            value = _hex_to_int(data_hex) if data_hex and data_hex != "0x" else 0
        except Exception:
            value = 0

        from_addr = None
        to_addr = None
        if len(topics) >= 3:
            # topics[1] and topics[2] are 32‑byte padded addresses
            from_addr = "0x" + topics[1][-40:] if topics[1].startswith("0x") else None
            to_addr = "0x" + topics[2][-40:] if topics[2].startswith("0x") else None

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

    # Sort by blockNumber/logIndex ascending
    transfers.sort(key=lambda x: (x["blockNumber"] or 0, x["logIndex"] or 0))
    return transfers


def fetch_evm_onchain_for_address(
    address: str,
    lookback_blocks: int,
    contracts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Fetch ERC‑20 transfer activity on Ethereum and Arbitrum for the given address.
    """
    if not address.startswith("0x"):
        return {}

    out: Dict[str, Any] = {"ethereum": {}, "arbitrum": {}}

    try:
        eth_transfers = _evm_get_erc20_transfers_for_address(
            ETH_RPC, address, lookback_blocks, contracts
        )
        out["ethereum"] = {"erc20_transfers": eth_transfers}
    except Exception as e:
        print(f"[WARN] Ethereum RPC failed for {address}: {e}")

    try:
        arb_transfers = _evm_get_erc20_transfers_for_address(
            ARBITRUM_RPC, address, lookback_blocks, contracts
        )
        out["arbitrum"] = {"erc20_transfers": arb_transfers}
    except Exception as e:
        print(f"[WARN] Arbitrum RPC failed for {address}: {e}")

    return out


# =========================
# Solana
# =========================

def fetch_solana_activity(address: str, limit: int = 25) -> Dict[str, Any]:
    """
    Fetch recent signatures for a Solana address.
    For richer details, follow up with getParsedTransaction per signature.
    """
    if not _looks_like_solana_address(address):
        return {}

    headers = {"Content-Type": "application/json"}
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [address, {"limit": limit}]
    }
    try:
        resp = requests.post(SOLANA_RPC, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        result = resp.json().get("result", [])
        signatures = [{
            "signature": r.get("signature"),
            "slot": r.get("slot"),
            "err": r.get("err"),
            "blockTime": r.get("blockTime")
        } for r in result]
        return {"signatures": signatures}
    except Exception as e:
        print(f"[WARN] Solana RPC failed for {address}: {e}")
        return {}


# =========================
# Save & Orchestrate
# =========================

def run_onchain_scraper_rpc(
    addresses: List[str],
    lookback_blocks: int = DEFAULT_BLOCK_LOOKBACK,
    chains: Optional[List[str]] = None,
    contracts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    For each address, collect EVM (ETH + ARB) and/or Solana activity and save snapshots.
    Returns an in-memory aggregate too.

    chains: subset of {"ethereum", "arbitrum", "solana"} or None for all.
    contracts: optional list of ERC‑20 token contract addresses to filter logs (recommended).
    """
    chosen = set(chains or ["ethereum", "arbitrum", "solana"])
    aggregate: Dict[str, Any] = {}

    for addr in addresses:
        record: Dict[str, Any] = {}

        if {"ethereum", "arbitrum"} & chosen and addr.startswith("0x"):
            evm_data = fetch_evm_onchain_for_address(addr, lookback_blocks, contracts)
            if evm_data:
                record.update(evm_data)
                _save_json(evm_data, "evm", addr)

        if "solana" in chosen and _looks_like_solana_address(addr):
            sol_data = fetch_solana_activity(addr)
            if sol_data:
                record["solana"] = sol_data
                _save_json(sol_data, "solana", addr)

        aggregate[addr] = record

    return aggregate


# =========================
# Agent Entry Point
# =========================

def gather_onchain_data_for_goal(goal: Any) -> Dict[str, Any]:
    """
    Entry point expected by agent.py.

    `goal` can be:
      - dict with keys:
          addresses: List[str]                # required
          lookback_blocks: int                # optional override
          chains: List[str]                   # optional subset of {"ethereum","arbitrum","solana"}
          contracts: List[str]                # optional ERC-20 contract list for EVM filter
      - a single address string
      - or any other -> returns {}

    Returns:
      {"status":"success","data":{address: {...}}, "count": N}
    """
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
        print("[INFO] gather_onchain_data_for_goal: No addresses provided in goal.")
        return {}

    if not addresses:
        print("[INFO] gather_onchain_data_for_goal: Empty address list.")
        return {}

    result = run_onchain_scraper_rpc(
        addresses=addresses,
        lookback_blocks=lookback_blocks,
        chains=chains,
        contracts=contracts,
    )
    return {"status": "success", "data": result, "count": len(addresses)}


# =========================
# CLI Test
# =========================

if __name__ == "__main__":
    # Replace with your own test addresses
    test_addresses = [
        # EVM example (random known ETH address)
        "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        # Solana example (System Program, signatures may be sparse)
        "11111111111111111111111111111111"
    ]
    out = run_onchain_scraper_rpc(test_addresses, lookback_blocks=1500)
    print(json.dumps(out, indent=2)[:2000] + "\n...\n")
 

 

