# Flash loan attack vectors (Defensive Overview)

> Purpose: Defensive knowledge for detection, prevention, and incident response. No exploit instructions.

## What is a flash loan?
Flash loans are uncollateralized loans executed within a single transaction. If not repaid by the end of the transaction, the operation reverts. They enable capital-intensive, atomic strategies but also create attack surfaces when protocols have logic that can be manipulated within the same block.

## Typical attack paths (high-level)
- **Price oracle manipulation**: Temporarily distort on-chain price sources (AMMs or “soft” oracles) to affect collateral valuations or swap rates.
- **Debt/Collateral mispricing**: Exploit lagging or manipulable prices to borrow against inflated collateral or repay at deflated values.
- **Reentrancy + intra-tx state assumptions**: Abuse callable hooks or state transitions that assume inter-block stability, not intra-tx changes.
- **Liquidity pool imbalances**: Create artificial imbalances to drain rewards, skew TWAP windows, or trigger liquidation cascades.
- **Governance race conditions**: Use flash voting power (if allowed) to pass malicious proposals or parameter changes within the same block.
- **Fee or rounding edge cases**: Exploit precision, rounding, or fee accounting to siphon value when volumes spike intra-tx.

## Defensive patterns
- **Use robust oracles**: Prefer time-weighted, manipulation-resistant oracles with sufficient lookback, anchor checks, and cross-source validation.
- **Block-based vs tx-based assumptions**: Avoid logic that assumes prices or balances are stable throughout a transaction.
- **Reentrancy & callback controls**: Guard external calls; use checks-effects-interactions, reentrancy guards, and limit external hooks during sensitive operations.
- **Invariant checks**: Assert economic invariants at function end (post-conditions) to detect intra-tx manipulation before state is finalized.
- **Slippage and bounds**: Enforce conservative slippage, liquidity, and price bounds; reject operations outside safe envelopes.
- **Rate limits / circuit breakers**: Introduce per-block caps or circuit breakers on sensitive actions (mint/burn/borrow/liquidate) triggered by volatility.
- **Governance safeguards**: Prevent flash-inflated voting; require timelocks, quorum based on long-term balances, and multi-block snapshots.

## Detection & monitoring signals
- Unusual **single-block** spikes in borrowed capital concurrent with large swaps.
- Divergence between local AMM price and **robust oracle** price beyond a configured threshold.
- Sudden **TWAP deviations** or short-window manipulation patterns.
- Clusters of liquidations triggered within the same block.
- Anomalous gas usage spikes tied to tightly sequenced DeFi leg calls.

## Testing & simulation
- **Adversarial unit tests** for intra-tx price changes, reentrancy, and boundary conditions.
- **Property-based tests** to search for invariant breaks under large capital swings.
- **Fork tests** against mainnet snapshots to reproduce real-world multi-leg sequences.

## Incident response checklist (high-level)
- Freeze affected functions via pause / circuit breaker (if built-in).
- Snapshot state, extract traces, quantify exposure, and identify the manipulated leg.
- Coordinate with oracle providers, exchanges, and impacted protocols.
- Publish transparent postmortem and roll out patches with audits.

## Key terms
- **TWAP**: Time-Weighted Average Price
- **Atomicity**: All-or-nothing transaction execution in a single block
# Flash loan attack vectors (Defensive Overview)

> Purpose: Defensive knowledge for detection, prevention, and incident response. No exploit instructions.

## What is a flash loan?
Flash loans are uncollateralized loans executed within a single transaction. If not repaid by the end of the transaction, the operation reverts. They enable capital-intensive, atomic strategies but also create attack surfaces when protocols have logic that can be manipulated within the same block.

## Typical attack paths (high-level)
- **Price oracle manipulation**: Temporarily distort on-chain price sources (AMMs or “soft” oracles) to affect collateral valuations or swap rates.
- **Debt/Collateral mispricing**: Exploit lagging or manipulable prices to borrow against inflated collateral or repay at deflated values.
- **Reentrancy + intra-tx state assumptions**: Abuse callable hooks or state transitions that assume inter-block stability, not intra-tx changes.
- **Liquidity pool imbalances**: Create artificial imbalances to drain rewards, skew TWAP windows, or trigger liquidation cascades.
- **Governance race conditions**: Use flash voting power (if allowed) to pass malicious proposals or parameter changes within the same block.
- **Fee or rounding edge cases**: Exploit precision, rounding, or fee accounting to siphon value when volumes spike intra-tx.

## Defensive patterns
- **Use robust oracles**: Prefer time-weighted, manipulation-resistant oracles with sufficient lookback, anchor checks, and cross-source validation.
- **Block-based vs tx-based assumptions**: Avoid logic that assumes prices or balances are stable throughout a transaction.
- **Reentrancy & callback controls**: Guard external calls; use checks-effects-interactions, reentrancy guards, and limit external hooks during sensitive operations.
- **Invariant checks**: Assert economic invariants at function end (post-conditions) to detect intra-tx manipulation before state is finalized.
- **Slippage and bounds**: Enforce conservative slippage, liquidity, and price bounds; reject operations outside safe envelopes.
- **Rate limits / circuit breakers**: Introduce per-block caps or circuit breakers on sensitive actions (mint/burn/borrow/liquidate) triggered by volatility.
- **Governance safeguards**: Prevent flash-inflated voting; require timelocks, quorum based on long-term balances, and multi-block snapshots.

## Detection & monitoring signals
- Unusual **single-block** spikes in borrowed capital concurrent with large swaps.
- Divergence between local AMM price and **robust oracle** price beyond a configured threshold.
- Sudden **TWAP deviations** or short-window manipulation patterns.
- Clusters of liquidations triggered within the same block.
- Anomalous gas usage spikes tied to tightly sequenced DeFi leg calls.

## Testing & simulation
- **Adversarial unit tests** for intra-tx price changes, reentrancy, and boundary conditions.
- **Property-based tests** to search for invariant breaks under large capital swings.
- **Fork tests** against mainnet snapshots to reproduce real-world multi-leg sequences.

## Incident response checklist (high-level)
- Freeze affected functions via pause / circuit breaker (if built-in).
- Snapshot state, extract traces, quantify exposure, and identify the manipulated leg.
- Coordinate with oracle providers, exchanges, and impacted protocols.
- Publish transparent postmortem and roll out patches with audits.

## Key terms
- **TWAP**: Time-Weighted Average Price
- **Atomicity**: All-or-nothing transaction execution in a single block
# Flash loan attack vectors (Defensive Overview)

> Purpose: Defensive knowledge for detection, prevention, and incident response. No exploit instructions.

## What is a flash loan?
Flash loans are uncollateralized loans executed within a single transaction. If not repaid by the end of the transaction, the operation reverts. They enable capital-intensive, atomic strategies but also create attack surfaces when protocols have logic that can be manipulated within the same block.

## Typical attack paths (high-level)
- **Price oracle manipulation**: Temporarily distort on-chain price sources (AMMs or “soft” oracles) to affect collateral valuations or swap rates.
- **Debt/Collateral mispricing**: Exploit lagging or manipulable prices to borrow against inflated collateral or repay at deflated values.
- **Reentrancy + intra-tx state assumptions**: Abuse callable hooks or state transitions that assume inter-block stability, not intra-tx changes.
- **Liquidity pool imbalances**: Create artificial imbalances to drain rewards, skew TWAP windows, or trigger liquidation cascades.
- **Governance race conditions**: Use flash voting power (if allowed) to pass malicious proposals or parameter changes within the same block.
- **Fee or rounding edge cases**: Exploit precision, rounding, or fee accounting to siphon value when volumes spike intra-tx.

## Defensive patterns
- **Use robust oracles**: Prefer time-weighted, manipulation-resistant oracles with sufficient lookback, anchor checks, and cross-source validation.
- **Block-based vs tx-based assumptions**: Avoid logic that assumes prices or balances are stable throughout a transaction.
- **Reentrancy & callback controls**: Guard external calls; use checks-effects-interactions, reentrancy guards, and limit external hooks during sensitive operations.
- **Invariant checks**: Assert economic invariants at function end (post-conditions) to detect intra-tx manipulation before state is finalized.
- **Slippage and bounds**: Enforce conservative slippage, liquidity, and price bounds; reject operations outside safe envelopes.
- **Rate limits / circuit breakers**: Introduce per-block caps or circuit breakers on sensitive actions (mint/burn/borrow/liquidate) triggered by volatility.
- **Governance safeguards**: Prevent flash-inflated voting; require timelocks, quorum based on long-term balances, and multi-block snapshots.

## Detection & monitoring signals
- Unusual **single-block** spikes in borrowed capital concurrent with large swaps.
- Divergence between local AMM price and **robust oracle** price beyond a configured threshold.
- Sudden **TWAP deviations** or short-window manipulation patterns.
- Clusters of liquidations triggered within the same block.
- Anomalous gas usage spikes tied to tightly sequenced DeFi leg calls.

## Testing & simulation
- **Adversarial unit tests** for intra-tx price changes, reentrancy, and boundary conditions.
- **Property-based tests** to search for invariant breaks under large capital swings.
- **Fork tests** against mainnet snapshots to reproduce real-world multi-leg sequences.

## Incident response checklist (high-level)
- Freeze affected functions via pause / circuit breaker (if built-in).
- Snapshot state, extract traces, quantify exposure, and identify the manipulated leg.
- Coordinate with oracle providers, exchanges, and impacted protocols.
- Publish transparent postmortem and roll out patches with audits.

## Key terms
- **TWAP**: Time-Weighted Average Price
- **Atomicity**: All-or-nothing transaction execution in a single block
