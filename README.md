# KindPath DFTE — Dual Field Trading Engine

```
M = [(Participant × Institutional × Sovereign) · ν]²     ← BMR
E = [(Me × Community × Country) · ν]²                    ← KEPE
Trade = f(MFS, WFS) | governance gate | execution
```

The first syntropy-governed, benevolence-first trading engine.
A trade only executes when both the price field (BMR) and the world field (KEPE) align.

## Architecture

```
bmr/
  bmr_server.py        — FastAPI signal server (BMR pipeline REST API)
  core/                — Normaliser, ν engine, LSII, curvature, profile synthesiser
  feeds/               — Raw data ingestors (market data, COT, options, macro)
  tests/               — BMR unit tests

kepe/
  indicators.py        — World field data feeds (ecological, social, narrative, macro)
  syntropy_engine.py   — SPI, WFS, OPC, EI, IL, UCS doctrine computations

dfte/
  dfte_engine.py       — Unified signal synthesis + gate logic + position sizing

governance/
  governance_layer.py  — Benevolence scoring, contradiction detection, influence tracking

wallet/
  wallet.py            — PaperWallet + AlpacaWallet execution abstraction

orchestrator.py        — Main entry point: KEPE + BMR → DFTE → governance → wallet
tests/
  test_dfte.py         — 46/46 passing test suite
```

## Running

Start the BMR signal server:
```bash
cd bmr && python bmr_server.py
```

Then run DFTE:
```bash
python orchestrator.py --symbols SPY QQQ GLD BTC-USD
python orchestrator.py --symbols ICLN NEE ENPH --mode paper  # syntropic basket
python orchestrator.py --symbols SPY --execute --mode paper   # paper execute
python orchestrator.py --watch --symbols SPY QQQ GLD          # continuous loop
```

## Trade Tier Logic

| Tier  | MFS req | WFS req | ν req  | Notes |
|-------|---------|---------|--------|-------|
| NANO  | any     | any     | any    | Pure market physics. Extractive blocked. |
| MID   | ≥ 0.50  | ≥ 0.35  | ≥ 0.35 | WFS modulates MFS conviction |
| LARGE | ≥ 0.65  | ≥ 0.55  | ≥ 0.55 | Impact-first. Syntropic preferred. Extractive blocked. |
| WAIT  | —       | —       | —      | Field incoherent or SIC event |

## Doctrine Mappings (from Copilot brief)

| KindPath Concept | Computation |
|-----------------|-------------|
| IN (Insecure Neutrality) | Entropy Indicator (EI) |
| ZPB | Syntropy Potential Index (SPI) |
| Curvature | Field Gradient Tensor (FGT) + Market Curvature Index (K) |
| Contradiction | Interference Load (IL) |
| Placebo of kindness | Optimism Propagation Coefficient (OPC) |
| System coherence | Unified Curvature Score (UCS) |

## The Feedback Loop

Trades → participant-scale signal in BMR → updated ν → refined MFS
At scale: coordinated impact-first investment creates coherence in syntropic assets.
The system doesn't just read the field. It contributes to the field it reads.
