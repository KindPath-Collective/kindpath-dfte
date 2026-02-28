# KindPath DFTE — Dual Field Trading Engine

Synthesises two independent field readings into unified trade signals.

```
KEPE (World Field Score) × BMR (Market Field Score) → DFTE → Trade
```

**The core insight:** MFS without WFS = technical without ethics. WFS without MFS = impact without timing. Both fields must agree before a position is taken.

## Architecture

```
kepe/
  indicators.py       — World field data feeds (ecological, social, narrative, macro)
  syntropy_engine.py  — SPI, WFS, OPC, EI, FGT, IL, UCS synthesis

dfte/
  dfte_engine.py      — Dual field signal synthesis + trade decision + sizing

governance/
  governance_layer.py — Benevolence scoring, contradiction detection, influence tracking

wallet/
  wallet.py           — PaperWallet + AlpacaWallet execution layer

orchestrator.py       — Full pipeline runner + rich terminal dashboard
tests/test_dfte.py    — 46/46 passing
```

## Quick Start

```bash
# 1. Start BMR signal server (in kindpath-bmr repo)
cd ../kindpath-bmr && python bmr_server.py

# 2. Run DFTE analysis (paper mode, no credentials needed)
cd kindpath-dfte
pip install -r requirements.txt
python orchestrator.py --symbols SPY QQQ GLD BTC-USD

# 3. Syntropic asset basket
python orchestrator.py --symbols ICLN NEE ENPH FSLR

# 4. Continuous watch mode
python orchestrator.py --symbols SPY GLD --watch
```

## Trade Tiers

| Tier  | Requires | Governed by |
|-------|----------|-------------|
| NANO  | ν > 0, not SIC | Market physics only. No extractive assets. |
| MID   | ν ≥ 0.35, WFS ≥ 0.35 | Both fields present. IL < 0.65. |
| LARGE | ν ≥ 0.55, WFS ≥ 0.55, SPI ≥ 0.45 | Impact-first. Syntropic preferred. Extractive blocked. |

## Governance

Every trade gates through:
1. **MFS Gate** — market field coherence (ν threshold per tier)
2. **WFS Gate** — world field quality (WFS + SPI threshold per tier)
3. **Governance Gate** — benevolence score, extractive block, contradiction check

Extractive assets (weapons, fossil fuel, private prison, tobacco, predatory lending) are permanently blocked from LARGE tier and flagged in all tiers.

## The Feedback Loop

DFTE doesn't just read the field. At scale, its positions contribute to the field it reads.
Large impact-first positions in syntropic sectors reinforce participant-layer signals in BMR.
This is the KindPath benevolence propagation mechanism operating through capital.

## Integration

BMR Server: `http://localhost:8001` (configurable via `--bmr-server`)
Alpaca (paper): Set `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`
FRED (macro):   Set `FRED_API_KEY` (free at fred.stlouisfed.org)
