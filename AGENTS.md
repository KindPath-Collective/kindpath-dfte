# AI Agent Rules for kindpath-dfte

## Session Init Protocol

Before reading code or making changes, run:
```bash
cat ~/.kindpath/HANDOVER.md
python3 ~/.kindpath/kp_memory.py dump --domain gotcha
python3 ~/.kindpath/kp_memory.py dump
```

---

## What This Is

KindPath Dual Field Trading Engine (DFTE) — models fair, benevolent economic
exchange using KindPath signal fields. Includes backtesting, governance,
KEPE (KindPath Economic Pressure Estimator), and wallet management.

## Structure

```
orchestrator.py         — Main entry point, full pipeline coordination
dfte/                   — Core DFTE engine and KPRE modules
kepe/                   — KindPath Economic Pressure Estimator
backtest/               — Backtesting simulators and price data fetching
governance/             — Governance layer for ethical constraints
wallet/                 — Wallet state management
logger/                 — Signal logger (SQLite, not tracked in git)
```

## Operational Commands

- **Install**: `pip install -r requirements.txt`
- **Run**: `python orchestrator.py`
- **Test**: `pytest`
- **Docker**: `docker build -t kindpath-dfte . && docker run --env-file .env kindpath-dfte`

## Rules

- Never commit `.db`, `.pkl`, or `*_wallet_state.json` files — these are runtime data
- Governance layer in `governance/governance_layer.py` must be respected — do not bypass
- KEPE signals represent real economic pressures — document assumptions clearly
- Backtests use synthetic data unless explicitly connecting to live feeds
- Follow KindPath doctrine: benevolence, syntropy, sovereignty

## Security Mandates

- No wallet keys, API keys, or real financial data in source control
- All credentials via `.env` (local) or Cloud Run environment variables
- Never commit `.env` files
