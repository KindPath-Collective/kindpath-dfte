# Syntropy Trajectory Score (STS)
## Theoretical Addition to DFTE — KindPath Trading Engine

**Evidence posture: [TESTABLE]** — trajectory as predictive signal requires
validation against outcomes before promotion to ESTABLISHED.

---

## The Core Problem

The current KEPE architecture computes a World Field Score (WFS) as a point-in-time
scalar. A WFS of 0.35 is treated identically whether it was 0.55 three runs ago
(field deteriorating) or 0.20 three runs ago (field recovering).

**WFS=0.35 rising is fundamentally different from WFS=0.35 falling.**

The current architecture cannot distinguish these two states. This creates a
timing blind spot: a position sized for a "loaded" world field may be entered
just as the field tips into deterioration, or avoided precisely when the world
field is beginning its recovery arc.

The STS resolves the **here-and-now vs future syntropy tension** by adding
a directional dimension to the scalar WFS reading.

---

## Definition

The **Syntropy Trajectory Score (STS)** measures the rate of change and
direction of WFS over the last three readings per instrument.

### Three States

| State | Condition | Meaning |
|-------|-----------|---------|
| `LOADING` | Net WFS change > +0.03 over last 3 readings | World field gaining syntropy |
| `DETERIORATING` | Net WFS change < −0.03 over last 3 readings | World field losing coherence |
| `STABLE` | Net WFS change within ±0.03 | World field in compression or equilibrium |

The threshold of ±0.03 is a provisional calibration value. Given WFS is
bounded [0, 1] and typically moves in small increments, 0.03 represents a
meaningful shift (~3 percentage points) without being noise-sensitive.
**[TESTABLE] — threshold requires empirical calibration against outcome data.**

### Slope Computation

With history `[w₁, w₂, w₃]`, the trajectory delta is:

```
δ = w₃ - w₁
```

Simple endpoint-to-endpoint delta over the three-reading window is used in
preference to linear regression because:
1. The window is too short (n=3) for regression to add signal over noise
2. The direction of the most recent transition is what matters for positioning

---

## 2×2 Positioning Matrix

Combined with the instrument's **governance category** (from `classify_symbol()`),
STS produces a positioning label that guides sizing and timing decisions.

```
                    │  LOADING    │  STABLE     │  DETERIORATING
────────────────────┼─────────────┼─────────────┼────────────────
Syntropic asset     │ ZPB_LOADING │ COMPRESSION │ REVIEW
Neutral asset       │ EMERGING    │ RANGE       │ FADING
Extractive asset    │  BLOCKED    │  BLOCKED    │  BLOCKED
```

### Position Labels

**`ZPB_LOADING`** — Syntropic + LOADING
The world field is rising toward the asset's natural operating domain.
*Guidance:* Early position, size up within tier limits. The field is building
the conditions the instrument benefits from. This is the primary ZPB signal.

**`COMPRESSION`** — Syntropic + STABLE
Field is flat. Asset is in a syntropy compression zone — conditions are present
but not yet propagating. *Guidance:* Watch for ν recovery as the trigger. Hold
existing positions; wait for LOADING to re-enter.

**`REVIEW`** — Syntropic + DETERIORATING
Tension between the asset's syntropic category and the current field direction.
The instrument's long-term alignment with KindPath values is intact, but the
world field is moving against it in the medium term.
*Guidance:* Do not add. Consider whether deterioration is temporary (cyclical
macro headwind) or structural (sector narrative shift). Existing positions may
remain if MFS confirms.

**`EMERGING`** — Neutral + LOADING
A non-classified instrument whose world field is rising. Could indicate sector
rotation or macro tailwind not yet captured in classification.
*Guidance:* Monitor. Do not size aggressively until STS confirms sustained
LOADING over ≥2 runs.

**`FADING`** — Neutral + DETERIORATING
World field softening for an unclassified instrument.
*Guidance:* Reduce or avoid. No KindPath equity case to hold through field
deterioration.

**`RANGE`** — Neutral + STABLE
Flat world field, no syntropic thesis. Purely market-mechanics-driven.
*Guidance:* DFTE falls back to BMR/ν as primary selector.

**`BLOCKED`** — Extractive (any trajectory)
Extractive assets are blocked from LARGE tier regardless of trajectory.
A rising world field does not rehabilitate an extractive instrument.

---

## Integration with DFTE

STS is computed in `syntropy_engine.synthesise_kepe_profile()` and flows
through the signal chain:

```
KEPEProfile.sts / .sts_position / .wfs_history
    → KEPESummary.sts / .sts_position
        → DFTESignal.sts / .sts_position
            → dashboard (STS column)
```

WFS history is persisted per-symbol in `/tmp/kepe_cache/wfs_history_{symbol}.json`
(last 5 readings). This is ephemeral storage — history resets on machine restart.
**Phase 2 work:** Migrate to durable storage for cross-session trajectory tracking.

---

## Phase 2 Intentions

The STS in its current form is a first-order trajectory signal. Future work:

1. **STS velocity** — second derivative (is LOADING accelerating or decelerating?)
2. **Cross-instrument STS coherence** — are multiple instruments in the same
   basket all LOADING simultaneously? Coherent loading = stronger ZPB signal.
3. **STS-gated sizing** — explicit position size multipliers by STS_POSITION label
   (e.g., ZPB_LOADING gets 1.2× size, REVIEW gets 0.5×, FADING gets 0.0×)
4. **Durable history** — persist WFS readings across sessions for multi-day
   trajectory computation
5. **Calibration study** — backtest STS_POSITION labels against 5-day forward
   returns to validate [TESTABLE] → [ESTABLISHED] promotion criteria

---

## Evidence Posture

| Component | Level | Basis |
|-----------|-------|-------|
| WFS trajectory direction as signal | [TESTABLE] | Directionally valid; calibration against forward returns required |
| ±0.03 LOADING/DETERIORATING threshold | [TESTABLE] | Provisional; no empirical calibration yet |
| Syntropic + LOADING = better entry timing | [SPECULATIVE] | Theoretically coherent; no backtesting |
| 3-reading window for trajectory | [TESTABLE] | Short window; may be noise-sensitive in volatile regimes |
| ZPB_LOADING position guidance | [SPECULATIVE] | Derived from KindPath field theory; requires outcome validation |

*In KindPath doctrine terms: the STS operationalises the E = [(Me × Community × Country) · ν]²
formulation at the temporal dimension — ν (the coherence coefficient) is not
static but has a direction. A rising WFS is the world field's ν increasing.*
