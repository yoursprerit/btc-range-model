# Adding more commodity sleeves to the Overall strategy — evaluation

**Question.** The Overall Trading universe already carries three commodity
signals — 🥇 gold (GLDM · GDX · UGL · NUGT), 🛢️ energy (XLE · OIH · ERX) and
🧲 strategic metals (REMX). Would adding **any other commodity app** — silver,
copper, uranium, platinum, palladium, agriculture, broad-commodity, natural
gas, lithium or diversified miners — boost the combined portfolio's return and
performance?

**Answer up front: no. Every one of the twelve candidates dilutes the book as
its own app.** The only instrument that survives the noise-free tests is
**AGQ (2× silver) traded off the existing gold signal** — a small,
Balanced-profile-only win (see §5). The earlier finding
([`SOXL_ERX_ADDITION_EVAL.md`](SOXL_ERX_ADDITION_EVAL.md)) generalises: the
book compounds so fast that a new sleeve helps only if its **own strategy
out-returns the book**, which no commodity trend signal does.

Reproduce with:

```bash
python scripts/eval_commodity_add.py
```

Window: full OOS **2021-01-05 → 2026-07-17** (same staggered-inception window
as the live app; candidate engines selected on full-period 2016→now data, the
same "tuned to maximise full-period results" convention REMX ships with).

---

## 1. Candidates & method

Twelve liquid US-listed commodity / commodity-equity ETFs not already in the
universe:

| Bucket | Tickers |
|---|---|
| Precious | SLV (silver) · SIL (silver miners) · PPLT (platinum) · PALL (palladium) |
| Industrial | CPER (copper) · COPX (copper miners) · XME (metals & mining) · LIT (lithium) |
| Energy/other | URA (uranium) · UNG (natural gas) · DBC (broad basket) · DBA (agriculture) |

Each candidate gets its own signal, swept over the repo's exact long/flat
trend-engine grid (`ma 150` · `dual_ma 50/200` · `dual_ma 25/100` ·
`macd 10/20/9` · `ma_vol 50`, each × {−5% stop, no stop}) and selected by
full-period Sharpe. The winning sleeve's OOS return stream is folded into the
live 17-instrument universe and the cross-asset weights re-optimised with the
**identical** `optimize_weights()` call the app uses (same caps, SATA
idle-cash yield, drawdown floors, objectives), for every risk profile.

Because adding a column changes the Monte-Carlo search dimension (which moves
the sampled optimum by ±hundreds of pp regardless of what was added — the same
search-noise the QQQ test documented), the verdict rests on two **noise-free**
reads:

* the deterministic **equal-weight** blend delta (clean attributable signal);
* a **marginal test**: the candidate blended into the *frozen* baseline
  optimal book at fixed 2% / 5% / 10% weights — pure arithmetic, no re-search.

---

## 2. Sleeve-level result — no commodity trend signal comes close to the book

Best engine per candidate (full-period Sharpe selection), then OOS 2021→now:

| Sleeve | Best engine | OOS return | OOS MDD | OOS Sharpe | B&H OOS |
|---|---|---:|---:|---:|---:|
| XME | MACD 10/20/9, no stop | +82% | −24% | 0.59 | +192% |
| COPX | MACD 10/20/9, no stop | +81% | −32% | 0.55 | +131% |
| DBC | dual-MA 25/100, −5% | +71% | −34% | 0.69 | +98% |
| SLV | MA 150, −5% | +64% | −42% | 0.45 | +100% |
| DBA | MACD 10/20/9, −5% | +43% | −18% | 0.71 | +73% |
| URA | MACD 10/20/9, no stop | +41% | −32% | 0.36 | +150% |
| LIT | MA 150, −5% | +40% | −44% | 0.38 | +3% |
| CPER | MACD 10/20/9, −5% | +38% | −32% | 0.41 | +72% |
| UNG | dual-MA 50/200, no stop | +30% | −49% | 0.32 | −72% |
| SIL | MA 150, no stop | +15% | −55% | 0.24 | +46% |
| PPLT | MA 150, no stop | +11% | −48% | 0.20 | +45% |
| PALL | dual-MA 50/200, −5% | −28% | −53% | −0.08 | −49% |

The strongest candidate sleeve (XME, Sharpe 0.59) is **half** the weakest
existing sleeve's quality — the current book's strategies run Sharpe
1.1–3.2 and the equal-weight book alone compounds ~+867% over the same window.
Several candidates (URA, XME, COPX, SLV) also fail to beat their own
buy-&-hold, because a generic trend filter gives up more upside than the
sideways-chop it avoids in these whippy, mean-reverting commodities.

## 3. Combined-blend result — every candidate dilutes

Deterministic equal-weight deltas (the clean attributable signal; baseline
+867% / −13.1% MDD / Sharpe 2.74):

| Add | Δ return | Δ Sharpe |
|---|---:|---:|
| + UNG | −51pp | +0.02 |
| + XME | −61pp | −0.04 |
| + COPX | −61pp | −0.05 |
| + DBC | −67pp | +0.04 |
| + URA | −70pp | −0.07 |
| + SLV | −74pp | −0.06 |
| + DBA | −78pp | +0.05 |
| + CPER | −79pp | −0.01 |
| + LIT | −82pp | −0.09 |
| + PPLT | −92pp | −0.07 |
| + SIL | −92pp | −0.09 |
| + PALL | −108pp | −0.12 |

All twelve **lower** the combined return; none moves Sharpe meaningfully. The
per-profile optimiser runs agree once the search-noise is discounted: every
candidate is assigned a residual 0.1–1.7% weight (the QQQ-noise signature),
versus the 14–30% a genuine winner (ERX, SOXL, NUGT) earned in the shipped
leveraged-sibling eval. The marginal test (§4) confirms it deterministically.

## 4. Marginal test — deterministic, and unanimous

Each candidate blended into the **frozen** baseline optimal book at fixed
weights (pure arithmetic — the only way the delta is 100% attributable):

| Add (Balanced book, +859% / Sharpe 3.16) | w=2% | w=5% | w=10% |
|---|---:|---:|---:|
| UNG *(best)* | −18pp | −45pp | −90pp |
| COPX | −19pp | −48pp | −94pp |
| XME | −20pp | −50pp | −97pp |
| SLV | −21pp | −52pp | −101pp |
| … | | | |
| PALL *(worst)* | −34pp | −84pp | −161pp |

On the Aggressive book (+2510%) the dilution is proportionally larger
(−87pp … −133pp at just 2% weight; −419pp … −607pp at 10%). Sharpe changes
are ±0.01–0.15 everywhere — noise-level. **All twelve candidates lose return
at every weight in every profile.** There is no tuning of the grid that
rescues this: it is the arithmetic of moving capital from Sharpe-1.1–3.2
sleeves to Sharpe-0.2–0.7 sleeves.

## 5. The one interesting result — silver ridden on the GOLD signal

Silver is the only candidate economically close enough to an existing signal
to borrow it. Traded off the gold app's tuned divergence signal (the exact
architecture that ships GDX / UGL / NUGT), with stops mirroring each sibling's
shipped analogue (SLV −3% like GLDM, SIL −5% like NUGT, AGQ signal-exit-only
like UGL), the sleeves transform:

| Sleeve | Own signal (from §2) | Off the gold signal (OOS) |
|---|---|---|
| SLV (1× silver) | +64% · −42% · Sharpe 0.45 | **+148% · −17% · Sharpe 1.14** |
| SIL (silver miners) | +15% · −55% · Sharpe 0.24 | **+180% · −24% · Sharpe 1.10** |
| AGQ (2× silver) | — | **+339% · −40% · Sharpe 0.98** |

Sharpe 1.1 is GDX-class quality — the gold signal genuinely carries silver.
But in the blend, the frozen-book marginal test still says the book is faster:

| Add | Balanced w=2% / 5% / 10% | Aggressive w=2% |
|---|---|---|
| SLV | −11pp / −28pp / −55pp (Sh +0.02…+0.09) | −70pp |
| SIL | −9pp / −22pp / −43pp (Sh +0.01…+0.04) | −64pp |
| **AGQ** | **+3pp / +7pp / +13pp (Sh +0.01, MDD −0.05…−0.23pp)** | −34pp |

**AGQ is the only instrument of the fifteen tested that adds return without
adding risk** — and it is also the only add whose deterministic equal-weight
delta is positive (+7.3pp, Sharpe +0.022). It is, however, a *small* win
confined to the Balanced book; in the return-maximising Growth/Aggressive
books even a 2× silver sleeve dilutes (−34pp at 2%), because the shipped NUGT
(2× gold miners, +1183% OOS, Sharpe 1.48) already occupies the
precious-metals-leverage slot with roughly 3× AGQ's firepower.

---

## 6. Verdict

**No — none of the twelve remaining commodity sleeves improves the Overall
book as its own app.** Every candidate lowers the combined return in the
noise-free equal-weight and frozen-book marginal tests, for at best a
rounding-error Sharpe change. The reason is structural, not a tuning miss: the
book's existing sleeves compound at Sharpe 1.1–3.2, while the best commodity
trend signal reaches ~0.6–0.7 — capital moved to any of them is capital taken
from strictly better strategies, the same maths that rejected QQQ/DBA/DBMF and
that only **leverage on an already-strong signal** (SOXL, ERX, NUGT) ever
overcame.

**The one qualified exception is silver ridden on the gold app's signal**
(the GDX/UGL/NUGT architecture — see §5): silver is the only candidate whose
economics let it borrow an existing high-Sharpe signal instead of a weak new
one. If any commodity expansion were to ship, it would be an **AGQ (2× silver)
`lev` sibling on the gold signal** — not a new commodity app — and it would be
a Balanced-book diversifier worth ~+7pp / +0.02 Sharpe, not a return engine:
NUGT already fills the precious-metals-leverage slot with ~3× the firepower.
The 1×/miner variants (SLV, SIL) are not worth their slots. **Recommendation:
ship nothing; revisit AGQ only if a Balanced-profile Sharpe polish is ever
wanted.**

**Honest caveats.** Candidate engines were selected on full-period data (the
repo's convention), which flatters them — the honest OOS numbers above are
*upper* bounds. PALL/UNG carry sub-−45% strategy drawdowns that would breach
every profile's budget on their own. And commodity futures ETFs (CPER, DBC,
DBA, UNG) carry roll costs that the daily-close back-test includes only via
the price series, not execution slippage.

---

## Appendix — where each number comes from

| Piece | Where |
|---|---|
| Eval script (all tables) | `scripts/eval_commodity_add.py` |
| Optimiser / caps / SATA / profiles | `app/overall_core.py` |
| Trend-engine grid | `backtest_ticker.py` (`trend_long_array`) |
| Sibling-off-parent architecture | `scripts/eval_leveraged_add.py` · [`SOXL_ERX_ADDITION_EVAL.md`](SOXL_ERX_ADDITION_EVAL.md) |
| Prior 1×-diversifier rejections (QQQ, DBA, DBMF) | [`SOXL_ERX_ADDITION_EVAL.md`](SOXL_ERX_ADDITION_EVAL.md) §"Why leverage" |
