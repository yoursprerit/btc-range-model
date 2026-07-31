"""Single source of truth for the deliberate strategy-logic version.

BUMP ``STRATEGY_VERSION`` whenever the Overall logic or any per-asset
strategy changes materially — gate/tilt/water-fill rules, optimiser
objective/caps, per-asset entry/exit logic or parameters, universe changes.
Display/UI-only changes do NOT warrant a bump.  The publisher stamps every
Targetbook it commits with this version (plus the publishing commit SHA), and
the Overall P&L section segments the as-published record wherever the stamp
changes, so old-logic and current-logic performance are never conflated.

Kept dependency-light (stdlib only) so the read-only artifact apps
(📋 Targetbook, ✅ Executed Book, 🕵️ Daily Audit, 🩺 Strategy Health) can show
the badge without importing the heavy engine stack.  ``overall_core``
re-exports the constant for the engine/publisher side.
"""
STRATEGY_VERSION = "v1"


def badge_caption() -> str:
    """The one-line badge every app shows under its title."""
    return (f"⚙️ Strategy logic version: **`{STRATEGY_VERSION}`** — bumped on "
            "material strategy changes; every published Targetbook is "
            "stamped with the version that produced it.")
