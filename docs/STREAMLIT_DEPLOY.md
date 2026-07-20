# Deploying to Streamlit Community Cloud

Step-by-step instructions to host the dashboard (all apps, one sidebar radio) for
free on **Streamlit Community Cloud**. Official reference:
<https://docs.streamlit.io/deploy/streamlit-community-cloud>.

The repo already ships everything the platform needs — the root entry-point
(`streamlit_app.py`), pinned `requirements.txt`, and `.streamlit/config.toml`
(headless, theme). The one setting you **must** get right is the Python version.

---

## ⚠️ Read this first — Python 3.12 is mandatory

Deploy on **Python 3.12**. `runtime.txt` is **ignored** by Streamlit Community
Cloud ([streamlit#15326](https://github.com/streamlit/streamlit/issues/15326)),
and the platform default (3.14) **segfaults this app's native ML stack at
runtime** — the model is pickled with scikit-learn 1.8.0, whose only installable
3.14 wheels pull a numpy/pandas/sklearn ABI combination that crashes after the UI
renders. See the header of [`requirements.txt`](../requirements.txt) for the full
rationale. You set the Python version in the deploy dialog (step 4 below), **not**
via `runtime.txt`.

---

## Prerequisites

1. A **GitHub account** with this repository (a fork is fine).
2. A free **Streamlit Community Cloud** account, signed in with GitHub:
   <https://share.streamlit.io>. Authorize it to read the repo.

---

## Steps

1. **Push the code to GitHub.** Make sure the branch you want to deploy contains
   `streamlit_app.py`, `requirements.txt`, and `.streamlit/config.toml` (the
   default branch already does).

2. **Create the app.** At <https://share.streamlit.io> → **Create app** →
   **Deploy a public app from GitHub**.

3. **Fill in the source:**
   - **Repository:** `<your-org>/btc-range-model`
   - **Branch:** `main` (or your chosen branch)
   - **Main file path:** `streamlit_app.py`

4. **Set the Python version — do not skip.** Open **Advanced settings** and
   choose **Python 3.12**. (This is the only reliable place to pin it; see the
   warning above.)

5. **Deploy.** Click **Deploy**. First build installs `requirements.txt` and can
   take a few minutes. When it finishes you get a public URL like
   `https://<your-app>.streamlit.app`; the sidebar **Application** radio switches
   between every app (Overall, BTC, Gold, the ETF apps, Target/Executed Book).

---

## Configuration notes

- **Theme & server** are pre-set in [`.streamlit/config.toml`](../.streamlit/config.toml)
  (`headless = true`, light theme, Bitcoin-orange accent) — no dashboard changes
  needed.
- **Secrets are not required.** All data is fetched from public feeds (Yahoo,
  Binance, blockchain.info, alternative.me); models load from the committed
  `models/` artefacts. Leave the **Secrets** box empty.
- **IBKR execution is *not* part of the cloud app.** The dashboard only *views*
  the target/executed books; placing real orders runs off-platform next to IB
  Gateway (see [`../IBKR_PAPER_TRADING.md`](../IBKR_PAPER_TRADING.md) and
  [`CLOUD_EXECUTOR.md`](CLOUD_EXECUTOR.md)).
- **First-load latency.** The Overall app runs every strategy live on first load
  (~30–60 s), then serves from cache. This is normal.

---

## Updating & troubleshooting

- **Redeploy** happens automatically on every push to the deployed branch.
- **Reboot / clear cache** from the app's **⋮ → Manage app** menu if a data feed
  wedges.
- **Segfault / "in the oven" hang after a UI render** → the app is almost
  certainly running on the wrong Python. Reopen **Manage app → Settings** and
  confirm **Python 3.12**; redeploy.
- **`pyarrow` serializer errors** → keep the pins in `requirements.txt`
  (`pyarrow < 25`); don't loosen them.

---

## Local smoke test before deploying

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py     # opens http://localhost:8501
```

If it runs clean locally on Python 3.12, it will run on the cloud with the same
version pinned.
