# External scheduler for the daily Targetbook publish

The `Publish target book (IBKR Option C)` workflow is supposed to run at
**7:15 AM America/Chicago** every day, but GitHub's `on: schedule` is
best-effort: fires are routinely delivered 1–3 hours late and are sometimes
dropped outright (2026-08-03: every slot was dropped for 2.5+ hours and the
day's book sat unpublished all morning). The precise fix is an **external
scheduler calling the `workflow_dispatch` API** at the anchor time; the
workflow's cron slots then remain as backup. This runbook sets that up with
[cron-job.org](https://cron-job.org) (free) and a fine-grained GitHub PAT.

## 1. Create the PAT (~2 minutes)

1. GitHub → **Settings** → **Developer settings** → **Personal access
   tokens** → **Fine-grained tokens** → **Generate new token**.
2. Name: `targetbook-dispatch` (or similar). Expiration: your call — one
   year is fine, but put the renewal date in your calendar; a silently
   expired PAT recreates the late-publish problem without any error you'd
   notice.
3. **Resource owner**: `yoursprerit`. **Repository access**: *Only select
   repositories* → `btc-range-model`.
4. **Repository permissions**: **Actions → Read and write** (Metadata: read
   is added automatically). Nothing else.
5. Generate and copy the token (shown once).

## 2. Verify the token from your own machine (optional, ~1 minute)

A successful dispatch returns **HTTP 204** with an empty body; harmless to run
even on a day whose book is already published (a same-day re-publish yields
the identical book).

```bash
curl -i -X POST \
  -H "Authorization: Bearer PASTE_TOKEN_HERE" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/yoursprerit/btc-range-model/actions/workflows/publish-target-book.yml/dispatches" \
  -d '{"ref":"main"}'
```

(On Windows PowerShell use `curl.exe`, backticks for line continuation, and
escape the body quotes: `-d "{\"ref\":\"main\"}"`.) Within seconds a *Publish
target book* run with event `workflow_dispatch` appears in the Actions tab.
Troubleshooting: `401 Bad credentials` → token pasted wrong; `403`/`404` →
token missing the **Actions: Read and write** permission or not scoped to
`btc-range-model`.

## 3. Create the cron-job.org job

Create a free account, then **Create cronjob** with:

| Setting | Value |
| --- | --- |
| URL | `https://api.github.com/repos/yoursprerit/btc-range-model/actions/workflows/publish-target-book.yml/dispatches` |
| Schedule | Every day at **7:16 AM**, timezone **America/Chicago** (Chicago) |
| Request method (Advanced tab) | `POST` |
| Request body | `{"ref":"main"}` |
| Header 1 | `Authorization: Bearer <YOUR_PAT>` |
| Header 2 | `Accept: application/vnd.github+json` |
| Header 3 | `X-GitHub-Api-Version: 2022-11-28` |
| Notifications | Enable failure notifications (Settings → notify on failure) |

Save, then use cron-job.org's **"Test run"** button: same success criteria as
the curl test above — **HTTP 204**, and a `Publish target book` run with event
`workflow_dispatch` in the repo's Actions tab within seconds.

### Why these choices

- **7:16, not 7:15, America/Chicago (not UTC):** a `workflow_dispatch`
  bypasses the workflow's schedule guard and always runs, so the job must
  never fire before the day's 7:15 AM CT anchor. Scheduling in the
  America/Chicago timezone handles the CDT/CST switch automatically
  (a fixed 12:15 UTC job would fire at 6:15 AM CST in winter — before the
  anchor). The one-minute offset keeps it safely after the anchor.
- **Safe if GitHub's own cron already published:** the publish's data basis
  is pinned to the 7:15 AM CT anchor and the prev-book slot only rolls
  forward on the first publish of a new day, so a same-day re-publish
  produces the identical book. The `concurrency` group serializes any
  overlap, and once a book exists the remaining scheduled slots guard-skip.

## Fallbacks if the external job is down

- The workflow's own cron slots (11:05 UTC through 23:15 UTC) still publish
  whenever GitHub delivers one — late, but the book lands.
- The 🚀 **Publish new target book** button in the Target Book app.
- The relay workflow `.github/workflows/dispatch-publish.yml` (on the
  `claude/targetbook-715-ct-delay-h3cck6` / `publish-trigger` branches):
  pushing any commit whose message contains `[dispatch-publish]` to one of
  those branches dispatches the publish using the relay's own
  `GITHUB_TOKEN` — useful for tokenless automation (e.g. Claude Code
  sessions, whose GitHub credentials have `contents: write` but not the
  Actions scope this API needs).
