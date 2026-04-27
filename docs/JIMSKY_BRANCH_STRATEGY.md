# Jimsky Branch Strategy for HY-World 2.0

This fork tracks upstream HY-World 2.0 while keeping Jimsky-specific reconstruction, Modal, Telegram/Drive/SIGIL, and product workflow changes isolated.

## Repositories

- Upstream source: <https://github.com/Tencent-Hunyuan/HY-World-2.0>
- Jimsky fork: <https://github.com/TheMindExpansionNetwork/HY-World-2.0>
- Jimsky wrapper/product repo: <https://github.com/TheMindExpansionNetwork/jimsky-universe-builder>

## Branch model

- `upstream/main` is Tencent-Hunyuan's source of truth.
- `origin/main` is the fork's default branch and can stay close to the upstream-compatible integration lane.
- `origin/jimsky/universe-builder` is the Jimsky custom branch for Modal reconstruction glue, packaging helpers, agent docs, and integration work that should not be assumed upstream-ready.
- Future upstream pulls should happen through `vendor-sync/*` branches, not destructive resets.

## Safe update workflow

```bash
git fetch upstream --prune
git fetch origin --prune
git checkout jimsky/universe-builder
git pull --ff-only origin jimsky/universe-builder
git checkout -b vendor-sync/upstream-$(date +%Y%m%d)
git merge upstream/main
# Resolve conflicts, run syntax/tests, then merge or PR back into jimsky/universe-builder.
```

## Rules

- Do not commit `.env`, credentials, API tokens, SSH keys, OAuth files, or private endpoint secrets.
- Do not commit model weights, large generated reconstruction outputs, caches, or random batch media.
- Keep attribution/license headers intact when touching upstream code.
- Prefer small, reviewable commits.
- Product-specific orchestration belongs in `jimsky-universe-builder`; low-level reproducibility fixes may live here.

## Current Jimsky lane

The first Jimsky lane is phone/photos/video → Modal GPU → HY-World / WorldMirror reconstruction → ZIP package/preview → delivery to chat, Drive, SIGIL, or MemPalace.

No GPU job is implied by this branch; GPU reconstruction still requires explicit runtime approval.
