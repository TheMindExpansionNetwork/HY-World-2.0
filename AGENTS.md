# AGENTS.md — HY-World 2.0 Jimsky Fork

## Fork context

- Upstream: <https://github.com/Tencent-Hunyuan/HY-World-2.0>
- Fork: <https://github.com/TheMindExpansionNetwork/HY-World-2.0>
- Custom Jimsky branch: `jimsky/universe-builder`
- Wrapper/product repo: <https://github.com/TheMindExpansionNetwork/jimsky-universe-builder>

This repository is a fork/workbench. Keep upstream updates recoverable and avoid destructive syncs.

## Agent rules

1. Preserve upstream attribution and license notices.
2. Keep Jimsky-specific work on `jimsky/*` branches unless it is a clean upstream-compatible fix.
3. Use `vendor-sync/*` branches for upstream merges.
4. Never run `git reset --hard upstream/main` on a branch containing Jimsky commits.
5. Never commit secrets: `.env`, tokens, API keys, OAuth files, private keys, webhooks, or generated credentials.
6. Avoid committing large generated reconstructions, model weights, caches, virtualenvs, or random media batches.
7. Before pushing: run syntax checks for changed Python, `git diff --check`, and a targeted secret scan.
8. GPU reconstruction tests are cost-bearing and require explicit approval before running.

## Current lane

Jimsky Universe Builder uses this fork as the technical base for photo/video → HY-World/WorldMirror reconstruction → preview/artifact packaging.
