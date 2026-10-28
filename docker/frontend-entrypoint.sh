#!/bin/sh
set -eu

# Host node_modules must never be reused inside the Linux container.
# The named volume survives rebuilds, so refresh it when the shared lock changes.
lock_hash="$(sha256sum package-lock.json | cut -d ' ' -f 1)"
marker="node_modules/.livestream-package-lock.sha256"
if [ ! -f "$marker" ] || [ "$(cat "$marker")" != "$lock_hash" ]; then
    npm ci --no-audit --no-fund
    printf '%s\n' "$lock_hash" > "$marker"
fi

exec "$@"
