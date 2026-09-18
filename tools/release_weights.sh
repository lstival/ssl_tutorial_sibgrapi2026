#!/usr/bin/env bash
# Create (or reuse) a GitHub Release and upload the pretrained encoders as its assets.
#
# GitHub Releases have no bandwidth quota for public repos (unlike Git LFS's 1 GB/month
# free tier, which is what caused the 404s reported off-cluster). This script is the
# one-time step that gets the .pt files onto a release; tutorial_rs.py/tutorial_ts.py
# then download from releases/download/<tag>/... instead of media.githubusercontent.com.
#
# Usage:
#   export GITHUB_TOKEN=ghp_xxx   # needs 'repo' scope (or 'contents:write' for fine-grained)
#   ./tools/release_weights.sh [tag]
#
# Re-running with the same tag is safe: existing assets with the same name are replaced.

set -euo pipefail

REPO="lstival/ssl_tutorial_sibgrapi2026"
TAG="${1:-weights-v1}"
API="https://api.github.com/repos/${REPO}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Set GITHUB_TOKEN first (a personal access token with write access to ${REPO})." >&2
  exit 1
fi

AUTH=(-H "Authorization: Bearer ${GITHUB_TOKEN}" -H "Accept: application/vnd.github+json")

FILES=(
  "artifacts/remote_sensing/checkpoints/contrastive_vit_s8.pt"
  "artifacts/remote_sensing/checkpoints/contrastive_vit_s8_ben.pt"
  "artifacts/remote_sensing/checkpoints/mae_vit_s8.pt"
  "artifacts/remote_sensing/checkpoints/mae_vit_s8_ben.pt"
  "artifacts/remote_sensing/checkpoints/dino_vit_s8.pt"
  "artifacts/remote_sensing/checkpoints/dino_vit_s8_ben.pt"
  "artifacts/remote_sensing/checkpoints/random_init_vit_s8.pt"
  "artifacts/time_series/checkpoints/contrastive_ts_encoder.pt"
  "artifacts/time_series/checkpoints/mae_ts_encoder.pt"
  "artifacts/time_series/checkpoints/dino_ts_encoder.pt"
)

echo "== Looking up release '${TAG}' =="
release_json="$(curl -sf "${AUTH[@]}" "${API}/releases/tags/${TAG}" || true)"
release_id="$(printf '%s' "$release_json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("id",""))' 2>/dev/null || true)"

if [[ -z "$release_id" ]]; then
  echo "== Creating release '${TAG}' =="
  release_json="$(curl -sf "${AUTH[@]}" -X POST "${API}/releases" \
    -d "$(python3 -c "import json; print(json.dumps({
      'tag_name': '${TAG}',
      'name': 'Pretrained encoder weights (${TAG})',
      'body': 'Pretrained SSL encoder checkpoints for the SIBGRAPI 2026 tutorial notebooks. Not a code release -- see main branch for that.',
      'draft': False,
      'prerelease': False,
    }))")")"
  release_id="$(printf '%s' "$release_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
fi
echo "release_id=${release_id}"

upload_url_base="https://uploads.github.com/repos/${REPO}/releases/${release_id}/assets"

for rel_path in "${FILES[@]}"; do
  name="$(basename "$rel_path")"
  path="${ROOT}/${rel_path}"
  if [[ ! -f "$path" ]]; then
    echo "SKIP (missing on disk): $rel_path" >&2
    continue
  fi

  # Delete any existing asset with the same name so re-runs replace it cleanly.
  existing_id="$(curl -sf "${AUTH[@]}" "${API}/releases/${release_id}/assets" \
    | python3 -c "
import json, sys
name = '${name}'
for a in json.load(sys.stdin):
    if a['name'] == name:
        print(a['id'])
        break
" || true)"
  if [[ -n "$existing_id" ]]; then
    echo "Replacing existing asset ${name} (id=${existing_id})"
    curl -sf "${AUTH[@]}" -X DELETE "${API}/releases/assets/${existing_id}" >/dev/null
  fi

  echo "Uploading ${name} ($(du -h "$path" | cut -f1)) ..."
  curl -sf "${AUTH[@]}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary "@${path}" \
    "${upload_url_base}?name=${name}" >/dev/null
done

echo "== Done. Assets: =="
curl -sf "${AUTH[@]}" "${API}/releases/${release_id}/assets" \
  | python3 -c "
import json, sys
for a in json.load(sys.stdin):
    print(f\"  {a['name']:<32} {a['size']/1_048_576:6.1f} MB  {a['browser_download_url']}\")
"
