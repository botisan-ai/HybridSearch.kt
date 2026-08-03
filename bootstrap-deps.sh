#!/bin/bash
# Fetches the tantivy-android and hnsw-android release maven repos this build depends on,
# verifies checksums, and unzips them into android/build/native-repos/ where
# settings.gradle.kts picks them up. Dev alternative: build sibling checkouts
# (../tantivy.kt, ../HNSW.kt) with `./gradlew publishReleasePublicationToBuildDirRepository`.
set -euo pipefail
cd "$(dirname "$0")"

TANTIVY_VERSION="${TANTIVY_VERSION:-0.1.0}"
HNSW_VERSION="${HNSW_VERSION:-0.1.0}"

DEST="android/build/native-repos"
mkdir -p "$DEST"

fetch() {
  local repo="$1" artifact="$2" version="$3"
  local zip="$artifact-$version-maven.zip"
  local url="https://github.com/$repo/releases/download/$version/$zip"
  echo "==> $url"
  curl -fsSL -o "$DEST/$zip" "$url"
  curl -fsSL -o "$DEST/$zip.sha256" "$url.sha256"
  (cd "$DEST" && shasum -a 256 -c "$zip.sha256")
  rm -rf "$DEST/$artifact-$version"
  unzip -qo "$DEST/$zip" -d "$DEST/$artifact-$version.tmp"
  mv "$DEST/$artifact-$version.tmp/maven-repo" "$DEST/$artifact-$version"
  rm -rf "$DEST/$artifact-$version.tmp" "$DEST/$zip" "$DEST/$zip.sha256"
}

fetch "botisan-ai/tantivy.kt" "tantivy-android" "$TANTIVY_VERSION"
fetch "lhr0909/HNSW.kt" "hnsw-android" "$HNSW_VERSION"

echo "==> done; repos in $DEST"
