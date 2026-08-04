#!/bin/bash
# Publishes a GitHub Release: zipped Maven repo (AAR + POM + module metadata) + bare AAR, each with SHA-256.
# Version comes from android/gradle.properties.
#
# Zip layout: the zip root IS the Maven repository root (ai/botisan/...), so
# consumers can point Gradle straight at the unzip directory.
set -euo pipefail
cd "$(dirname "$0")"

ARTIFACT="hybridsearch-android"
VERSION=$(grep '^version=' android/gradle.properties | cut -d= -f2)

# Fetch the *released* dependency zips first (checksummed downloads): the
# gates below then build against the same repos consumers will resolve, and a
# missing upstream release fails here instead of after upload.
echo "==> release channel deps (bootstrap-deps.sh)"
./bootstrap-deps.sh

echo "==> release gates (test + lintRelease + assembleRelease)"
(cd android && ./gradlew --console=plain test lintRelease assembleRelease)

echo "==> publishing $ARTIFACT $VERSION to a fresh local maven layout"
rm -rf android/build/maven-repo
(cd android && ./gradlew --console=plain publishReleasePublicationToBuildDirRepository)

STAGING=$(mktemp -d)
UNZIPPED=$(mktemp -d)
trap 'rm -rf "$STAGING" "$UNZIPPED"' EXIT

(cd android/build/maven-repo && zip -qr "$STAGING/$ARTIFACT-$VERSION-maven.zip" .)
cp "android/lib/build/outputs/aar/lib-release.aar" "$STAGING/$ARTIFACT-$VERSION.aar"

echo "==> asserting zip layout resolves the documented Maven path"
ZIP_ENTRIES=$(unzip -Z1 "$STAGING/$ARTIFACT-$VERSION-maven.zip")
EXPECTED_AAR="ai/botisan/$ARTIFACT/$VERSION/$ARTIFACT-$VERSION.aar"
if ! grep -Fqx "$EXPECTED_AAR" <<< "$ZIP_ENTRIES"; then
  echo "FAIL: maven zip does not contain ai/botisan/$ARTIFACT/$VERSION/$ARTIFACT-$VERSION.aar at its root" >&2
  exit 1
fi
if grep -Eq '(^|/)maven-repo/' <<< "$ZIP_ENTRIES"; then
  echo "FAIL: maven zip contains a maven-repo/ wrapper directory" >&2
  exit 1
fi

# Grep proves the member list; only real Gradle resolution proves the POM,
# module metadata, transitive deps, and repository layout. Resolve the exact
# zip being shipped — against the *downloaded release* repos of tantivy/hnsw,
# never the sibling dev checkouts — from a cache-isolated consumer.
unzip -qo "$STAGING/$ARTIFACT-$VERSION-maven.zip" -d "$UNZIPPED"
TANTIVY_REPO=$(ls -d android/build/native-repos/tantivy-android-*/ | head -1)
HNSW_REPO=$(ls -d android/build/native-repos/hnsw-android-*/ | head -1)
./verify-consumer-resolution.sh "ai.botisan:$ARTIFACT:$VERSION" "$UNZIPPED" "$TANTIVY_REPO" "$HNSW_REPO"

(cd "$STAGING" \
  && shasum -a 256 "$ARTIFACT-$VERSION-maven.zip" > "$ARTIFACT-$VERSION-maven.zip.sha256" \
  && shasum -a 256 "$ARTIFACT-$VERSION.aar" > "$ARTIFACT-$VERSION.aar.sha256")

CHECKSUMS=$(cat "$STAGING/$ARTIFACT-$VERSION-maven.zip.sha256" "$STAGING/$ARTIFACT-$VERSION.aar.sha256")

echo "==> creating GitHub release $VERSION"
gh release create "$VERSION" --generate-notes --notes "SHA-256:
\`\`\`
$CHECKSUMS
\`\`\`" || true
gh release upload "$VERSION" \
  "$STAGING/$ARTIFACT-$VERSION-maven.zip" \
  "$STAGING/$ARTIFACT-$VERSION-maven.zip.sha256" \
  "$STAGING/$ARTIFACT-$VERSION.aar" \
  "$STAGING/$ARTIFACT-$VERSION.aar.sha256" \
  --clobber

echo "==> done"
