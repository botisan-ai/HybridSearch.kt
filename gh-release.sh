#!/bin/bash
# Publishes a GitHub Release: zipped Maven repo (AAR + POM + module metadata) + bare AAR, each with SHA-256.
# Version comes from android/gradle.properties.
set -euo pipefail
cd "$(dirname "$0")"

ARTIFACT="hybridsearch-android"
VERSION=$(grep '^version=' android/gradle.properties | cut -d= -f2)

echo "==> test + assemble + publish $ARTIFACT $VERSION"
(cd android && ./gradlew --console=plain test assembleRelease publishReleasePublicationToBuildDirRepository)

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

cp -R android/build/maven-repo "$STAGING/maven-repo"
(cd "$STAGING" && zip -qr "$ARTIFACT-$VERSION-maven.zip" maven-repo)
cp "android/lib/build/outputs/aar/lib-release.aar" "$STAGING/$ARTIFACT-$VERSION.aar"

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
