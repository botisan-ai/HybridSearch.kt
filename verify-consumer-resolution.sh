#!/bin/bash
# Proves a staged/downloaded Maven repo actually resolves for a clean consumer.
#
# Generates a minimal AGP library project in a temp dir whose only repositories
# are the given file:// repos plus google()/mavenCentral(), then resolves its
# release runtime classpath (graph AND artifacts) under a fresh
# GRADLE_USER_HOME — no repository-local conveniences, no cache reuse, so bad
# POM/module metadata or a broken zip layout fails here instead of at a
# consumer's desk. Only the Gradle wrapper binary is shared with this repo.
#
# Usage: verify-consumer-resolution.sh <group:artifact:version> <repo-dir> [<repo-dir>...]
set -euo pipefail
cd "$(dirname "$0")"

COORDINATE="$1"
shift
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"

CONSUMER=$(mktemp -d)
GRADLE_HOME=$(mktemp -d)
trap 'rm -rf "$CONSUMER" "$GRADLE_HOME"' EXIT

FILE_REPOS=""
for repo in "$@"; do
  abs=$(cd "$repo" && pwd)
  FILE_REPOS="$FILE_REPOS        maven { url = uri(\"file://$abs\") }"$'\n'
done

cat > "$CONSUMER/settings.gradle.kts" <<EOF
pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
$FILE_REPOS        google()
        mavenCentral()
    }
}
rootProject.name = "consumer-check"
include(":consumer")
EOF

cat > "$CONSUMER/build.gradle.kts" <<'EOF'
plugins { id("com.android.library") version "9.0.1" apply false }
EOF

mkdir -p "$CONSUMER/consumer/src/main"
printf '<manifest />\n' > "$CONSUMER/consumer/src/main/AndroidManifest.xml"

cat > "$CONSUMER/consumer/build.gradle.kts" <<EOF
plugins { id("com.android.library") }
android {
    namespace = "ai.botisan.consumercheck"
    compileSdk = 36
    defaultConfig { minSdk = 24 }
}
dependencies { implementation("$COORDINATE") }
tasks.register("verifyResolves") {
    doLast {
        val cfg = configurations.getByName("releaseRuntimeClasspath")
        val unresolved = cfg.incoming.resolutionResult.allDependencies
            .filterIsInstance<org.gradle.api.artifacts.result.UnresolvedDependencyResult>()
        check(unresolved.isEmpty()) { "unresolved: " + unresolved.joinToString { it.attempted.displayName } }
        val ids = cfg.incoming.resolutionResult.allComponents.map { it.id.displayName }
        check(ids.contains("$COORDINATE")) { "did not resolve $COORDINATE; components: \$ids" }
        val files = cfg.files // forces artifact download, not just metadata
        println("verifyResolves OK: \${ids.size} components, \${files.size} artifacts")
    }
}
EOF

echo "sdk.dir=$ANDROID_HOME" > "$CONSUMER/local.properties"

echo "==> resolving $COORDINATE in a cache-isolated consumer (fresh GRADLE_USER_HOME)"
(cd android && ./gradlew --no-daemon --console=plain -g "$GRADLE_HOME" -p "$CONSUMER" :consumer:verifyResolves)
echo "==> consumer resolution OK: $COORDINATE"
