pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode = RepositoriesMode.FAIL_ON_PROJECT_REPOS
    repositories {
        google()
        mavenCentral()
        // Release channel: unzipped <artifact>-<version>-maven.zip repos (run ../bootstrap-deps.sh).
        val bootstrapped = File(rootDir, "build/native-repos")
        if (bootstrapped.exists()) {
            bootstrapped.listFiles()?.filter { it.isDirectory }?.forEach { maven { url = it.toURI() } }
        }
        // Dev fallback: sibling checkouts' locally-published repos
        // (cd ../tantivy.kt/android && ./gradlew publishReleasePublicationToBuildDirRepository; same for HNSW.kt).
        listOf(
            "../../tantivy.kt/android/build/maven-repo",
            "../../HNSW.kt/android/build/maven-repo",
        ).forEach {
            val dir = File(rootDir, it)
            if (dir.exists()) maven { url = dir.toURI() }
        }
    }
}

rootProject.name = "hybridsearch-android"
include(":lib")
