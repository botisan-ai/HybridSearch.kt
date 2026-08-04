plugins {
    id("com.android.library")
    id("maven-publish")
}

android {
    namespace = "ai.botisan.hybridsearch"
    compileSdk = 36

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    publishing {
        singleVariant("release")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    api("ai.botisan:tantivy-android:0.2.0")
    api("ai.botisan:hnsw-android:0.2.0")
    api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("net.java.dev.jna:jna:5.19.1")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")

    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}

// Host-JVM unit tests need the sibling repos' host dylibs on jna.library.path.
val hostLibDirs = listOf(
    "../../../tantivy.kt/rust/target/release",
    "../../../HNSW.kt/rust/target/release",
).map { project.file(it).absolutePath }

tasks.withType<Test>().configureEach {
    systemProperty("jna.library.path", hostLibDirs.joinToString(File.pathSeparator))
}

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "ai.botisan"
            artifactId = "hybridsearch-android"
            version = project.version.toString()
            afterEvaluate { from(components["release"]) }
            pom {
                name = "hybridsearch-android"
                description = "Hybrid search (BM25 + HNSW vectors fused with RRF) over tantivy-android and hnsw-android"
                url = "https://github.com/botisan-ai/HybridSearch.kt"
                licenses { license { name = "MIT License" } }
            }
        }
    }
    repositories {
        maven {
            name = "buildDir"
            url = uri(rootProject.layout.buildDirectory.dir("maven-repo"))
        }
    }
}
