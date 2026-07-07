# VaRimi Sentinel — Android device harness

A minimal on-device harness that runs the three bundled ONNX heads with
**ONNX Runtime Mobile**, fully offline, and logs per-advisory latency so a true
**hardware-in-the-loop** benchmark can be recorded on a low-end device
(milestone M3 of the proposal roadmap).

## Honest status
**Unverified build.** This scaffold was authored on a machine without the
Android SDK, so it has not been compile-tested. It uses only stable, standard
APIs (plain `Activity`, `onnxruntime-android` 1.19.0, AGP 8.5.2, Kotlin 2.0)
and is expected to build in Android Studio with at most minor fixes. Treat any
build error as a scaffold bug to fix, not a design problem.

## Build steps
1. Sync the assets (ONNX heads + vocab + lookup snapshot) from the repo root:
   ```bash
   python -m varimi.model.train        # if models/ is empty
   python -m varimi.edge.export
   python scripts/sync_android_assets.py
   ```
   This populates `app/src/main/assets/` (risk/yield/price .onnx,
   `encoder_vocab.json`, `lookup.csv`). Assets are intentionally not committed.
2. Open the `android/` folder in **Android Studio** (Koala or newer). Let it
   generate the Gradle wrapper and sync.
3. Run on a device or emulator (minSdk 24).

## Recording the device benchmark
Tap **Get advisory** a few dozen times, then pull the numbers:
```bash
adb logcat -s VaRimiBench
```
Each line reports `advisory_ms` for a full three-head advisory. Record the
median/p95 on the target low-end device and update
`reports/edge_benchmark.md` ("Device validation status").

## Feature-layout contract
The Kotlin feature vector mirrors `varimi.model.pipeline.FEATURE_ORDER`
(5 numerics + province/crop/season ordinal codes). Codes come from
`encoder_vocab.json`, generated from the fitted sklearn OrdinalEncoder —
never hand-edit it.