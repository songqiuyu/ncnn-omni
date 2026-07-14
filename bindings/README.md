# Language and platform bindings

Planned bindings:

- `c/`: stable C ABI and ownership rules;
- `android/`: JNI/Kotlin wrapper;
- `apple/`: Objective-C/Swift wrapper for iOS and macOS;
- `python/`: validation and tooling wrapper, not a device runtime dependency.

Bindings consume only public API/C ABI contracts and never reach into a model
adapter or raw ncnn runtime.
