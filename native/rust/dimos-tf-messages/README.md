This crate contains the standard ROS 2 TF message closure, packaged by
`dimos-cdrgen==0.1.2`. Cargo generates Rust types with `ros2msg`; the SDK uses
`cdr` for serialization. No TF codec is implemented here.

The definitions are the same standard schemas shipped in DimOS's message
source package. The SDK carries this small closure so it can build without
first installing or generating the full DimOS message package.

Regenerate from the checked-in definitions with:

```sh
uv run --no-sync python bin/generate-native-tf-messages
```
