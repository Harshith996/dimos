# Verify messages in the installed DimOS wheel

Build the layer-2 DimOS wheel, install it into a clean environment,
then run outside the checkout. This example constructs an upstream `Image` and
the existing custom `LineSegments3D`, checks their CDR roundtrips, and inspects
the packaged definitions and native build inputs.

Use Python 3.12 and uv on Linux or macOS. No ROS installation, CMake, Rust
toolchain or MCAP package is needed for message use. Building DimOS from a checkout needs its normal C++ compiler and
prebuilt web bundles. Initial preparation, building, and dependency installation need network access unless their inputs
are cached.

From the repository root on this example branch:

```sh
message_demo_dir="$(mktemp -d)"
uv run --no-project --with 'dimos-cdrgen[prepare]==0.1.0' python -m scripts.prepare_builtin_sources
uv build --wheel --out-dir "$message_demo_dir/wheels"
uv venv --python 3.12 "$message_demo_dir/venv"
uv pip install --python "$message_demo_dir/venv/bin/python" "$message_demo_dir"/wheels/dimos-*.whl
cp examples/message-package/demo_installed_wheel.py "$message_demo_dir/"
cd "$message_demo_dir"
"$message_demo_dir/venv/bin/python" -I demo_installed_wheel.py
```

Expected output, with your temporary directory in the first line:

```text
Installed package: <temporary-directory>/venv/lib/python3.12/site-packages/dimos_generated/__init__.py
Catalog: 73 types
PASS: sensor_msgs/msg/Image pixels, encoding and source stamp
PASS: dimos_msgs/msg/LineSegments3D geometry, weight and nested type identity
PASS: packaged definitions and C++/Rust build inputs; no compiled message libraries
```

`-I` ignores `PYTHONPATH` and the script directory during imports. The script
also verifies that the imported package lives inside the clean environment
and that the owning DimOS distribution is installed. It uses the installed
wheel rather than an editable checkout.

## What the package contains

Git stores 14 custom definitions, `catalog.json`, upstream pins, hashes, and
thin build entry points. Preparation copies the selected upstream definitions
and custom inputs into ignored `packages/dimos-generated/interfaces/`.
The root `setup.py` calls `dimos_cdrgen.source_package.setup_arguments` to generate and
package Python imports and stubs, complete schemas, C++ CMake projects, and
Rust Cargo inputs. Those generated files are shipped in the wheel, not checked
into Git. There is no separately published built-in Python message distribution.
A source distribution also carries the prepared `.msg` inputs.

At this layer's head, 69 catalog roots resolve to 73 types: 59 upstream and
14 custom. This is an explicit supported selection, not proof that every
message used elsewhere in the repository is covered. Later stack layers add
further types.

The installed DimOS wheel includes these messages alongside the main runtime.
Messages use `dimos-cdrgen`, rosbags, and NumPy; an ordinary DimOS installation
also installs its other runtime dependencies. Running this example does not fetch `.msg` files or compile
native bindings. For offline installation, make those dependency wheels
available locally and pass `--no-index --find-links <wheelhouse>` to
`uv pip install`. Offline source preparation separately requires cached
archives and `scripts.prepare_builtin_sources --offline`.

MCAP recording is a separate layer. The existing
[layer-3 example](https://github.com/dimensionalOS/dimos/blob/3cb96b780433dcdac605f69453721aac2df1d800/examples/message-codegen/demo_mcap.py)
writes messages and their schemas into a file for external readers. It is not
required for this wheel check.
