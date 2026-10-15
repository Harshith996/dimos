# DimOS integration with dimos-cdrgen

This layer uses the published `dimos-cdrgen==0.1.0` package. The parser,
generators, CDR registry, ownership checks and native build recipes are maintained
in [dimos-cdrgen](https://github.com/TomCC7/dimos-cdrgen). They are not copied here.

`dimos-message-build==0.1.0` is a local compatibility distribution for existing
DimOS build hooks. Its modules forward to `dimos_cdrgen`. Both namespaces use the
same types and registry. Generated packages depend directly on `dimos-cdrgen`.
The compatibility distribution itself is not claimed to be published on PyPI.

## Install and generate

From this checkout, in an active Python environment:

```sh
python -m pip install './dimos/message_codegen[native]'
```

The dependency wheel already contains its verified parser. No parser preparation
or native compilation runs during this install. For a standalone external project,
install `dimos-cdrgen==0.1.0` directly and follow its
[Python, C++ and Rust quickstarts](https://tomcc7.github.io/dimos-cdrgen/).

The existing project configuration and command remain available:

```toml
[project]
name = "sample-messages"
version = "0.1.0"

[tool.dimos.messages]
source = "interfaces"
```

```sh
dimos-message-generate
python -m pip install build/message-source/python
```

The `.msg` files must exist before generation. `dimos-message-generate` and
`dimos-cdrgen` call the same implementation. Existing
`dimos_message_build.source_package` and `dimos.message_codegen.source_package`
build APIs delegate to that implementation too. See the
[source-package guide](../../docs/development/message-source-packages.md) for
package hooks, catalog selection and dependency configuration.

## Native integration

`prepare_cpp()` builds native artifacts explicitly. The DimOS
`write_cmake_toolchain()` adapter also discovers an available DimOS SDK in the
installed or checkout layout. Source hashes, caches and target settings remain
owned by the library. `dimos/native/cdr_codec.hpp` forwards to the library's CDR
header and retains the `dimos::native` function names.

Standard schema catalogs, transport and recording integration belong to later
stack layers. This integration does not add ROS nodes or prove live ROS/DDS
interoperability. Existing [codec limits](../../docs/development/message-limitations.md)
still apply.

## Verify the integration

With CMake, a C++17 compiler, Cargo, and the workflow's Python test dependencies:

```sh
bash scripts/test_message_codegen.sh
```

This prepares pinned schema fixtures, checks generated typing and cross-package
ownership, and compiles the Python/C++/Rust exchange. Set
`DIMOS_MESSAGE_WHEELHOUSE` and `DIMOS_MESSAGE_INSTALL_REQUIRED=1` to require the
isolated source/wheel/sdist/editable checks. The optional #4461 joint check needs
`DIMOS_BUILD_CONFIG_SOURCE`; a skip is not evidence for that adapter.

Library parser/emitter/cache regression tests live in the independent repository.
DimOS retains its compatibility, packaging and consumer acceptance tests here.
