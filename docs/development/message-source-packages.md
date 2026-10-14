# Build a message package from local inputs

The generic generator and source-package hooks do not select a DimOS catalog.
They read explicit project inputs and installed message dependencies. A package
build never downloads schemas or invokes a native compiler.

## Prepare the build tools

The backend is a small, separate Python distribution so external projects do not need the DimOS robot runtime as a build dependency. Its import name is `dimos_message_build`.

From this PR checkout, in an environment with pip, setuptools, wheel, and requests:

```sh
python dimos/message_codegen/prepare_parser.py
python -m pip wheel --wheel-dir /tmp/message-wheels ./dimos/message_codegen
python -m pip wheel --wheel-dir /tmp/message-wheels /path/to/native-package-checkout/packages/dimos-build-config
```

Use the matching `dimos-build-config` integration from [PR #4461](https://github.com/dimensionalOS/dimos/pull/4461) for the second path. These commands prepare the pinned upstream parser and build both tools. The wheel directory also contains its Python dependencies. Use an absolute path for pip build isolation. No PyPI release of this proposal is assumed.

## Create an external message package

Create `interfaces/sample_msgs/msg/Reading.msg`:

```text
int32 sequence
float64 value
```

Configure the package once in `pyproject.toml`:

```toml
[build-system]
requires = ["scikit-build-core>=1.0,<2", "dimos-build-config==0.1.0", "dimos-message-build==0.1.0"]
build-backend = "scikit_build_core.build"

[project]
name = "sample-messages"
version = "0.1.0"
dynamic = ["entry-points"]
dependencies = ["dimos-message-build==0.1.0", "rosbags==0.11.0", "numpy>=1.26.4"]

[tool.dimos.messages]
source = "interfaces"

[[tool.dynamic-metadata]]
provider = "dimos"
```

`[tool.dimos.messages]` is configuration for the build tools, not a pip feature.
No `setup.py` or `MANIFEST.in` is needed with this integration. The module name
defaults to `sample_messages`.

```text
pip install . → isolated build environment → scikit-build-core
    → dimos-build-config reads tool.dimos
    → dimos-message-build reads messages, generates code and schemas
    → wheel contains generated Python packages → pip installs wheel
```

The same generator can be run explicitly, like a linter reading project configuration:

```sh
python -m pip install --no-index --find-links /tmp/message-wheels dimos-message-build==0.1.0
dimos-message-generate
# Equivalent, including an optional project directory:
python -m dimos_message_build.source_package .
```

Both routes call `prepare()` in `dimos/message_codegen/source_package.py`.
Its `configuration()` reads the TOML and its generator writes `build/message-source`.
Explicit generation is useful for inspection; it is not a prerequisite for installation.

Use matching proposal backend wheels. They are not assumed to exist on PyPI.
For an offline build, prepare those wheels and all Python dependencies first:

```sh
pip install --no-index --find-links /tmp/message-wheels .
pip install --no-index --find-links /tmp/message-wheels -e .
```

The same implementation generates local package sources in `build/message-source` for
wheel, source and editable builds. They emit native source projects but do not
invoke ROS, CMake or Cargo. A source distribution copies declared inputs and
resources into the archive. A wheel contains generated values and their complete
schema closure. Import does not build files or access the network.

```python
from sample_messages.sample_msgs.msg import Reading
from dimos_message_build.registry import encode, decode

value = Reading(sequence=17, value=21.5)
assert decode(encode(value), Reading).value == 21.5
```

After editing a source, repeat the editable install and restart Python processes.
Schemas and type identities are frozen within each process. Installing an existing wheel does not invoke the generator. This layer uses generated Python value types and the Python CDR codec; it does not build a Python native extension.

## Include messages with a module package

Add these tables to the same project when it also ships application code and blueprints:

```toml
[tool.dimos]
package = "src/sample_app"

[tool.dimos.blueprints]
reader = "sample_app.module:Reader"
```

The application package and generated `sample_messages` package must have different
module names. Both entry-point groups are included in the same wheel. Build and
metadata discovery do not import application code. A native module's selected
CMake/Cargo command still runs during module preparation as described in #4461;
message generation during packaging does not compile that application.

## Select roots and dependencies

Without `roots`, the project owns all messages declared under its source folder.
Use `roots = ["sample_msgs/msg/Reading"]` to select a dependency closure explicitly.
Alternatively, `catalog = "catalog.json"` reads a format-1 JSON manifest whose
`roots` maps type names to support reasons. Do not specify both.

`resources = ["pins.json"]` carries declared metadata into both source archives
and generated package resources. Input paths must remain inside the project.
The generic hooks have no repository-path or built-in package-name convention.

For another installed message package, declare an exact module/version under
`[tool.dimos.messages.dependencies]`. Add that distribution to both build-system
and project dependencies. The generator imports its owned Python/C++/Rust types,
checks schema/version/ABI identity and retains the full schema closure. It does
not generate another owner for the same type.

## Prepare upstream data explicitly

`dimos_message_build.source_inputs.prepare_sources(lock, output, cache,
offline=True)` reads a source lock. Each source specifies its URL, archive SHA256,
archive prefix and selected files with their source paths and SHA256 values.
This preparation API is separate from source-package hooks. It needs the explicit
`dimos-message-build[prepare]` dependency when used.

Online preparation downloads only declared archives. Offline preparation requires
cached archives. Both paths check archive and selected-file hashes before replacing
the prepared input directory. Unsafe paths and non-regular files fail. Product
layers own their upstream revisions, type selection, custom messages and thin
preparation/release entrypoints. No package manager or core/extras scheme is added.

## Configuration reference

| Field under `[tool.dimos.messages]` | Meaning |
| --- | --- |
| `source` | Project-local input directory; defaults to `interfaces`. |
| `module` | Python module name; defaults to project name with hyphens replaced by underscores. |
| `roots` | Optional list of message names; defaults to all local messages. |
| `catalog` | Alternative root-selection file; cannot be combined with `roots`. |
| `resources` | Project-local files included in source and generated packages. |
| `dependencies` | Mapping from dependency module names to exact versions. |

This layer does not implement a `languages` configuration field or a `dimos build` command. Its install hooks generate package sources without running native compilers.
The low-level generator is also executable after installing the backend:

```sh
python -m pip install --no-index --find-links /tmp/message-wheels dimos-message-build==0.1.0
python -m dimos_message_build.generate --package-root interfaces \
  --type sample_msgs/msg/Reading --python-module sample_messages --output build/manual
```

Setuptools consumers can retain `setup(**setup_arguments(project))` from
`dimos_message_build.source_package`; it delegates to the same generator. It is
not needed for the shared scikit-build-core workflow above.
