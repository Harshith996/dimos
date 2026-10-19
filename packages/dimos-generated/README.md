# Built-in message packages

Git keeps the 14 DimOS definitions under `custom/dimos_msgs`, the supported
`catalog.json`, and `schema-sources.json` with pinned revisions, archive hashes,
selected-file hashes and license paths. Bulk upstream `.msg` files are not Git
inputs. Explicit preparation puts their selected closure in ignored `interfaces/`.
Python aliases/stubs and native source projects are also ignored build outputs.

## Supported types

`catalog.json` lists explicit roots and a reason for every root. Builds
resolve their complete dependency closure. Adding another upstream `.msg` to the
source tree does not silently add an installed type.

The default package exports 73 types. The starting set is the 63 static consumer
types audited at published runtime commit
`59ce3b8ddb2aedcecd8ca0c592b64e2824899874`. The additional roots preserve:

- The two remaining project-owned bounding-box array messages. This layout
  change does not retire DimOS's own message contracts.
- `Empty` and `Int64` used by the dynamic topic/CDR fixtures.
- `UInt32MultiArray` and its layout dependencies used by strict array/truncation
  acceptance.
- `SolidPrimitive`, `Polygon` and `Point32` used by the existing strict bounds
  acceptance. Python construction remains available, but encoding/decoding this
  bounded type remains unsupported. CDR-L07/L08 assertions remain strict xfails.

All 14 project-owned types remain exported. The pinned source lock selects the upstream definitions needed by this package.
Other upstream types are not part of the default installed API. This is a deliberate proposal API reduction from
153 exports, not a claim that static imports prove every other type unused.
Dynamic topic lookup and typed replay accept installed, supported type names.
External packages can reuse the exported dependency-owned types, including the
same Header identity. We do not promise an arbitrary omitted standard type or
split ROS-package ownership across Rust crates. There is no core/extras scheme.

## Install and use

Built-in Python messages and their schemas ship inside the main `dimos` wheel.
There is no separate `dimos-generated` Python distribution to install or publish.
Until this proposal is released, install the matching review wheel:

```sh
pip install review/wheels/dimos-*.whl
```

After release, ordinary users just run `pip install dimos`. Imports stay the same:

```python
from dimos.messages.builtin_interfaces.msg import Time
from dimos.messages.std_msgs.msg import Header
from dimos_cdrgen.registry import encode, decode

header = Header(Time(17, 23), "map")
assert decode(encode(header), Header) == header
```

CI prepares pinned inputs, generates the catalog through `dimos-cdrgen`, and
packages `dimos.messages`, `dimos.messages_schemas`, licenses and the
`dimos.messages` provider entry point into the main wheel. Message versions
follow the root `dimos` project version. Python messages use rosbags; installing
a wheel generates nothing and adds no message compiler requirement. The main
package's existing native extensions have their own platform wheel requirements.

## Source and editable builds

A maintainer prepares inputs explicitly in a checkout before normal builds:

```sh
uv run --no-project --with 'dimos-cdrgen[prepare]==0.1.1' python -m scripts.prepare_builtin_sources
pip install .
# Or for development:
pip install -e .
```

For cached archives, pass `--offline` to the preparation command. Source
preparation verifies archive and selected-file hashes and includes the licenses.
It never runs inside pip hooks or runtime imports. An unprepared checkout fails
with the missing-input path rather than downloading schemas.

The root PEP 517/660 build uses the shared generator and configuration in
`pyproject.toml`. Only the existing main-package extension is compiled; Python
message generation runs no CMake or Cargo. Build isolation installs the declared
Python build dependencies. Offline builds need those dependencies cached locally.
A release sdist includes prepared inputs, catalog, pins and licenses, so it can
rebuild outside the checkout without fetching schemas. The normal main-package
compiler and web-bundle prerequisites still apply to source builds.

After changing a definition, rerun preparation and reinstall editable output,
then restart existing Python processes. External message projects declare
`dimos` as their Python build/runtime dependency and use the installed
`dimos.messages` schema owner at the matching DimOS version:
```toml
[tool.dimos.messages]
python-module = "my_robot.messages"

[tool.dimos.messages.dependencies]
"dimos.messages" = "0.0.14"
```

The version above must match the installed DimOS artifact. `python-module`
controls Python imports; the retained `module = "dimos_generated"` in the main
project controls CMake/Cargo and schema ownership metadata. Their generated
classes import the same `Header`; they must not package another copy. Use the
source project's own packaging metadata rather than publishing the generator's
standalone intermediate Python project, which infers distribution names from
module names. C++ and Rust source artifacts remain separate native workflows.

## Maintainer artifacts and checks

```sh
python -m scripts.generate_builtin_messages
python -m scripts.generate_builtin_messages --check
```

Outputs live in `build/message-source/python`. The check
compares independent regeneration with those build artifacts and does not repair
them. Artifact acceptance verifies wheel, source and editable installs and external
Header identity. Undeclared source additions require a catalog decision.

`bash scripts/package_messages.sh` creates the explicit C++ source archive and
Cargo crate using the same roots/closure. It invokes Cargo as a native packaging
step. Consumers still supply their native toolchains and prepared dependencies.
MCAP writers can embed the complete schema closure of a supported type even when
its nested values come from another package.
