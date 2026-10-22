# dimos gateway

`python -m experimental.gateway --socket <path> --dimos-dir <checkout>` serves the stable API for building apps on top of
dimos (versioned by `desktop-gateway-v` tags; only its code lives in `experimental/`): the catalog (blueprints, modules, RPCs, robots, config schemas), the running blueprint
(`/active`: its module instances, RPCs and streams, through one `Dimos.connect()`), past runs (`/history`), Dimensional cloud uploads, and the
extras installs and blueprint scan (as jobs). It saves no config: each launch carries all of its own. `--port 8123` also serves it on `127.0.0.1:8123`; `--detach` starts it in the
background. Routes have no prefix: Desktop puts them under `/dimos-gateway`, so `/active` here is `/dimos-gateway/active`
through Desktop.

- `GET /openapi.json` lists every route (FastAPI's own document); `x-slow: true` marks the ones that import dimos.
- Events go on zenoh at `<ns>/dimos/events/<type>` (`launch`, `blueprints`, `upload`, `uploads`, `upload-removed`,
  `cloud-login`); the namespace is `--zenoh-namespace` or `$DIMOS_ZENOH_NAMESPACE`, with no namespace there are no events.
- The last launch (its record, and what it was started with for `/active/relaunch`, 0600) lives in `$XDG_STATE_HOME/dimos/gateway/`.
- `annotations.yaml` describes every robot and its blueprints (`GET /catalog/robots`, and on each blueprint in `GET /catalog/blueprints`).
- Anything that imports a blueprint runs in a child process (`introspect.py`), so a blueprint that hangs or crashes
  can't take the gateway down.

See [docs/usage/gateway.md](/docs/usage/gateway.md) for examples.

```sh
python -m pytest experimental/gateway/tests
python -m mypy experimental/gateway
```
