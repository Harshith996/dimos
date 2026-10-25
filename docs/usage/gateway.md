# The dimos gateway

The stable API for building apps on top of dimos: what could run (`/catalog`), what is running (`/active`) and what
ran (`/history`). Desktop starts it from dimos.yaml's `start:` and serves it at `/dimos-gateway/`; an app page at
`/apps/<name>/` reaches it as `../../dimos-gateway/`. `GET /openapi.json` has every route with its request and answer;
`GET /info` `api` is the API version (released as `desktop-gateway-v<version>` tags).

## Rules

1. **Relative URLs only:** `../../dimos-gateway/active`, never a host or port.
2. **Declare every route you call** in your app's `dimos.yaml`: `uses: {"@dimos-gateway": [GET /active, POST /active/launch]}`.
3. **Config is per launch.** The gateway saves none: keep your defaults and send all of them in each
   `POST /active/launch` `{blueprint, config: {global, modules}}`. `GET /catalog/blueprints/{name}/config` is the schema.
4. **Secrets** (keys like `*_key`, `*token*`) reach dimos through the environment, never its command line, and come
   back as `•••`.
5. **One blueprint runs at a time.** `/active` routes answer 409 when nothing runs; `POST /active/launch` answers 409
   when something does.
6. **A module instance's id** is its namespace and lowercase class joined by `:` (`robot0:go2connection`, or
   `go2connection` with no namespace).
7. **RPCs are for buttons.** `POST /active/modules/{id}/rpcs/{name}` calls the module directly, not through MCP, so a
   skill's capability holds don't apply; call one only from a user's action. Agents use the python API.
8. **The catalog is a cache.** It changes only when a `POST /catalog/scan` job finishes (`?timeout=` per blueprint);
   follow the job at `GET /jobs/{job}`.

## Routes

| | |
| -- | -- |
| `/catalog` | `blueprints`, `modules`, `rpcs`, `robots`, `config` (global schema), `scan`, each `{name}/config` schema |
| `/active` | the running blueprint; `launch`, `exec`, `relaunch`, `stop`, `log`, `blueprint`, `modules/{id}` (+ `restart`, `stop`, `rpcs/{name}`, `streams`), `skills`, `streams` (+ `{name}/peek`), `stats` |
| `/history` | every run with a log; `{runId}` (or `latest`), `{runId}/logs`, `recordings` (its directory) |
| other | `info`, `extras` (+ `install`), `jobs/{job}`, `docs/search`, `server/stop` |

## Examples

```js
const gateway = "../../dimos-gateway"
const post = (path, body) =>
    fetch(`${gateway}/${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })

await post("active/launch", { blueprint: "unitree-go2", config: { global: { replay: true } } }) // phase "starting"
const now = await (await fetch(`${gateway}/active`)).json() // { running, runId, blueprint, launch: { phase, output, ... } }
const { modules } = await (await fetch(`${gateway}/active/modules`)).json() // [{ id, class, rpcs, streams }]
await post("active/modules/unitreeskillcontainer/rpcs/execute_sport_command", { kwargs: { command_name: "FrontJump" } })
const page = await (await fetch(`${gateway}/active/log?level=warning&limit=200`)).json() // then ?after=${page.offset}
```

Live launch phases (and added or removed blueprints) also arrive on zenoh at `<namespace>/dimos/events/<type>`
(`getZenoh().subscribeDimos("launch", ...)` from `dim-app/mod.js`).

A change to the shared `config.yaml` arrives the same way, one topic per setting:
`<namespace>/dimos/events/config/<setting>`, where `<setting>` is a key of `GET /config`
(`recordingsDir`, `path`). The payload is `{"type": "config", "setting": "<setting>"}` and carries no
value: `GET /config` stays the one answer, so a listener's cache and the event can never say
different things. Subscribe to the setting you read, or to `<namespace>/dimos/events/**` for
everything. A single `*` matches one chunk, so it would miss these.

## Running it yourself

```sh skip
.venv/bin/python -m experimental.gateway --detach      # in the background, on its unix socket
.venv/bin/python -m experimental.gateway --port 8123   # in the foreground, also on --host (default 127.0.0.1):8123
```

- `DIMOS_APP` is what Desktop starts it with: which component this is (`name`) and the home whose config.yaml to read
  (`home`), as one JSON object rather than a variable per setting. `DIMOS_HOME` still works and still means the same
  thing. Children inherit the variable; nothing reads it back out of their environment.
- Settings come from `~/.dimos/config.yaml` (or `$DIMOS_HOME/config.yaml`): `dimos-gateway.socket`, `zenoh.namespace`,
  `zenoh.connect`, `zenoh.gossip`, `recordings.dir`. Flags (`--socket`, `--zenoh-namespace`, ...) win.
  `GET /config` answers the settings that can change while it runs (the file's path, the recordings directory);
  the ones a flag can win are fixed at startup, so `GET /info` has those.
- Its tuning numbers are in `experimental/gateway/constants.py`; robot and blueprint descriptions in
  `experimental/gateway/annotations.yaml`.
- Anything that imports a blueprint runs in a child process, so a blueprint that hangs or crashes can't take it down.
- Tests: `python -m pytest experimental/gateway/tests` and `python -m mypy experimental/gateway`.
