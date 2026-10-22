# Dimos gateway: how do I...?

The stable API for building apps on top of dimos. It covers the catalog of what could run (blueprints, modules, RPCs,
robots, config schemas), what is running (`/active`: launch, stop, its module instances, their RPCs and streams) and
what ran (`/history`: logs and recordings). It saves no config: the caller sends all of a launch's config with the
launch. It's versioned (`GET /info` `api`, the `desktop-gateway-v` tags); `GET /openapi.json` lists every route.

Snippets run in an app page served by Desktop at `/apps/<name>/`, so `../../dimos-gateway/` is the gateway
(Desktop serves it under `/dimos-gateway`; the gateway's own routes have no prefix). An app's `dimos.yaml` lists
each gateway route it calls:

```yaml
uses:
    "@dimos-gateway":
        - GET /catalog/blueprints
        - POST /active/launch
```

## How do I list blueprints?

```js
const { blueprints, scan } = await (await fetch("../../dimos-gateway/catalog/blueprints")).json()
// blueprints: [{ name, kind, robot, title, description, starter, hidden, recommended_config, modules,
//   importable, import_error, suggested_extras, scanned_at }]; the scan's fields are null until a scan
// scan: { job, state, scanned, finished_at, age_seconds, stale }: the scan is cached (on disk too) and only a POST
// catalog/scan changes it; stale says the checkout or venv changed since
const { blueprints: names } = await (await fetch("../../dimos-gateway/catalog/blueprints?fields=blueprints.name")).json()
```

To scan (the first time, or after a change) and follow it:

```js
const { job } = await (await fetch("../../dimos-gateway/catalog/scan", { method: "POST" })).json() // 202
const { done, ok, progress } = await (await fetch(`../../dimos-gateway/jobs/${job}`)).json() // progress: { done, total, current }
await fetch("../../dimos-gateway/catalog/scan/blueprints/unitree-go2", { method: "POST" }) // just one blueprint: seconds
```

## How do I launch a blueprint?

```js
await fetch("../../dimos-gateway/active/launch", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
        blueprint: "unitree-go2",
        config: { global: { replay: true, n_workers: 4 }, modules: { go2connection: { lidar: false } } },
    }),
}) // answers at once, phase "starting"; 409 if something runs; { module: "..." } launches one module on its own
await fetch("../../dimos-gateway/active/exec", { method: "POST", body: JSON.stringify({ args: ["unitree-go2", "--replay"] }) }) // exactly `dimos run <args>`
```

The gateway saves no config: keep your defaults (global and per blueprint) yourself and send all of them with each
launch. Secret values (keys named `*_key`, `*token*`, ...) go to dimos through the environment, never on its command
line, and come back as `•••`.

## How do I see what's running?

```js
const now = await (await fetch("../../dimos-gateway/active")).json()
// { running, runId, pid, blueprint, startedAt, logDir, launch: { phase, output, config, error, ... } | null }
getZenoh().subscribeDimos("launch", (launch) => show(launch.phase)) // live phase changes; getZenoh from ./dim-app/mod.js
const { name, config, modules } = await (await fetch("../../dimos-gateway/active/blueprint")).json() // 409 when idle
```

## How do I stop or relaunch it?

```js
await fetch("../../dimos-gateway/active/stop", { method: "POST" }) // exactly `dimos stop`
await fetch("../../dimos-gateway/active/relaunch", { method: "POST" }) // the same launch again (one started in a terminal: a copy of its command)
```

## How do I read a run's logs?

```js
const page = await (await fetch("../../dimos-gateway/active/log?level=warning&limit=200")).json() // { runId, records, offset }
const newer = await (await fetch(`../../dimos-gateway/active/log?after=${page.offset}`)).json() // tail
const { runs } = await (await fetch("../../dimos-gateway/history")).json() // [{ runId, blueprint, startedAt, endedAt, running }]
const old = await (await fetch(`../../dimos-gateway/history/${runs[1].runId}/logs`)).json() // or history/latest/logs
const { path } = await (await fetch("../../dimos-gateway/history/recordings")).json() // where recordings (mcap, db) go
```

## How do I see the running modules, their RPCs and streams?

```js
const { modules } = await (await fetch("../../dimos-gateway/active/modules")).json()
// [{ id: "robot0:go2connection", class, catalogId, namespace, doc, rpcs: [{ name, params, doc, skill }], streams }]
await fetch("../../dimos-gateway/active/modules/robot0:go2connection/restart", { method: "POST" }) // reloads its source
await fetch("../../dimos-gateway/active/modules/robot0:go2connection/stop", { method: "POST" }) // unloads just that one
const { streams } = await (await fetch("../../dimos-gateway/active/streams")).json() // [{ name, type, direction, channel, module }]
const frame = await (await fetch("../../dimos-gateway/active/streams/odom/peek?timeout=2")).json() // one message; 504 on timeout
```

An instance's id is its namespace and lowercase class joined by `:` (`go2connection` with no namespace).

## How do I list a blueprint's modules, their streams and RPCs?

```js
const { modules } = await (await fetch("../../dimos-gateway/catalog/blueprints/unitree-go2")).json()
// modules[i]: { name, rpcs: [{ name, params, doc, skill }], streams: [{ name, type, direction, topic }] }
```

## How do I list every module's inputs and outputs?

```js
const { modules } = await (await fetch("../../dimos-gateway/catalog/modules")).json() // [{ name, inputs, outputs, rpcs, blueprints }]
const cam = await (await fetch("../../dimos-gateway/catalog/modules/go2_connection")).json() // + rpcs (with params, doc, skill), file, line
```

## How do I list RPCs (and skills)?

```js
const { skills } = await (await fetch("../../dimos-gateway/active/skills")).json() // the running ones with skill: true
const { rpcs } = await (await fetch("../../dimos-gateway/active/modules/go2connection/rpcs")).json()
const { rpcs: all } = await (await fetch("../../dimos-gateway/catalog/rpcs")).json() // every module's, from the last scan
```

## How do I call an RPC (or a skill)?

```js
const answer = await (await fetch("../../dimos-gateway/active/modules/unitreeskillcontainer/rpcs/execute_sport_command", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ kwargs: { command_name: "FrontJump" } }),
})).json() // { ok, result, error }; this acts on the robot, so call it only from a user's action
```

This is for buttons: the call goes straight to the module, not through MCP, so a skill's capability hold (movement...)
doesn't apply. Agents use the python API.

## How do I decode dimos messages in a page?

```js
import { DimApp } from "./dim-app/mod.js"

const app = new DimApp({ msgDecodeEndpoint: "../../dimos-gateway/msgs.js" })
app.subscribe("odom", (odom) => console.log(odom.pose.pose.position)) // decoded
const twist = app.msgs.geometry_msgs.Twist.encode({ linear: { x: 0.3 } }) // bytes
```

## How do I list robots and their recommended blueprints?

```js
const { robots } = await (await fetch("../../dimos-gateway/catalog/robots")).json()
robots.go2.recommended // ["unitree-go2-basic", ...]
robots.go2.blueprints // names; each one's title, description, starter and recommended_config are on its blueprint
```

## How do I get dimos's python?

```js
const { python } = await (await fetch("../../dimos-gateway/info")).json() // run `${python} -c 'import dimos'`
```

## How do I read a blueprint's config schema?

```js
const schema = await (await fetch("../../dimos-gateway/catalog/blueprints/unitree-go2/config")).json()
// { global: { robot_ip: { type, default, description, secret } }, modules: { go2connection: { class, args: {...} } } }
const globals = await (await fetch("../../dimos-gateway/catalog/config")).json() // { schema, defaults, secrets }
const one = await (await fetch("../../dimos-gateway/catalog/modules/go2connection/config")).json() // { name, class, args }
```

## How do I run it myself?

Desktop starts it (dimos.yaml `start:`), but it also runs on its own from a dimos checkout:

```sh
.venv/bin/python -m experimental.gateway --detach      # in the background, on its unix socket
.venv/bin/python -m experimental.gateway --port 8123   # in the foreground, also on --host (default 127.0.0.1):8123
```

- Its routes have no prefix: Desktop serves them under `/dimos-gateway/`, so `/active` here is `/dimos-gateway/active` there.
- It reads `~/.dimos/config.yaml` (or `$DIMOS_HOME/config.yaml`): `dimos-gateway.socket`, `zenoh.namespace`,
  `zenoh.connect`, `zenoh.gossip` and `recordings.dir`. Command-line flags (`--socket`, `--zenoh-namespace`, ...) win.
- Events go on zenoh at `<namespace>/dimos/events/<type>` (`launch`, `blueprints`, `upload`, `uploads`, `upload-removed`,
  `cloud-login`); without a namespace there are none.
- Its tuning numbers (timeouts, cache lifetimes, limits) are in `experimental/gateway/constants.py`.
- Robot and blueprint descriptions come from `experimental/gateway/annotations.yaml`.
- Anything that imports a blueprint runs in a child process, so a blueprint that hangs or crashes can't take it down.
- Its last launch, including what `/active/relaunch` re-runs (0600), is kept in `$XDG_STATE_HOME/dimos/gateway/`.
- Tests: `python -m pytest experimental/gateway/tests` and `python -m mypy experimental/gateway`.
