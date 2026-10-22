# Go2 mapping

## Real time mapping

The default blueprint maps dynamic environments in real time.

`dimos --robot-ip ... run unitree-go2`

## Saving maps between runs

`dimos --robot-ip ... run unitree-go2-map`

The map blueprint saves the map between runs, relocalizes against it, and updates it as the robot explores more or the environment changes. The map is kept in `map_go2.map.db`.

To relocalize against a prerecorded map without updating it, give the default blueprint a map file:

`dimos --robot-ip ... run unitree-go2 --map-file=<map>` (loads `<map>.map.db`)

## A hand-built model as the map

`ModelMesh` (`dimos/mapping/experimental/model_mesh.py`) shows a hand-built model (a glTF) in place of the lidar mesh. It relocalizes against the recorded lidar map the model was built on, not against the model: a clean model has no clutter or outdoors and its surfaces sit 10-30 cm off the real ones, so a live scan matches it poorly. The model rides on each fix at a static `model_pose` in the map frame (zero when it was authored in that frame) and goes out on `mesh` in `odom`, chunked like the lidar mesher's output.

The map is a directory with `lidar.pc2.lcm` (the recorded map, frame `map`) and `model.glb`, or a `cloud.pc2.lcm,mesh.glb` pair. The robot runs `go2-nav` without a map; the viewer machine, which holds the directory, runs

`dimos run go2-nav-viewer-model --map-file=<dir>`

or, as one hosted run, `dimos run go2-nav-model-hosted --map-file=<dir>`. Fixes go out as `odom -> map` and repeat every `reloc_interval` (10 s) to correct drift; the recorded map seeds the ray tracer like any premap.
