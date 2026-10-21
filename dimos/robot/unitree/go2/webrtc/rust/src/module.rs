// Copyright 2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! GO2Connection's real-robot side: the Go2 over its WebRTC link. `cmd_vel` and `go2_request` (GO2Connection's
//! RPCs as JSON) in; `odom` (+ its tf chain), `lidar`, the decoded front camera on `color_image`, `camera_info` and
//! the RPC answers on `go2_response` out. Poses are stamped with the Go2's own stamp moved onto the host clock
//! (clock.rs); frames with their first packet's arrival, de-jittered by their RTP tick spacing (CaptureClock), and
//! scans with their arrival, since the Go2 sends no usable capture time for either.
//! One thread decodes video, one the data channel; tokio runs the session and the requests.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

use dimos_module::{native_config, Input, Module, Output};
use lcm_msgs::geometry_msgs::{
    Point, Pose, PoseStamped, Quaternion, Transform, TransformStamped, Twist, Vector3,
};
use lcm_msgs::sensor_msgs::{CameraInfo, Image, PointCloud2, PointField};
use lcm_msgs::std_msgs::{self, Header, Time};
use lcm_msgs::tf2_msgs::TFMessage;
use serde_json::{json, Value};
use tokio::runtime::Handle;
use tokio::sync::Notify;
use tracing::{error, info, warn};

use crate::clock::{now_ns, place, CaptureClock, HostClock};
use crate::conn::{Conn, Event};
use crate::h264::{Depacketizer, Packet};
use crate::lidar::decode_voxels;
use crate::timing::{secs, Timing};

const SPORT: &str = "rt/api/sport/request";
const MOTION_SWITCHER: &str = "rt/api/motion_switcher/request";
const VUI: &str = "rt/api/vui/request";
const OBSTACLES_AVOID: &str = "rt/api/obstacles_avoid/request";
const WIRELESS_CONTROLLER: &str = "rt/wirelesscontroller";
const LIDAR_TOPIC: &str = "rt/utlidar/voxel_map_compressed";
const POSE_TOPIC: &str = "rt/utlidar/robot_pose";
const LOWSTATE_TOPIC: &str = "rt/lf/lowstate";

const BALANCE_STAND: i64 = 1002;
const STAND_UP: i64 = 1004;
const STAND_DOWN: i64 = 1005;
const MOVE: i64 = 1008;
const SWITCH_JOYSTICK: i64 = 1027;
const RAGE_MODE: i64 = 2059;
/// The motion switcher's CheckMode / SelectMode.
const CHECK_MODE: i64 = 1001;
const SELECT_MODE: i64 = 1002;
const VUI_SET_BRIGHTNESS: i64 = 1005;
const OBSTACLES_AVOID_SWITCH_SET: i64 = 1001;

#[native_config]
#[derive(Clone)]
pub struct Config {
    pub ip: String,
    /// Per-device AES-128 key (Go2 firmware >= 1.1.15), "" for older firmware.
    pub aes_128_key: Secret,
    /// "default" or "rage" (Rage Mode switched on at start).
    pub mode: String,
    pub lidar: bool,
    pub camera: bool,
    /// Drive with sport Move (1008) instead of the wireless-controller stick emulation.
    pub velocity_api: bool,
    /// "mcf" (stairs), "normal", or "" to leave the motion controller as it is.
    pub motion_mode: String,
    /// The robot's built-in obstacle avoidance, applied at start.
    pub obstacle_avoidance: bool,
    /// TF parent of the odometry (odom_frame_id -> base_link).
    pub odom_frame_id: String,
    pub publish_tf: bool,
    /// Namespace for this robot's frames ("" for none).
    pub frame_id_prefix: String,
    /// Reconnect when the Go2 drops the link (or refuses the first connect) instead of exiting.
    pub auto_reconnect: bool,
    #[serde(with = "camera_info_dict")]
    pub camera_info: CameraInfo,
    pub timing: Timing,
}

/// A config string kept out of logs: the module prints its config at start.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Secret(pub String);

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_empty() {
            "\"\""
        } else {
            "<redacted>"
        })
    }
}

/// python's `CameraInfo` config dict <-> the message; the stamp is set per publish. (As the Go2 DDS module's.)
mod camera_info_dict {
    use lcm_msgs::sensor_msgs::CameraInfo;
    use lcm_msgs::std_msgs::Header;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[allow(non_snake_case)]
    struct Dict {
        frame_id: String,
        height: i32,
        width: i32,
        distortion_model: String,
        D: Vec<f64>,
        K: [f64; 9],
        R: [f64; 9],
        P: [f64; 12],
        #[serde(default)]
        binning_x: i32,
        #[serde(default)]
        binning_y: i32,
    }

    pub fn deserialize<'de, De: Deserializer<'de>>(d: De) -> Result<CameraInfo, De::Error> {
        let c = Dict::deserialize(d)?;
        Ok(CameraInfo {
            header: Header {
                frame_id: c.frame_id,
                ..Default::default()
            },
            height: c.height,
            width: c.width,
            distortion_model: c.distortion_model,
            D: c.D,
            K: c.K,
            R: c.R,
            P: c.P,
            binning_x: c.binning_x,
            binning_y: c.binning_y,
            ..Default::default()
        })
    }

    pub fn serialize<S: Serializer>(c: &CameraInfo, s: S) -> Result<S::Ok, S::Error> {
        Dict {
            frame_id: c.header.frame_id.clone(),
            height: c.height,
            width: c.width,
            distortion_model: c.distortion_model.clone(),
            D: c.D.clone(),
            K: c.K,
            R: c.R,
            P: c.P,
            binning_x: c.binning_x,
            binning_y: c.binning_y,
        }
        .serialize(s)
    }
}

fn prefixed(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}/{name}")
    }
}

fn time_of_ns(ns: u64) -> Time {
    Time {
        sec: (ns / 1_000_000_000) as i32,
        nsec: (ns % 1_000_000_000) as i32,
    }
}

fn header(ns: u64, frame_id: &str) -> Header {
    Header {
        seq: 0,
        stamp: time_of_ns(ns),
        frame_id: frame_id.to_string(),
    }
}

fn f(value: &Value) -> f64 {
    value.as_f64().unwrap_or(0.0)
}

/// The Go2's time for a pose (`header.stamp` sec/nanosec), ns on its clock.
fn pose_robot_ns(data: &Value) -> Option<i64> {
    let stamp = &data["header"]["stamp"];
    Some(stamp["sec"].as_i64()? * 1_000_000_000 + stamp["nanosec"].as_i64().unwrap_or(0))
}

/// What the session shares with the request handlers and the threads.
struct Shared {
    config: Config,
    handle: Handle,
    conn: Mutex<Option<Arc<Conn>>>,
    lowstate: Mutex<Option<Value>>,
    /// the last cmd_vel's arrival (ns) while the base is being driven, 0 once stopped
    driving_since: AtomicU64,
    move_ids: AtomicU64,
    stopping: AtomicBool,
    lost: Notify,
}

impl Shared {
    fn conn(&self) -> Result<Arc<Conn>, String> {
        self.conn
            .lock()
            .unwrap()
            .clone()
            .filter(|c| !c.is_closed())
            .ok_or_else(|| "the Go2 isn't connected".into())
    }

    async fn request(
        &self,
        topic: &str,
        api_id: i64,
        parameter: Option<Value>,
    ) -> Result<Value, String> {
        self.conn()?
            .request(
                topic,
                api_id,
                parameter.as_ref(),
                secs(self.config.timing.answer_timeout),
            )
            .await
    }

    async fn sport(&self, api_id: i64, parameter: Option<Value>) -> Result<Value, String> {
        self.request(SPORT, api_id, parameter).await
    }

    /// One velocity command, as GO2Connection sent it: the stick emulation, or sport Move with velocity_api.
    async fn drive(&self, x: f64, y: f64, yaw: f64) -> Result<(), String> {
        let conn = self.conn()?;
        if self.config.velocity_api {
            let id = self.move_ids.fetch_add(1, Ordering::Relaxed) + 1;
            let payload = json!({
                "header": { "identity": { "id": id, "api_id": MOVE } },
                "parameter": json!({ "x": x, "y": y, "z": yaw }).to_string(),
            });
            return conn.send_req(SPORT, payload).await;
        }
        conn.publish(
            WIRELESS_CONTROLLER,
            json!({ "lx": -y, "ly": x, "rx": -yaw, "ry": 0 }),
        )
        .await
    }

    async fn stop_movement(&self) -> Result<(), String> {
        self.driving_since.store(0, Ordering::Relaxed);
        self.drive(0.0, 0.0, 0.0).await
    }

    async fn set_rage_mode(&self, enable: bool) -> Result<Value, String> {
        // BalanceStand first; the FSM needs time to settle before SwitchJoystick
        if let Err(err) = self.sport(BALANCE_STAND, None).await {
            warn!("balance stand before the rage toggle failed: {err}");
        }
        tokio::time::sleep(secs(self.config.timing.rage_balance_settle)).await;
        self.sport(RAGE_MODE, Some(json!({ "data": enable })))
            .await?;
        tokio::time::sleep(secs(self.config.timing.rage_switch_settle)).await;
        self.sport(SWITCH_JOYSTICK, Some(json!({ "data": true })))
            .await
    }

    async fn set_motion_mode(&self, name: &str) -> Result<(), String> {
        let current = self
            .request(MOTION_SWITCHER, CHECK_MODE, None)
            .await
            .ok()
            .and_then(|answer| serde_json::from_str::<Value>(answer["data"]["data"].as_str()?).ok())
            .and_then(|data| data["name"].as_str().map(str::to_string));
        if current.as_deref() == Some(name) {
            return Ok(());
        }
        self.request(MOTION_SWITCHER, SELECT_MODE, Some(json!({ "name": name })))
            .await?;
        tokio::time::sleep(secs(self.config.timing.motion_mode_settle)).await;
        Ok(())
    }

    /// A GO2Connection RPC: `verb` with its `args`, answered with its result.
    async fn rpc(&self, verb: &str, args: &Value) -> Result<Value, String> {
        let flag = |name: &str| args[name].as_bool().unwrap_or(true);
        let answered = |r: Result<Value, String>| r.map(|_| json!(true));
        match verb {
            "move" => {
                let (x, y, yaw) = (f(&args["x"]), f(&args["y"]), f(&args["yaw"]));
                let duration = f(&args["duration"]);
                if duration > 0.0 {
                    let until = now_ns() + (duration * 1e9) as u64;
                    while now_ns() < until {
                        self.drive(x, y, yaw).await?;
                        tokio::time::sleep(secs(self.config.timing.move_period)).await;
                    }
                    self.stop_movement().await?;
                } else {
                    self.driving_since.store(now_ns(), Ordering::Relaxed);
                    self.drive(x, y, yaw).await?;
                }
                Ok(json!(true))
            }
            "stop_movement" => self.stop_movement().await.map(|_| Value::Null),
            "standup" => answered(self.sport(STAND_UP, None).await),
            "liedown" => answered(self.sport(STAND_DOWN, None).await),
            "balance_stand" => answered(self.sport(BALANCE_STAND, None).await),
            "sport_command" => answered(
                self.sport(
                    args["api_id"]
                        .as_i64()
                        .ok_or("sport_command needs api_id")?,
                    None,
                )
                .await,
            ),
            "set_rage_mode" => answered(self.set_rage_mode(flag("enable")).await),
            "switch_joystick" => answered(
                self.sport(SWITCH_JOYSTICK, Some(json!({ "data": flag("enable") })))
                    .await,
            ),
            "set_light" => {
                let level = args["level"].as_i64().unwrap_or(0).clamp(0, 10);
                answered(
                    self.request(
                        VUI,
                        VUI_SET_BRIGHTNESS,
                        Some(json!({ "brightness": level })),
                    )
                    .await,
                )
            }
            "set_obstacle_avoidance" => {
                let enable = flag("enabled") as i64;
                answered(
                    self.request(
                        OBSTACLES_AVOID,
                        OBSTACLES_AVOID_SWITCH_SET,
                        Some(json!({ "enable": enable })),
                    )
                    .await,
                )
            }
            "set_motion_mode" => self
                .set_motion_mode(args["name"].as_str().unwrap_or(""))
                .await
                .map(|_| Value::Null),
            "publish_request" => {
                let data = &args["data"];
                let api_id = data["api_id"]
                    .as_i64()
                    .ok_or("publish_request needs data.api_id")?;
                let parameter = data.get("parameter").cloned();
                self.request(args["topic"].as_str().unwrap_or(""), api_id, parameter)
                    .await
            }
            "lowstate" => Ok(self.lowstate.lock().unwrap().clone().unwrap_or(Value::Null)),
            _ => Err(format!("unknown GO2Connection request {verb:?}")),
        }
    }
}

#[derive(Module)]
#[module(name = "go2_webrtc", setup = start, teardown = stop)]
pub struct Go2WebRtc {
    #[input(decode = Twist::decode, handler = on_cmd_vel)]
    cmd_vel: Input<Twist>,

    #[input(decode = std_msgs::String::decode, handler = on_go2_request)]
    go2_request: Input<std_msgs::String>,

    #[output(encode = PoseStamped::encode)]
    odom: Output<PoseStamped>,

    #[output(encode = PointCloud2::encode)]
    lidar: Output<PointCloud2>,

    /// GO2Connection declares it (the Pointcloud spec); nothing publishes on it.
    #[allow(dead_code)]
    #[output(encode = PointCloud2::encode)]
    pointcloud: Output<PointCloud2>,

    #[output(encode = Image::encode)]
    color_image: Output<Image>,

    #[output(encode = CameraInfo::encode)]
    camera_info: Output<CameraInfo>,

    #[output(encode = TFMessage::encode)]
    tf: Output<TFMessage>,

    #[output(encode = std_msgs::String::encode)]
    go2_response: Output<std_msgs::String>,

    #[config]
    config: Config,

    shared: Option<Arc<Shared>>,
}

impl Go2WebRtc {
    async fn start(&mut self) {
        let shared = Arc::new(Shared {
            config: self.config.clone(),
            handle: Handle::current(),
            conn: Mutex::new(None),
            lowstate: Mutex::new(None),
            driving_since: AtomicU64::new(0),
            move_ids: AtomicU64::new(0),
            stopping: AtomicBool::new(false),
            lost: Notify::new(),
        });
        self.shared = Some(shared.clone());

        let (video_tx, video_rx) = mpsc::sync_channel(self.config.timing.video_queue_packets);
        let video_dropped = Arc::new(AtomicBool::new(false));
        let (data_tx, data_rx) = mpsc::channel();
        if self.config.camera {
            let video = VideoLoop {
                shared: shared.clone(),
                color_image: self.color_image.clone(),
                camera_info: self.camera_info.clone(),
                dropped: video_dropped.clone(),
            };
            std::thread::Builder::new()
                .name("go2-video".into())
                .spawn(move || video.run(video_rx))
                .expect("spawn the video thread");
        }
        let data = DataLoop {
            shared: shared.clone(),
            odom: self.odom.clone(),
            lidar: self.lidar.clone(),
            tf: self.tf.clone(),
        };
        std::thread::Builder::new()
            .name("go2-data".into())
            .spawn(move || data.run(data_rx))
            .expect("spawn the data thread");

        let on_event: crate::conn::OnEvent = {
            let shared = shared.clone();
            let video_tx: SyncSender<Event> = video_tx;
            let data_tx = Mutex::new(data_tx);
            Arc::new(move |event| match event {
                Event::Rtp { .. } => {
                    if let Err(TrySendError::Full(_)) = video_tx.try_send(event) {
                        video_dropped.store(true, Ordering::Relaxed);
                    }
                }
                Event::Data { .. } => {
                    let _ = data_tx.lock().unwrap().send(event);
                }
                Event::Lost => shared.lost.notify_one(),
            })
        };
        tokio::spawn(session(shared.clone(), on_event));
        tokio::spawn(deadman(shared));
    }

    async fn stop(&mut self) {
        if let Some(shared) = &self.shared {
            shared.stopping.store(true, Ordering::Relaxed);
            let _ = shared.stop_movement().await;
            let conn = shared.conn.lock().unwrap().take();
            if let Some(conn) = conn {
                conn.close().await;
            }
        }
    }

    async fn on_cmd_vel(&mut self, msg: Twist) {
        let Some(shared) = &self.shared else { return };
        shared.driving_since.store(now_ns(), Ordering::Relaxed);
        if let Err(err) = shared
            .drive(msg.linear.x, msg.linear.y, msg.angular.z)
            .await
        {
            warn!("cmd_vel: {err}");
        }
    }

    async fn on_go2_request(&mut self, msg: std_msgs::String) {
        let Some(shared) = self.shared.clone() else {
            return;
        };
        let response = self.go2_response.clone();
        tokio::spawn(async move {
            let request: Value = serde_json::from_str(&msg.data).unwrap_or(Value::Null);
            let id = request["id"].clone();
            let verb = request["verb"].as_str().unwrap_or("").to_string();
            let answer = match shared.rpc(&verb, &request["args"]).await {
                Ok(result) => json!({ "id": id, "ok": true, "result": result }),
                Err(err) => {
                    warn!(verb, "GO2Connection request failed: {err}");
                    json!({ "id": id, "ok": false, "error": err })
                }
            };
            let _ = response
                .publish(&std_msgs::String {
                    data: answer.to_string(),
                })
                .await;
        });
    }
}

/// Connects, sets the link up, runs GO2Connection's start sequence once, then waits for the link to die; with
/// auto_reconnect it connects again (no stand-up), otherwise the process exits so the run sees the Go2 is gone.
async fn session(shared: Arc<Shared>, on_event: crate::conn::OnEvent) {
    let config = &shared.config;
    let mut started = false;
    loop {
        info!(ip = %config.ip, "connecting to the Go2");
        match Conn::connect(
            &config.ip,
            &config.aes_128_key.0,
            &config.timing,
            on_event.clone(),
        )
        .await
        {
            Ok(conn) => {
                *shared.conn.lock().unwrap() = Some(conn.clone());
                if let Err(err) = link_up(&shared, &conn).await {
                    warn!("setting up the Go2 link: {err}");
                }
                if !started {
                    started = true;
                    start_sequence(&shared).await;
                }
                shared.lost.notified().await;
                conn.close().await;
                if shared.stopping.load(Ordering::Relaxed) {
                    return;
                }
                error!("the Go2 dropped the WebRTC link");
            }
            Err(err) => error!("connecting to the Go2 at {} failed: {err}", config.ip),
        }
        if shared.stopping.load(Ordering::Relaxed) {
            return;
        }
        if !config.auto_reconnect {
            error!("auto_reconnect is off: exiting");
            std::process::exit(1);
        }
        tokio::time::sleep(secs(config.timing.reconnect_pause)).await;
    }
}

/// What every connection needs: big messages unthrottled, the "ai" controller (unitree_webrtc_connect's default),
/// the subscriptions and the camera.
async fn link_up(shared: &Shared, conn: &Conn) -> Result<(), String> {
    conn.send_raw(
        "rtc_inner_req",
        json!({ "req_type": "disable_traffic_saving", "instruction": "on" }),
    )
    .await?;
    if let Err(err) = conn
        .request(
            MOTION_SWITCHER,
            SELECT_MODE,
            Some(&json!({ "name": "ai" })),
            secs(shared.config.timing.answer_timeout),
        )
        .await
    {
        warn!("selecting the ai motion controller: {err}");
    }
    if shared.config.lidar {
        conn.subscribe(LIDAR_TOPIC).await?;
    }
    conn.subscribe(POSE_TOPIC).await?;
    conn.subscribe(LOWSTATE_TOPIC).await?;
    if shared.config.camera {
        conn.send_raw("vid", json!("on")).await?;
        conn.request_keyframe().await;
    }
    Ok(())
}

/// GO2Connection.start's sequence on the robot: motion mode, stand, balance, rage, obstacle avoidance.
async fn start_sequence(shared: &Shared) {
    let config = &shared.config;
    if !config.motion_mode.is_empty() {
        if let Err(err) = shared.set_motion_mode(&config.motion_mode).await {
            warn!("motion mode {}: {err}", config.motion_mode);
        }
    }
    if let Err(err) = shared.sport(STAND_UP, None).await {
        warn!("stand up: {err}");
    }
    tokio::time::sleep(secs(config.timing.stand_up_settle)).await;
    if let Err(err) = shared.sport(BALANCE_STAND, None).await {
        warn!("balance stand: {err}");
    }
    if config.mode == "rage" {
        if let Err(err) = shared.set_rage_mode(true).await {
            warn!("rage mode: {err}");
        }
    }
    let enable = config.obstacle_avoidance as i64;
    if let Err(err) = shared
        .request(
            OBSTACLES_AVOID,
            OBSTACLES_AVOID_SWITCH_SET,
            Some(json!({ "enable": enable })),
        )
        .await
    {
        warn!("obstacle avoidance: {err}");
    }
}

/// Stops the base once `cmd_vel` has been silent for cmd_vel_timeout (checked ten times per timeout).
async fn deadman(shared: Arc<Shared>) {
    let timeout_ns = secs(shared.config.timing.cmd_vel_timeout).as_nanos() as u64;
    loop {
        tokio::time::sleep(secs(shared.config.timing.cmd_vel_timeout / 10.0)).await;
        let since = shared.driving_since.load(Ordering::Relaxed);
        if since != 0 && now_ns().saturating_sub(since) > timeout_ns {
            if let Err(err) = shared.stop_movement().await {
                warn!("deadman stop: {err}");
            }
        }
    }
}

struct DataLoop {
    shared: Arc<Shared>,
    odom: Output<PoseStamped>,
    lidar: Output<PointCloud2>,
    tf: Output<TFMessage>,
}

impl DataLoop {
    fn run(self, rx: Receiver<Event>) {
        let config = &self.shared.config;
        let prefix = config.frame_id_prefix.as_str();
        let (base_link, camera_link, camera_optical) = (
            prefixed(prefix, "base_link"),
            prefixed(prefix, "camera_link"),
            prefixed(prefix, "camera_optical"),
        );
        let mut pose_clock = HostClock::new(config.timing.clock_window, config.timing.clock_reset);
        let mut stats = Stats::default();
        while let Ok(event) = rx.recv() {
            let Event::Data {
                topic,
                json,
                binary,
                arrived_ns,
            } = event
            else {
                continue;
            };
            let data = &json["data"];
            match topic.as_str() {
                POSE_TOPIC => {
                    let robot_ns = pose_robot_ns(data);
                    let ns = place(&mut pose_clock, robot_ns, arrived_ns);
                    stats.pose(robot_ns, arrived_ns, &pose_clock);
                    let p = &data["pose"]["position"];
                    let o = &data["pose"]["orientation"];
                    let position = Point {
                        x: f(&p["x"]),
                        y: f(&p["y"]),
                        z: f(&p["z"]),
                    };
                    let orientation = Quaternion {
                        x: f(&o["x"]),
                        y: f(&o["y"]),
                        z: f(&o["z"]),
                        w: o["w"].as_f64().unwrap_or(1.0),
                    };
                    let odom = PoseStamped {
                        header: header(ns, &config.odom_frame_id),
                        pose: Pose {
                            position: position.clone(),
                            orientation: orientation.clone(),
                        },
                    };
                    if config.publish_tf {
                        let edge =
                            |parent: &str,
                             child: &str,
                             translation: Vector3,
                             rotation: Quaternion| TransformStamped {
                                header: header(ns, parent),
                                child_frame_id: child.to_string(),
                                transform: Transform {
                                    translation,
                                    rotation,
                                },
                            };
                        let transforms = vec![
                            edge(
                                &config.odom_frame_id,
                                &base_link,
                                Vector3 {
                                    x: position.x,
                                    y: position.y,
                                    z: position.z,
                                },
                                orientation,
                            ),
                            edge(
                                &base_link,
                                &camera_link,
                                Vector3 {
                                    x: 0.3,
                                    y: 0.0,
                                    z: 0.0,
                                },
                                Quaternion {
                                    x: 0.0,
                                    y: 0.0,
                                    z: 0.0,
                                    w: 1.0,
                                },
                            ),
                            edge(
                                &camera_link,
                                &camera_optical,
                                Vector3::default(),
                                Quaternion {
                                    x: -0.5,
                                    y: 0.5,
                                    z: -0.5,
                                    w: 0.5,
                                },
                            ),
                        ];
                        let _ = self
                            .shared
                            .handle
                            .block_on(self.tf.publish(&TFMessage { transforms }));
                    }
                    let _ = self.shared.handle.block_on(self.odom.publish(&odom));
                }
                LIDAR_TOPIC => {
                    let Some(binary) = binary else { continue };
                    let points = match decode_voxels(data, &binary) {
                        Ok(points) => points,
                        Err(err) => {
                            warn!("lidar: {err}");
                            continue;
                        }
                    };
                    // a scan's `stamp` comes as a float32 (128 s steps at today's epoch): arrival is the better time
                    let ns = arrived_ns;
                    let _ = self
                        .shared
                        .handle
                        .block_on(self.lidar.publish(&cloud(ns, points)));
                }
                LOWSTATE_TOPIC => *self.shared.lowstate.lock().unwrap() = Some(json),
                _ => {}
            }
        }
    }
}

/// xyz float32 points as a PointCloud2 in "world", as GO2Connection published the voxel map.
fn cloud(ns: u64, points: Vec<[f32; 3]>) -> PointCloud2 {
    let field = |name: &str, offset: i32| PointField {
        name: name.into(),
        offset,
        datatype: PointField::FLOAT32 as u8,
        count: 1,
    };
    let mut data = Vec::with_capacity(points.len() * 12);
    for point in &points {
        for value in point {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    PointCloud2 {
        header: header(ns, "world"),
        height: 1,
        width: points.len() as i32,
        fields: vec![field("x", 0), field("y", 4), field("z", 8)],
        is_bigendian: false,
        point_step: 12,
        row_step: 12 * points.len() as i32,
        data,
        is_dense: true,
    }
}

struct VideoLoop {
    shared: Arc<Shared>,
    color_image: Output<Image>,
    camera_info: Output<CameraInfo>,
    dropped: Arc<AtomicBool>,
}

impl VideoLoop {
    fn want_keyframe(&self) {
        if let Ok(conn) = self.shared.conn() {
            self.shared
                .handle
                .spawn(async move { conn.request_keyframe().await });
        }
    }

    fn run(self, rx: Receiver<Event>) {
        let frame_id = prefixed(&self.shared.config.frame_id_prefix, "camera_optical");
        // NoFlush: flushing after every decode errors on a stream with reordered frames
        let decoder_config = openh264::decoder::DecoderConfig::new()
            .flush_after_decode(openh264::decoder::Flush::NoFlush);
        let mut decoder = match openh264::decoder::Decoder::with_api_config(
            openh264::OpenH264API::from_source(),
            decoder_config,
        ) {
            Ok(decoder) => decoder,
            Err(err) => {
                error!("openh264 didn't start, no color_image: {err}");
                return;
            }
        };
        let mut depacketizer = Depacketizer::default();
        let mut capture = CaptureClock::new(self.shared.config.timing.capture_max_waiting);
        // when each RTP time's first packet arrived (a frame's packets share one)
        let mut first_arrival: VecDeque<(u32, u64)> = VecDeque::new();
        let mut rgb = Vec::new();
        let mut last_info_ns = 0u64;
        let mut last_keyframe_ask_ns = 0u64;
        let keyframe_retry_ns =
            secs(self.shared.config.timing.keyframe_retry_period).as_nanos() as u64;
        let mut stats = Stats::default();
        while let Ok(event) = rx.recv() {
            let Event::Rtp { packet, arrived_ns } = event else {
                continue;
            };
            // a lost keyframe request (or one the Go2 ignored) would leave the camera dark until its next keyframe
            if !depacketizer.synced()
                && arrived_ns.saturating_sub(last_keyframe_ask_ns) > keyframe_retry_ns
            {
                last_keyframe_ask_ns = arrived_ns;
                self.want_keyframe();
            }
            if self.dropped.swap(false, Ordering::Relaxed) {
                // packets were dropped before this one: the frame in progress is broken
                depacketizer = Depacketizer::default();
                capture = CaptureClock::new(self.shared.config.timing.capture_max_waiting);
                self.want_keyframe();
            }
            if first_arrival
                .back()
                .is_none_or(|&(rtp, _)| rtp != packet.header.timestamp)
            {
                first_arrival.push_back((packet.header.timestamp, arrived_ns));
                if first_arrival.len() > 64 {
                    first_arrival.pop_front();
                }
            }
            let was_synced = depacketizer.synced();
            let view = Packet {
                sequence_number: packet.header.sequence_number,
                timestamp: packet.header.timestamp,
                marker: packet.header.marker,
                payload: &packet.payload,
            };
            let unit = depacketizer.push(&view);
            if was_synced && !depacketizer.synced() {
                self.want_keyframe();
            }
            let Some(unit) = unit else { continue };
            let unit_first_arrival = first_arrival
                .iter()
                .find(|&&(rtp, _)| rtp == unit.rtp_ts)
                .map_or(arrived_ns, |&(_, at)| at);
            capture.submitted(unit.rtp_ts, unit_first_arrival);
            let decoded = match decoder.decode(&unit.data) {
                Ok(Some(yuv)) => yuv,
                Ok(None) => continue,
                Err(err) => {
                    warn!("H.264 decode failed ({err}); waiting for a keyframe");
                    depacketizer = Depacketizer::default();
                    capture = CaptureClock::new(self.shared.config.timing.capture_max_waiting);
                    self.want_keyframe();
                    continue;
                }
            };
            // the Go2 sends no capture time for a frame: its arrival, de-jittered by the RTP tick spacing
            let ns = capture.decoded().unwrap_or(arrived_ns);
            stats.frame(ns, now_ns());
            use openh264::formats::YUVSource;
            let (width, height) = decoded.dimensions();
            rgb.resize(width * height * 3, 0);
            decoded.write_rgb8(&mut rgb);
            let image = Image {
                header: header(ns, &frame_id),
                height: height as i32,
                width: width as i32,
                encoding: "rgb8".into(),
                is_bigendian: 0,
                step: (width * 3) as i32,
                data: rgb.clone(),
            };
            let _ = self
                .shared
                .handle
                .block_on(self.color_image.publish(&image));
            if ns.saturating_sub(last_info_ns) >= 1_000_000_000 {
                last_info_ns = ns;
                let mut info = self.shared.config.camera_info.clone();
                info.header = header(ns, &frame_id);
                let _ = self.shared.handle.block_on(self.camera_info.publish(&info));
            }
        }
    }
}

/// Every 10 s, how each stream's stamps sit against the host clock: the offset in use and the stamp -> arrival
/// (and -> published, for video) spread. The numbers that say whether the Go2's clocks agree with each other.
#[derive(Default)]
struct Stats {
    since_ns: u64,
    pose: Vec<i64>,
    frame: Vec<i64>,
}

impl Stats {
    fn gap(robot_ns: Option<i64>, arrived_ns: u64, clock: &HostClock) -> Option<i64> {
        Some(arrived_ns as i64 - clock.host_ns(robot_ns?)? as i64)
    }

    fn pose(&mut self, robot_ns: Option<i64>, arrived_ns: u64, clock: &HostClock) {
        self.pose.extend(Self::gap(robot_ns, arrived_ns, clock));
        self.report(arrived_ns, clock.offset_ns(), "pose");
    }

    fn frame(&mut self, stamp_ns: u64, published_ns: u64) {
        self.frame.push(published_ns as i64 - stamp_ns as i64);
        self.report(stamp_ns, None, "video");
    }

    fn report(&mut self, now: u64, offset_ns: Option<i64>, stream: &str) {
        if self.since_ns == 0 {
            self.since_ns = now;
        }
        if now - self.since_ns < 10_000_000_000 {
            return;
        }
        let ms = |v: &mut Vec<i64>| -> String {
            if v.is_empty() {
                return "-".into();
            }
            v.sort_unstable();
            let pick = |q: f64| v[((v.len() - 1) as f64 * q) as usize] as f64 / 1e6;
            format!(
                "p50 {:.1} p95 {:.1} max {:.1} ms (n {})",
                pick(0.5),
                pick(0.95),
                pick(1.0),
                v.len()
            )
        };
        info!(
            stream,
            offset_ms = offset_ns.map(|o| o as f64 / 1e6),
            pose_capture_to_arrival = ms(&mut self.pose),
            video_stamp_to_published = ms(&mut self.frame),
            "[go2 clock]"
        );
        self.pose.clear();
        self.frame.clear();
        self.since_ns = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_stays_out_of_debug_output() {
        assert_eq!(
            format!("{:?}", Secret("00112233445566778899aabbccddeeff".into())),
            "<redacted>"
        );
        assert_eq!(format!("{:?}", Secret::default()), "\"\"");
    }

    #[test]
    fn stamps_come_from_the_robots_fields() {
        let pose = json!({ "header": { "stamp": { "sec": 1746565669, "nanosec": 448350564 } } });
        assert_eq!(pose_robot_ns(&pose), Some(1_746_565_669_448_350_564));
    }

    #[test]
    fn a_cloud_packs_xyz_float32() {
        let cloud = cloud(1_500_000_000, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(
            (
                cloud.width,
                cloud.point_step,
                cloud.row_step,
                cloud.data.len()
            ),
            (2, 12, 24, 24)
        );
        assert_eq!(&cloud.data[12..16], &4.0f32.to_le_bytes());
        assert_eq!(
            (cloud.header.stamp.sec, cloud.header.stamp.nsec),
            (1, 500_000_000)
        );
    }

    #[test]
    fn frames_are_namespaced_only_with_a_prefix() {
        assert_eq!(prefixed("", "base_link"), "base_link");
        assert_eq!(prefixed("go2_a", "base_link"), "go2_a/base_link");
    }
}
