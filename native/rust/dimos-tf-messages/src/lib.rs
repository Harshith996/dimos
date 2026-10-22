pub const ROS2MSG_SCHEMAS: &str = include_str!("../schemas.json");
include!(concat!(env!("OUT_DIR"), "/mod.rs"));
