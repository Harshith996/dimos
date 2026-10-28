// Brick map bindings and helpers shared by every map kernel.

struct MapParams {
    origin: vec4<f32>,
    voxel_size: f32,
    inv_voxel: f32,
    shadow_depth: f32,
    grace_depth: f32,
    graze_cos: f32,
    min_health: i32,
    max_health: i32,
    fine_divisor: u32,
    point_count: u32,
    ray_step: u32,
    frame: u32,
    mode: u32,
    table_mask: u32,
    brick_capacity: u32,
    list_capacity: u32,
    trace_pass: u32,
}

const MODE_TRACE: u32 = 0u;
const MODE_SEED: u32 = 1u;
const MODE_INGEST: u32 = 2u;

const EMPTY_KEY: u32 = 0xffffffffu;
const NO_SLOT: u32 = 0xffffffffu;
const BRICK_BITS: i32 = 3;
const BRICK_VOXELS: u32 = 512u;

const C_BRICKS: u32 = 0u;
const C_NEW: u32 = 1u;
const C_HITS: u32 = 2u;
const C_FITS: u32 = 3u;
const C_MISSES: u32 = 4u;
const C_CHANGED: u32 = 5u;
const C_OVERFLOW: u32 = 6u;

const L_NEW: u32 = 0u;
const L_HITS: u32 = 1u;
const L_FITS: u32 = 2u;
const L_MISSES: u32 = 3u;
const L_CHANGED: u32 = 4u;

const S_EXISTS: u32 = 1u;
const S_STALE: u32 = 2u;
const S_CREATED: u32 = 4u;

const STAMP_HIT: u32 = 0u;
const STAMP_FIT: u32 = 1u;
const STAMP_MISS: u32 = 2u;

// Count, then sum (3) and second moment (xx, xy, xz, yy, yz, zz) as 64-bit fixed point.
const MOMENT_WORDS: u32 = 19u;
const FIXED_SCALE: f32 = 16777216.0;

const NORMAL_MIN_POINTS: u32 = 3u;

@group(0) @binding(0) var<uniform> params: MapParams;
@group(0) @binding(1) var<storage, read_write> table_keys: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> table_slots: array<u32>;
@group(0) @binding(3) var<storage, read_write> brick_coords: array<vec4<i32>>;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> lists: array<u32>;
@group(0) @binding(6) var<storage, read_write> points: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> voxel_state: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> health: array<i32>;
@group(0) @binding(9) var<storage, read_write> stamps: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> fine: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> moments: array<atomic<u32>>;
@group(0) @binding(12) var<storage, read_write> next_fit: array<u32>;
@group(0) @binding(13) var<storage, read_write> normals: array<vec4<f32>>;

fn thread_index(wg: vec3<u32>, nwg: vec3<u32>, local: u32) -> u32 {
    return (wg.y * nwg.x + wg.x) * 64u + local;
}

fn voxel_of(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p * params.inv_voxel));
}

fn brick_of(key: vec3<i32>) -> vec3<i32> {
    return key >> vec3<u32>(3u);
}

fn local_index(key: vec3<i32>) -> u32 {
    let l = vec3<u32>(key & vec3<i32>(7));
    return (l.x << 6u) | (l.y << 3u) | l.z;
}

// 11 bits x, 11 bits y, 10 bits z, biased; the all-ones key is reserved as empty.
fn pack_brick(b: vec3<i32>) -> u32 {
    let u = vec3<u32>(b + vec3<i32>(1024, 1024, 512));
    return (u.x & 2047u) | ((u.y & 2047u) << 11u) | ((u.z & 1023u) << 22u);
}

fn brick_in_range(b: vec3<i32>) -> bool {
    let inside = all(b >= vec3<i32>(-1024, -1024, -512)) && all(b < vec3<i32>(1023, 1023, 511));
    return inside;
}

fn hash_key(key: u32) -> u32 {
    var h = key;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return h;
}

// The brick's pool slot, NO_SLOT when absent.
fn find_brick(b: vec3<i32>) -> u32 {
    if (!brick_in_range(b)) {
        return NO_SLOT;
    }
    let key = pack_brick(b);
    var i = hash_key(key) & params.table_mask;
    for (var probe = 0u; probe <= params.table_mask; probe++) {
        let k = atomicLoad(&table_keys[i]);
        if (k == key) {
            return table_slots[i];
        }
        if (k == EMPTY_KEY) {
            return NO_SLOT;
        }
        i = (i + 1u) & params.table_mask;
    }
    return NO_SLOT;
}

// Global voxel index of a key, NO_SLOT when its brick is absent.
fn find_voxel(key: vec3<i32>) -> u32 {
    let slot = find_brick(brick_of(key));
    if (slot == NO_SLOT) {
        return NO_SLOT;
    }
    return slot * BRICK_VOXELS + local_index(key);
}

fn voxel_exists(v: u32) -> bool {
    return (atomicLoad(&voxel_state[v]) & S_EXISTS) != 0u;
}

fn push(list: u32, counter: u32, value: u32) {
    let i = atomicAdd(&counters[counter], 1u);
    if (i < params.list_capacity) {
        lists[list * params.list_capacity + i] = value;
    } else {
        atomicOr(&counters[C_OVERFLOW], 1u << list);
    }
}

fn list_len(counter: u32) -> u32 {
    return min(atomicLoad(&counters[counter]), params.list_capacity);
}

fn voxel_center(key: vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(key) + vec3<f32>(0.5)) * params.voxel_size;
}

// The key of global voxel index v from its brick's recorded coordinate.
fn key_of_voxel(v: u32) -> vec3<i32> {
    let b = brick_coords[v / BRICK_VOXELS].xyz;
    let l = v % BRICK_VOXELS;
    return (b << vec3<u32>(3u)) | vec3<i32>(i32(l >> 6u), i32((l >> 3u) & 7u), i32(l & 7u));
}

// Read a 64-bit fixed-point moment word pair as f32 in voxel units.
fn moment_value(v: u32, word: u32) -> f32 {
    let base = v * MOMENT_WORDS + 1u + word * 2u;
    var lo = atomicLoad(&moments[base]);
    var hi = atomicLoad(&moments[base + 1u]);
    let negative = (hi & 0x80000000u) != 0u;
    if (negative) {
        lo = ~lo + 1u;
        hi = ~hi + select(0u, 1u, lo == 0u);
    }
    let magnitude = (f32(hi) * 4294967296.0 + f32(lo)) / FIXED_SCALE;
    return select(magnitude, -magnitude, negative);
}

fn moment_count(v: u32) -> u32 {
    return atomicLoad(&moments[v * MOMENT_WORDS]);
}
