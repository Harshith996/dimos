// Surface extraction over a dense window of columns: occupancy from the
// brick map, the standable test, bit-parallel round hole closing per z slice
// (32 slices per word op), and
// jump-flood wall distance. Appended to map_common.wgsl.
//
// Column bit layout: word (y * W + x) * ZW + z / 32, bit z % 32.
// Cell layout: (z * H + y) * W + x.
//
// Wall distance runs per plane: a connected surface (4-neighbor links with a
// step of at most step_cells, as the CPU adjacency) projected from above.
// Where a connected surface overlaps itself in x,y (a ramp under a balcony),
// it splits into sheets: within a column, a cell more than a step above the
// surface's cell below it starts the next sheet. Each (surface, sheet) piece
// connected under equal sheets is a plane; cells of one plane sharing a
// column (a hole filled at two adjacent z) share that column's distance.

struct SurfaceParams {
    window_min: vec4<i32>,
    // W, H, D, ZW
    size: vec4<u32>,
    clearance_cells: i32,
    closing_cells: i32,
    step_cells: i32,
    hole_span_cells: i32,
    support_radius: i32,
    support_z_tol: i32,
    voxel_size: f32,
    _pad: i32,
}

struct FloodStep {
    step: i32,
    // 1 reads seeds_a and writes seeds_b, 0 the other way.
    a_to_b: u32,
}

const NO_SEED: u32 = 0xffffffffu;

@group(1) @binding(0) var<uniform> sp: SurfaceParams;
@group(1) @binding(1) var<storage, read_write> occupied: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> standable: array<u32>;
@group(1) @binding(3) var<storage, read_write> grown: array<u32>;
@group(1) @binding(4) var<storage, read_write> surface: array<u32>;
@group(1) @binding(5) var<storage, read_write> wall: array<u32>;
@group(1) @binding(6) var<storage, read_write> seeds_a: array<u32>;
@group(1) @binding(7) var<storage, read_write> seeds_b: array<u32>;
@group(1) @binding(8) var<storage, read_write> wall_distance: array<f32>;
// Two union-find parent arrays over cells: components, then planes.
@group(1) @binding(9) var<storage, read_write> labels: array<atomic<u32>>;
@group(2) @binding(0) var<uniform> flood: FloodStep;

fn column_word(x: i32, y: i32, z: i32) -> u32 {
    return (u32(y) * sp.size.x + u32(x)) * sp.size.w + u32(z) / 32u;
}

fn in_window_xy(x: i32, y: i32) -> bool {
    return x >= 0 && y >= 0 && x < i32(sp.size.x) && y < i32(sp.size.y);
}

fn occupied_at(x: i32, y: i32, z: i32) -> bool {
    if (!in_window_xy(x, y) || z < 0 || z >= i32(sp.size.z)) {
        return false;
    }
    return ((atomicLoad(&occupied[column_word(x, y, z)]) >> (u32(z) & 31u)) & 1u) != 0u;
}

fn bit_at(x: i32, y: i32, z: i32, which: u32) -> bool {
    if (!in_window_xy(x, y) || z < 0 || z >= i32(sp.size.z)) {
        return false;
    }
    let w = column_word(x, y, z);
    var word = 0u;
    switch which {
        case 0u: { word = standable[w]; }
        case 1u: { word = surface[w]; }
        case 2u: { word = grown[w]; }
        default: { word = wall[w]; }
    }
    return ((word >> (u32(z) & 31u)) & 1u) != 0u;
}

fn column_has_occupied(x: i32, y: i32) -> bool {
    if (!in_window_xy(x, y)) {
        return false;
    }
    let base = column_word(x, y, 0);
    for (var w = 0u; w < sp.size.w; w++) {
        if (atomicLoad(&occupied[base + w]) != 0u) {
            return true;
        }
    }
    return false;
}

fn cell_index(x: i32, y: i32, z: i32) -> u32 {
    return (u32(z) * sp.size.y + u32(y)) * sp.size.x + u32(x);
}

fn pack_seed(x: i32, y: i32) -> u32 {
    return u32(x) | (u32(y) << 16u);
}

fn seed_dist_sq(x: i32, y: i32, seed: u32) -> i32 {
    if (seed == NO_SEED) {
        return 0x7fffffff;
    }
    let dx = i32(seed & 0xffffu) - x;
    let dy = i32(seed >> 16u) - y;
    return dx * dx + dy * dy;
}

// Healthy voxels inside the window set their column bit.
@compute @workgroup_size(64)
fn build_occupancy(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let v = thread_index(wg, nwg, li);
    if (v >= min(atomicLoad(&counters[C_BRICKS]), params.brick_capacity) * BRICK_VOXELS) {
        return;
    }
    if (!voxel_exists(v) || health[v] <= 0) {
        return;
    }
    let local = key_of_voxel(v) - sp.window_min.xyz;
    if (any(local < vec3<i32>(0)) || any(vec3<u32>(local) >= sp.size.xyz)) {
        return;
    }
    atomicOr(&occupied[column_word(local.x, local.y, local.z)], 1u << (u32(local.z) & 31u));
}

// Per column: occupied cells whose next occupied cell above is more than the
// clearance away, plus the column top.
@compute @workgroup_size(8, 8, 1)
fn standable_columns(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let base = column_word(x, y, 0);
    for (var w = 0u; w < sp.size.w; w++) {
        standable[base + w] = 0u;
    }
    var next_above = 0x3fffffff;
    for (var z = i32(sp.size.z) - 1; z >= 0; z--) {
        if (!occupied_at(x, y, z)) {
            continue;
        }
        if (next_above - z > sp.clearance_cells) {
            standable[base + u32(z) / 32u] |= 1u << (u32(z) & 31u);
        }
        next_above = z;
    }
}

// The bits of column (x, y) at z + shift, aligned to word `word`'s z range:
// bit b holds z = 32 * word + b + shift. Outside the window reads empty.
fn shifted_word(which: u32, x: i32, y: i32, word: i32, shift: i32) -> u32 {
    if (!in_window_xy(x, y)) {
        return 0u;
    }
    let base = column_word(x, y, 0);
    let words = i32(sp.size.w);
    let start = word * 32 + shift;
    let first = start >> 5u;
    let offset = u32(start & 31);
    var out = 0u;
    if (first >= 0 && first < words) {
        out = column_bits(which, base + u32(first)) >> offset;
    }
    if (offset != 0u && first + 1 >= 0 && first + 1 < words) {
        out |= column_bits(which, base + u32(first + 1)) << (32u - offset);
    }
    return out;
}

fn column_bits(which: u32, w: u32) -> u32 {
    switch which {
        case 0u: { return standable[w]; }
        case 2u: { return grown[w]; }
        default: { return atomicLoad(&occupied[w]); }
    }
}

const BITS_STANDABLE: u32 = 0u;
const BITS_GROWN: u32 = 2u;
const BITS_OCCUPIED: u32 = 4u;

// Grow: a cell is grown when a standable cell of its slice lies within the
// closing radius (Euclidean, in cells). 32 slices per word op.
@compute @workgroup_size(8, 8, 1)
fn grow_columns(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let n = sp.closing_cells;
    let base = column_word(x, y, 0);
    for (var w = 0u; w < sp.size.w; w++) {
        var bits = 0u;
        for (var dy = -n; dy <= n; dy++) {
            for (var dx = -n; dx <= n; dx++) {
                if (dx * dx + dy * dy > n * n || !in_window_xy(x + dx, y + dy)) {
                    continue;
                }
                bits |= standable[column_word(x + dx, y + dy, 0) + w];
            }
        }
        grown[base + w] = bits;
    }
}

// Shrink the grown set by the same disk, outside the window unfilled, then
// keep the filled cells with headroom and nearby support, as close_at_z.
@compute @workgroup_size(8, 8, 1)
fn compose_surface(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let n = sp.closing_cells;
    let base = column_word(x, y, 0);
    for (var w = 0u; w < sp.size.w; w++) {
        let original = standable[base + w];
        var closed = 0u;
        if (n > 0) {
            closed = grown[base + w];
            for (var dy = -n; dy <= n && closed != 0u; dy++) {
                for (var dx = -n; dx <= n; dx++) {
                    if (dx * dx + dy * dy > n * n) {
                        continue;
                    }
                    if (!in_window_xy(x + dx, y + dy)) {
                        closed = 0u;
                        break;
                    }
                    closed &= grown[column_word(x + dx, y + dy, 0) + w];
                }
            }
        }
        var added = closed & ~original;
        if (added != 0u) {
            var above = 0u;
            for (var k = 1; k <= sp.clearance_cells; k++) {
                above |= shifted_word(BITS_OCCUPIED, x, y, i32(w), k);
            }
            added &= ~above;
        }
        if (added != 0u) {
            var support = 0u;
            let r = sp.support_radius;
            for (var dy = -r; dy <= r; dy++) {
                for (var dx = -r; dx <= r; dx++) {
                    for (var dz = -sp.support_z_tol; dz <= sp.support_z_tol; dz++) {
                        support |= shifted_word(BITS_OCCUPIED, x + dx, y + dy, i32(w), dz);
                    }
                }
            }
            added &= support;
        }
        surface[base + w] = original | added;
    }
}

// The neighbor column holds a surface cell within a step of z.
fn surface_linked(x: i32, y: i32, z: i32) -> bool {
    for (var dz = -sp.step_cells; dz <= sp.step_cells; dz++) {
        if (bit_at(x, y, z + dz, 1u)) {
            return true;
        }
    }
    return false;
}

// The CPU edge_in_direction: the first non-empty column within the hole span
// decides, an all-empty span is an edge.
fn edge_in_direction(x: i32, y: i32, z: i32, dx: i32, dy: i32) -> bool {
    for (var k = 1; k <= sp.hole_span_cells; k++) {
        let nx = x + dx * k;
        let ny = y + dy * k;
        if (!column_has_occupied(nx, ny)) {
            continue;
        }
        for (var dz = -sp.step_cells; dz <= sp.step_cells; dz++) {
            if (bit_at(nx, ny, z + dz, 0u)) {
                return false;
            }
        }
        return true;
    }
    return true;
}

@compute @workgroup_size(8, 8, 1)
fn find_walls(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let base = column_word(x, y, 0);
    for (var w = 0u; w < sp.size.w; w++) {
        wall[base + w] = 0u;
    }
    let dirs = array<vec2<i32>, 4>(vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1));
    for (var z = 0; z < i32(sp.size.z); z++) {
        if (!bit_at(x, y, z, 1u)) {
            continue;
        }
        var is_wall = false;
        for (var k = 0; k < 4; k++) {
            let d = dirs[k];
            if (!surface_linked(x + d.x, y + d.y, z) && edge_in_direction(x, y, z, d.x, d.y)) {
                is_wall = true;
                break;
            }
        }
        if (is_wall) {
            wall[base + u32(z) / 32u] |= 1u << (u32(z) & 31u);
        }
    }
}


const COMPONENTS: u32 = 0u;

fn planes_base() -> u32 {
    return sp.size.x * sp.size.y * sp.size.z;
}

fn is_surface(x: i32, y: i32, z: i32) -> bool {
    return bit_at(x, y, z, 1u);
}

fn uf_find(base: u32, c: u32) -> u32 {
    var x = c;
    for (var guard = 0u; guard < 0xffffffu; guard++) {
        let p = atomicLoad(&labels[base + x]);
        if (p == x) {
            break;
        }
        x = p;
    }
    return x;
}

// Hook the larger root under the smaller with atomicMin, retrying when a
// concurrent hook got there first (Playne and Hawick).
fn uf_union(base: u32, a: u32, b: u32) {
    var x = uf_find(base, a);
    var y = uf_find(base, b);
    for (var guard = 0u; guard < 0xffffffu; guard++) {
        if (x == y) {
            return;
        }
        if (x < y) {
            let t = x;
            x = y;
            y = t;
        }
        let old = atomicMin(&labels[base + x], y);
        if (old == x) {
            return;
        }
        x = uf_find(base, old);
        y = uf_find(base, y);
    }
}

@compute @workgroup_size(8, 8, 1)
fn uf_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y || gid.z >= sp.size.z) {
        return;
    }
    let c = cell_index(i32(gid.x), i32(gid.y), i32(gid.z));
    let own = select(NO_SEED, c, is_surface(i32(gid.x), i32(gid.y), i32(gid.z)));
    atomicStore(&labels[c], own);
    atomicStore(&labels[planes_base() + c], own);
}

// Which sheet of its component a cell is on, counted from below: cells of
// the component in this column within a step of the one beneath share a sheet.
fn component_rank(x: i32, y: i32, z: i32, component: u32) -> u32 {
    var rank = 0u;
    var previous = -1;
    for (var below = 0; below <= z; below++) {
        if (!is_surface(x, y, below) || atomicLoad(&labels[cell_index(x, y, below)]) != component) {
            continue;
        }
        if (previous >= 0 && below - previous > sp.step_cells) {
            rank += 1u;
        }
        previous = below;
    }
    return rank;
}

// Union each surface cell with its linked +x and +y neighbors; the plane pass
// also requires equal rank.
fn link_neighbors(gid: vec3<u32>, planes: bool) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y || gid.z >= sp.size.z) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (!is_surface(x, y, z)) {
        return;
    }
    let c = cell_index(x, y, z);
    var rank = 0u;
    var component = 0u;
    if (planes) {
        component = atomicLoad(&labels[c]);
        rank = component_rank(x, y, z, component);
    }
    let base = select(COMPONENTS, planes_base(), planes);
    let dirs = array<vec2<i32>, 2>(vec2<i32>(1, 0), vec2<i32>(0, 1));
    for (var k = 0; k < 2; k++) {
        let nx = x + dirs[k].x;
        let ny = y + dirs[k].y;
        for (var dz = -sp.step_cells; dz <= sp.step_cells; dz++) {
            let nz = z + dz;
            if (!is_surface(nx, ny, nz)) {
                continue;
            }
            if (planes && component_rank(nx, ny, nz, component) != rank) {
                continue;
            }
            uf_union(base, c, cell_index(nx, ny, nz));
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn link_components(@builtin(global_invocation_id) gid: vec3<u32>) {
    link_neighbors(gid, false);
}

@compute @workgroup_size(8, 8, 1)
fn link_planes(@builtin(global_invocation_id) gid: vec3<u32>) {
    link_neighbors(gid, true);
}

fn flatten(gid: vec3<u32>, base: u32) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y || gid.z >= sp.size.z) {
        return;
    }
    let c = cell_index(i32(gid.x), i32(gid.y), i32(gid.z));
    if (atomicLoad(&labels[base + c]) != NO_SEED) {
        atomicStore(&labels[base + c], uf_find(base, c));
    }
}

@compute @workgroup_size(8, 8, 1)
fn flatten_components(@builtin(global_invocation_id) gid: vec3<u32>) {
    flatten(gid, COMPONENTS);
}

@compute @workgroup_size(8, 8, 1)
fn flatten_planes(@builtin(global_invocation_id) gid: vec3<u32>) {
    flatten(gid, planes_base());
}

struct PlaneInfo {
    planes: u32,
    cells: u32,
    capacity: u32,
    _pad: u32,
}

@group(3) @binding(0) var<uniform> plane_info: PlaneInfo;
@group(3) @binding(1) var<storage, read_write> plane_count: array<atomic<u32>>;
// Per plane: min x, min y, max x, max y, in window columns.
@group(3) @binding(2) var<storage, read_write> plane_bounds: array<atomic<i32>>;
// Per plane: first plane cell, min x, min y, width.
@group(3) @binding(3) var<storage, read_write> plane_table: array<vec4<u32>>;
@group(3) @binding(4) var<storage, read_write> plane_cells_a: array<u32>;
@group(3) @binding(5) var<storage, read_write> plane_cells_b: array<u32>;
@group(3) @binding(6) var<storage, read_write> plane_owner: array<u32>;

fn plane_of_cell(c: u32) -> u32 {
    return seeds_a[atomicLoad(&labels[planes_base() + c])];
}

// Number each plane root; the ids index plane_bounds and plane_table.
@compute @workgroup_size(8, 8, 1)
fn number_planes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y || gid.z >= sp.size.z) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = cell_index(x, y, i32(gid.z));
    if (atomicLoad(&labels[planes_base() + c]) != c) {
        return;
    }
    let id = atomicAdd(&plane_count[0], 1u);
    if (id >= plane_info.capacity) {
        seeds_a[c] = NO_SEED;
        return;
    }
    seeds_a[c] = id;
    atomicStore(&plane_bounds[id * 4u], x);
    atomicStore(&plane_bounds[id * 4u + 1u], y);
    atomicStore(&plane_bounds[id * 4u + 2u], x);
    atomicStore(&plane_bounds[id * 4u + 3u], y);
}

@compute @workgroup_size(8, 8, 1)
fn bound_planes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y || gid.z >= sp.size.z) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (!is_surface(x, y, z)) {
        return;
    }
    let id = plane_of_cell(cell_index(x, y, z));
    if (id == NO_SEED) {
        return;
    }
    atomicMin(&plane_bounds[id * 4u], x);
    atomicMin(&plane_bounds[id * 4u + 1u], y);
    atomicMax(&plane_bounds[id * 4u + 2u], x);
    atomicMax(&plane_bounds[id * 4u + 3u], y);
}

// The plane whose cell range holds plane cell t; the table is sorted by first cell.
fn owner_of(t: u32) -> u32 {
    var lo = 0u;
    var hi = plane_info.planes;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (plane_table[mid].x <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn plane_height(id: u32) -> u32 {
    let next = select(plane_info.cells, plane_table[id + 1u].x, id + 1u < plane_info.planes);
    return (next - plane_table[id].x) / plane_table[id].w;
}

@compute @workgroup_size(64)
fn init_plane_cells(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let t = thread_index(wg, nwg, li);
    if (t >= plane_info.cells) {
        return;
    }
    plane_owner[t] = owner_of(t);
    plane_cells_a[t] = NO_SEED;
}

fn plane_cell(id: u32, x: i32, y: i32) -> u32 {
    let e = plane_table[id];
    return e.x + (u32(y) - e.z) * e.w + (u32(x) - e.y);
}

@compute @workgroup_size(8, 8, 1)
fn scatter_walls(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y || gid.z >= sp.size.z) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    if (!bit_at(x, y, z, 3u)) {
        return;
    }
    let id = plane_of_cell(cell_index(x, y, z));
    if (id != NO_SEED) {
        plane_cells_a[plane_cell(id, x, y)] = pack_seed(x, y);
    }
}

fn plane_read(t: u32) -> u32 {
    if (flood.a_to_b != 0u) {
        return plane_cells_a[t];
    }
    return plane_cells_b[t];
}

@compute @workgroup_size(64)
fn plane_flood_pass(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let t = thread_index(wg, nwg, li);
    if (t >= plane_info.cells) {
        return;
    }
    let id = plane_owner[t];
    let e = plane_table[id];
    let local = t - e.x;
    let lx = i32(local % e.w);
    let ly = i32(local / e.w);
    let width = i32(e.w);
    let height = i32(plane_height(id));
    let x = lx + i32(e.y);
    let y = ly + i32(e.z);
    var best = plane_read(t);
    var best_d = seed_dist_sq(x, y, best);
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            let qx = lx + ox * flood.step;
            let qy = ly + oy * flood.step;
            if ((ox == 0 && oy == 0) || qx < 0 || qy < 0 || qx >= width || qy >= height) {
                continue;
            }
            let seed = plane_read(e.x + u32(qy * width + qx));
            let d = seed_dist_sq(x, y, seed);
            if (d < best_d) {
                best = seed;
                best_d = d;
            }
        }
    }
    if (flood.a_to_b != 0u) {
        plane_cells_b[t] = best;
    } else {
        plane_cells_a[t] = best;
    }
}

// Straight-line distance in meters from each surface cell to the nearest
// wall cell of its plane, 0 on wall cells, -1 off the surface or on a plane
// with no wall.
@compute @workgroup_size(8, 8, 1)
fn measure_walls(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y || gid.z >= sp.size.z) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    let c = cell_index(x, y, z);
    var d = -1.0;
    if (bit_at(x, y, z, 3u)) {
        d = 0.0;
    } else if (is_surface(x, y, z)) {
        let id = plane_of_cell(c);
        if (id != NO_SEED) {
            let seed = plane_read(plane_cell(id, x, y));
            // A cell stacked over a wall cell of its sheet reads its own
            // column; any non-wall cell is at least one cell from a wall.
            if (seed != NO_SEED) {
                d = max(sqrt(f32(seed_dist_sq(x, y, seed))), 1.0) * sp.voxel_size;
            }
        }
    }
    wall_distance[c] = d;
}

// The plane id of every surface cell, for inspecting the split.
@compute @workgroup_size(8, 8, 1)
fn export_planes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sp.size.x || gid.y >= sp.size.y || gid.z >= sp.size.z) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let z = i32(gid.z);
    let c = cell_index(x, y, z);
    var id = NO_SEED;
    if (is_surface(x, y, z)) {
        id = plane_of_cell(c);
    }
    seeds_b[c] = id;
}
