// Global error handler
window.onerror = (msg, url, line, col, error) => {
    console.error(`[ERROR] ${msg} at ${url}:${line}:${col}`, error);
    document.getElementById('status').textContent = `Error: ${msg}`;
};

import { geometry_msgs, std_msgs, sensor_msgs } from "https://esm.sh/jsr/@dimos/msgs@0.1.4";
import { captureBody } from "./webxr_body.mjs";

// WebSocket and WebXR state
let ws = null;
let xrSession = null;
let xrRefSpace = null;
let xrBodyRefSpace = null;
let xrBodyRefSpaceType = null;
let gl = null;
let lastSendTime = 0;
const sendInterval = 1000 / 80; // ~80Hz target
let webXRClientConfig = null;
const sessionModeSupport = new Map();
const handSelectActive = new Map();
const GRIPPER_PINCH_DISTANCE_METERS = 0.04;

// Camera panels. /teleop/config lists them (camera_panels) in wire order; a
// binary /ws message is one byte of panel index followed by a JPEG. The panels
// are world-locked: placed once per session relative to the operator's level
// starting pose (position and heading only, so looking down at the arms while
// connecting does not drop them to the floor).
let quadProgram = null;
let quadVbo = null;
let quadAttribs = null;
let quadUniforms = null;
let solidProgram = null;
let solidAttribs = null;
let solidUniforms = null;
let panels = [];
let panelsPlaced = false;
const panelAnchorMatrix = new Float32Array(16);
const gripHeld = { left: false, right: false };

// Offsets from the level anchor in meters (x right, y up, z forward is
// negative). yawDeg turns a side panel in toward the operator; tiltDeg leans a
// raised panel's face down toward them. height is the image height.
const PANEL_LAYOUT = {
    // Egocentric view: large, centered above the workspace and within easy view.
    center: { x: 0.0, y: 0.38, z: -1.2, yawDeg: 0, tiltDeg: 10, height: 0.68 },
    // Wrist views: slightly smaller, close beside their arm.
    left: { x: -0.82, y: 0.24, z: -1.1, yawDeg: 34, tiltDeg: 6, height: 0.46 },
    right: { x: 0.82, y: 0.24, z: -1.1, yawDeg: -34, tiltDeg: 6, height: 0.46 },
};
const PANEL_BORDER_METERS = 0.012;
const PANEL_DEFAULT_ASPECT = 16 / 9;
const PANEL_STALE_MS = 2000;
// Border colors (RGBA 0-1). Wrist panels follow their controller's grip, the
// center panel follows episode recording, and any panel without fresh frames
// goes gray.
const BORDER_COLORS = {
    idle: [0.78, 0.82, 0.86, 0.9],
    engaged: [0.22, 0.83, 0.33, 1.0],
    live: [0.96, 0.65, 0.14, 1.0],
    recording: [1.0, 0.23, 0.19, 1.0],
    stale: [0.45, 0.47, 0.5, 0.8],
};
const PANEL_BLANK_COLOR = [0.03, 0.05, 0.08, 0.75];

// Collection HUD: a compact card below the center panel with a dotted status
// glyph, a large state line (READY / REC 00:07), the operator hint from the
// server config, the task label, take counters, and dataset export progress.
// It is placed with the camera panels and redrawn only when its text changes.
let episodeStatus = null;
let episodeStatusReceivedAtMs = 0;
let exportStatus = null;
let hudOffline = false;
let hudCanvas = null;
let hudContext = null;
let hudTexture = null;
let hudDirty = false;
let hudDrawnKey = null;
const hudLocalMatrix = new Float32Array(16);
const hudModelMatrix = new Float32Array(16);

const HUD_WIDTH_PX = 1024;
const HUD_HEIGHT_PX = 384;
const HUD_WIDTH_METERS = 0.62;
const HUD_HEIGHT_METERS = HUD_WIDTH_METERS * HUD_HEIGHT_PX / HUD_WIDTH_PX;
// Below the center panel, leaning up toward the operator.
const HUD_LAYOUT = { x: 0.0, y: -0.2, z: -1.05, yawDeg: 0, tiltDeg: -8 };
const HUD_FONT = 'ui-monospace, SFMono-Regular, Menlo, monospace';
const HUD_COLORS = {
    text: '#f4f7fb',
    dim: '#9fb0c3',
    ready: '#f5a623',
    recording: '#ff3b30',
    saved: '#39d353',
    failed: '#ff3b30',
    offline: '#8a939e',
};
// How long a SAVED / DISCARDED outcome stays on the state line before READY.
const HUD_OUTCOME_MS = 4000;

// UI elements
const statusEl = document.getElementById('status');
const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const canvas = document.getElementById('canvas');
const datasetLabelForm = document.getElementById('datasetLabelForm');
const datasetLabelInput = document.getElementById('datasetLabel');
const datasetLabelBtn = document.getElementById('datasetLabelBtn');
const datasetLabelStatus = document.getElementById('datasetLabelStatus');
let selectedDatasetSuffix = null;

function setStatus(msg) {
    statusEl.textContent = msg;
}

function setDatasetLabelStatus(message, isError = false) {
    datasetLabelStatus.textContent = message;
    datasetLabelStatus.classList.toggle('error', isError);
}

datasetLabelForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const label = datasetLabelInput.value.trim();
    if (!label) {
        setDatasetLabelStatus('Enter a collection label first.', true);
        datasetLabelInput.focus();
        return;
    }

    datasetLabelInput.disabled = true;
    datasetLabelBtn.disabled = true;
    setDatasetLabelStatus('Saving label...');
    try {
        const response = await fetch('/teleop/dataset-label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label }),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
        selectedDatasetSuffix = result.suffix;
        datasetLabelInput.value = result.suffix;
        setDatasetLabelStatus(`Dataset folder will end in _${result.suffix}`);
    } catch (error) {
        setDatasetLabelStatus(error?.message || String(error), true);
    } finally {
        datasetLabelInput.disabled = false;
        datasetLabelBtn.disabled = false;
    }
});

async function loadWebXRClientConfig() {
    const response = await fetch('/teleop/config', { cache: 'no-store' });
    if (!response.ok) {
        throw new Error(`Failed to load teleop configuration: HTTP ${response.status}`);
    }
    return response.json();
}

function describeSessionRequestError(mode, error) {
    const name = error?.name || 'Error';
    const message = error?.message || String(error);
    return `${mode} (${name}: ${message})`;
}

// WebSocket setup (LCM bridge)
function setupWebSocket() {
    return new Promise((resolve, reject) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        setStatus('Connecting to server...');
        ws = new WebSocket(wsUrl);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            hudOffline = false;
            hudDirty = true;
            setStatus('Server connected');
            resolve();
        };
        ws.onerror = (error) => {
            setStatus('WebSocket error');
            console.error('WebSocket error:', error);
            reject(error);
        };
        ws.onclose = () => {
            hudOffline = true;
            hudDirty = true;
            if (xrSession) setStatus('WebSocket closed');
        };
        ws.onmessage = (e) => {
            if (typeof e.data === 'string') {
                handleServerMessage(e.data);
                return;
            }
            if (!(e.data instanceof ArrayBuffer) || e.data.byteLength < 2) return;
            const bytes = new Uint8Array(e.data);
            receivePanelFrame(bytes[0], bytes.subarray(1));
        };
    });
}

// Initialize WebGL
function initGL() {
    gl = canvas.getContext('webgl', {
        xrCompatible: true,
        alpha: true
    });
    if (!gl) {
        throw new Error('WebGL not supported');
    }
    gl.clearColor(0, 0, 0, 0); // Transparent background for passthrough
    initQuadPipelines();
    initCameraPanels(webXRClientConfig.camera_panels ?? []);
    initCollectionHud();
}

function compileShader(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        throw new Error('Shader compile failed: ' + gl.getShaderInfoLog(sh));
    }
    return sh;
}

function linkProgram(vsSrc, fsSrc) {
    const program = gl.createProgram();
    gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error('Program link failed: ' + gl.getProgramInfoLog(program));
    }
    return program;
}

// One textured-quad pipeline for the camera panels and HUD, and one
// solid-color pipeline for panel borders and blank panels. Both share a quad.
function initQuadPipelines() {
    const quadVsSrc = `
        attribute vec2 a_pos;
        attribute vec2 a_uv;
        uniform mat4 u_proj;
        uniform mat4 u_view;
        uniform mat4 u_model;
        varying vec2 v_uv;
        void main() {
            gl_Position = u_proj * u_view * u_model
                        * vec4(a_pos.x, a_pos.y, 0.0, 1.0);
            v_uv = a_uv;
        }`;
    const texturedFsSrc = `
        precision mediump float;
        varying vec2 v_uv;
        uniform sampler2D u_tex;
        void main() {
            gl_FragColor = texture2D(u_tex, v_uv);
        }`;
    const solidFsSrc = `
        precision mediump float;
        uniform vec4 u_color;
        void main() {
            gl_FragColor = u_color;
        }`;

    quadProgram = linkProgram(quadVsSrc, texturedFsSrc);
    quadAttribs = {
        pos: gl.getAttribLocation(quadProgram, 'a_pos'),
        uv:  gl.getAttribLocation(quadProgram, 'a_uv'),
    };
    quadUniforms = {
        proj:  gl.getUniformLocation(quadProgram, 'u_proj'),
        view:  gl.getUniformLocation(quadProgram, 'u_view'),
        model: gl.getUniformLocation(quadProgram, 'u_model'),
        tex:   gl.getUniformLocation(quadProgram, 'u_tex'),
    };

    solidProgram = linkProgram(quadVsSrc, solidFsSrc);
    solidAttribs = {
        pos: gl.getAttribLocation(solidProgram, 'a_pos'),
    };
    solidUniforms = {
        proj:  gl.getUniformLocation(solidProgram, 'u_proj'),
        view:  gl.getUniformLocation(solidProgram, 'u_view'),
        model: gl.getUniformLocation(solidProgram, 'u_model'),
        color: gl.getUniformLocation(solidProgram, 'u_color'),
    };

    // Quad as TRIANGLE_STRIP: x, y, u, v. v=0 at top of quad +
    // UNPACK_FLIP_Y_WEBGL=false → image displays upright.
    const verts = new Float32Array([
        -1, -1, 0, 1,
         1, -1, 1, 1,
        -1,  1, 0, 0,
         1,  1, 1, 0,
    ]);
    quadVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
}

function createTexture() {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return texture;
}

function initCameraPanels(configuredPanels) {
    for (const panel of panels) {
        gl.deleteTexture(panel.texture);
        panel.bitmap?.close();
    }
    panels = [];
    panelsPlaced = false;
    configuredPanels.forEach((configured, index) => {
        const layout = PANEL_LAYOUT[configured.slot];
        if (!layout) {
            console.warn(`Ignoring camera panel ${configured.name}: unknown slot ${configured.slot}`);
            return;
        }
        panels[index] = {
            index,
            name: configured.name,
            slot: configured.slot,
            layout,
            texture: createTexture(),
            aspect: PANEL_DEFAULT_ASPECT,
            hasFrame: false,
            lastFrameMs: 0,
            decoding: false,
            pendingBlob: null,  // newest undecoded frame; older ones are dropped
            bitmap: null,       // decoded frame awaiting texture upload
            localMatrix: new Float32Array(16),
            modelMatrix: new Float32Array(16),
            borderLocalMatrix: new Float32Array(16),
            borderModelMatrix: new Float32Array(16),
        };
    });
}

function receivePanelFrame(panelIndex, jpegBytes) {
    const panel = panels[panelIndex];
    if (!panel) return;
    panel.pendingBlob = new Blob([jpegBytes], { type: 'image/jpeg' });
    if (!panel.decoding) decodePendingFrames(panel);
}

// Decode off the render loop, always skipping to the newest frame so a slow
// decode never builds a backlog of stale video.
async function decodePendingFrames(panel) {
    panel.decoding = true;
    while (panel.pendingBlob) {
        const blob = panel.pendingBlob;
        panel.pendingBlob = null;
        try {
            const bitmap = await createImageBitmap(blob);
            panel.bitmap?.close();
            panel.bitmap = bitmap;
            panel.lastFrameMs = performance.now();
        } catch (error) {
            console.warn(`Dropping undecodable ${panel.name} frame`, error);
        }
    }
    panel.decoding = false;
}

function uploadPanelFrames() {
    for (const panel of panels) {
        const bitmap = panel.bitmap;
        if (!bitmap) continue;
        panel.bitmap = null;
        gl.bindTexture(gl.TEXTURE_2D, panel.texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, bitmap);
        if (bitmap.height) panel.aspect = bitmap.width / bitmap.height;
        bitmap.close();
        panel.hasFrame = true;
        if (panelsPlaced) updatePanelMatrices(panel);
    }
}

// Column-major T(x, y, z) * Ry(yaw) * Rx(tilt) * S(halfWidth, halfHeight, 1).
// Positive yaw turns the quad's face toward +x; positive tilt leans it down.
function setPanelMatrix(matrix, halfWidth, halfHeight, x, y, z, yawDeg, tiltDeg) {
    const yaw = yawDeg * Math.PI / 180;
    const tilt = tiltDeg * Math.PI / 180;
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    matrix.fill(0);
    matrix[0] = cy * halfWidth;
    matrix[1] = 0;
    matrix[2] = -sy * halfWidth;
    matrix[4] = sy * st * halfHeight;
    matrix[5] = ct * halfHeight;
    matrix[6] = cy * st * halfHeight;
    matrix[8] = sy * ct;
    matrix[9] = -st;
    matrix[10] = cy * ct;
    matrix[12] = x;
    matrix[13] = y;
    matrix[14] = z;
    matrix[15] = 1;
}

// Column-major T(x, y, z) * Ry(yaw): a level frame with the operator's heading.
function setLevelAnchorMatrix(matrix, yaw, x, y, z) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    matrix.fill(0);
    matrix[0] = c;
    matrix[2] = -s;
    matrix[5] = 1;
    matrix[8] = s;
    matrix[10] = c;
    matrix[12] = x;
    matrix[13] = y;
    matrix[14] = z;
    matrix[15] = 1;
}

function multiplyMatrices(out, left, right) {
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            let value = 0;
            for (let index = 0; index < 4; index++) {
                value += left[index * 4 + row] * right[column * 4 + index];
            }
            out[column * 4 + row] = value;
        }
    }
}

function placeCameraPanels(pose) {
    const m = pose.transform.matrix;
    // The headset looks down its -z axis; m[8..10] is that axis in world
    // space, so this is the heading with pitch and roll discarded.
    const yaw = Math.atan2(m[8], m[10]);
    setLevelAnchorMatrix(panelAnchorMatrix, yaw, m[12], m[13], m[14]);
    panelsPlaced = true;
    for (const panel of panels) updatePanelMatrices(panel);
    multiplyMatrices(hudModelMatrix, panelAnchorMatrix, hudLocalMatrix);
}

function updatePanelMatrices(panel) {
    const { x, y, z, yawDeg, tiltDeg, height } = panel.layout;
    const halfHeight = height * 0.5;
    const halfWidth = halfHeight * panel.aspect;
    setPanelMatrix(panel.localMatrix, halfWidth, halfHeight, x, y, z, yawDeg, tiltDeg);
    setPanelMatrix(
        panel.borderLocalMatrix,
        halfWidth + PANEL_BORDER_METERS,
        halfHeight + PANEL_BORDER_METERS,
        x, y, z, yawDeg, tiltDeg,
    );
    multiplyMatrices(panel.modelMatrix, panelAnchorMatrix, panel.localMatrix);
    multiplyMatrices(panel.borderModelMatrix, panelAnchorMatrix, panel.borderLocalMatrix);
}

function bindQuad(program, attribs, withUv) {
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
    gl.enableVertexAttribArray(attribs.pos);
    gl.vertexAttribPointer(attribs.pos, 2, gl.FLOAT, false, 16, 0);
    if (withUv) {
        gl.enableVertexAttribArray(attribs.uv);
        gl.vertexAttribPointer(attribs.uv, 2, gl.FLOAT, false, 16, 8);
    }
}

function renderTexturedQuad(view, viewport, texture, modelMatrix, viewMatrix) {
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    bindQuad(quadProgram, quadAttribs, true);

    gl.uniformMatrix4fv(quadUniforms.proj, false, view.projectionMatrix);
    gl.uniformMatrix4fv(quadUniforms.view, false, viewMatrix);
    gl.uniformMatrix4fv(quadUniforms.model, false, modelMatrix);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(quadUniforms.tex, 0);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function renderSolidQuad(view, viewport, color, modelMatrix, viewMatrix) {
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    bindQuad(solidProgram, solidAttribs, false);

    gl.uniformMatrix4fv(solidUniforms.proj, false, view.projectionMatrix);
    gl.uniformMatrix4fv(solidUniforms.view, false, viewMatrix);
    gl.uniformMatrix4fv(solidUniforms.model, false, modelMatrix);
    gl.uniform4fv(solidUniforms.color, color);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function panelBorderColor(panel, nowMs) {
    if (hudOffline || !panel.hasFrame || nowMs - panel.lastFrameMs > PANEL_STALE_MS) {
        return BORDER_COLORS.stale;
    }
    if (panel.slot === 'center') {
        return episodeStatus?.state === 'recording' ? BORDER_COLORS.recording : BORDER_COLORS.live;
    }
    return gripHeld[panel.slot] ? BORDER_COLORS.engaged : BORDER_COLORS.idle;
}

// Border first, then the frame (or a dark blank while waiting for one).
// Depth testing stays off, so draw order is paint order.
function renderCameraPanels(view, viewport) {
    if (!panelsPlaced) return;
    const viewMatrix = view.transform.inverse.matrix;
    const nowMs = performance.now();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    for (const panel of panels) {
        renderSolidQuad(
            view, viewport, panelBorderColor(panel, nowMs), panel.borderModelMatrix, viewMatrix,
        );
        if (panel.hasFrame) {
            renderTexturedQuad(view, viewport, panel.texture, panel.modelMatrix, viewMatrix);
        } else {
            renderSolidQuad(view, viewport, PANEL_BLANK_COLOR, panel.modelMatrix, viewMatrix);
        }
    }
    gl.disable(gl.BLEND);
}

function handleServerMessage(data) {
    try {
        const message = JSON.parse(data);
        if (message.type === 'episode_status') {
            episodeStatus = message;
            episodeStatusReceivedAtMs = performance.now();
            hudOffline = false;
            hudDirty = true;
        } else if (message.type === 'export_status') {
            exportStatus = message;
            hudDirty = true;
        }
    } catch (error) {
        console.warn('Ignoring invalid server message', error);
    }
}

function initCollectionHud() {
    hudCanvas = document.createElement('canvas');
    hudCanvas.width = HUD_WIDTH_PX;
    hudCanvas.height = HUD_HEIGHT_PX;
    hudContext = hudCanvas.getContext('2d');
    hudTexture = createTexture();
    hudDrawnKey = null;
    setPanelMatrix(
        hudLocalMatrix,
        HUD_WIDTH_METERS * 0.5,
        HUD_HEIGHT_METERS * 0.5,
        HUD_LAYOUT.x, HUD_LAYOUT.y, HUD_LAYOUT.z, HUD_LAYOUT.yawDeg, HUD_LAYOUT.tiltDeg,
    );
}

// The HUD shows whenever a collection stack reports episodes, or the server
// configured an operator hint.
function hudVisible() {
    return episodeStatus !== null || Boolean(webXRClientConfig?.hud_hint);
}

function collectionElapsedSeconds() {
    if (!episodeStatus || episodeStatus.state !== 'recording') return 0;
    const sinceUpdate = (performance.now() - episodeStatusReceivedAtMs) / 1000;
    return Math.max(0, Math.floor(episodeStatus.elapsed_s + sinceUpdate));
}

function formatElapsed(seconds) {
    const minutes = Math.floor(seconds / 60).toString().padStart(2, '0');
    const remainder = (seconds % 60).toString().padStart(2, '0');
    return `${minutes}:${remainder}`;
}

function hudState() {
    if (hudOffline) return { label: 'OFFLINE', color: HUD_COLORS.offline };
    if (!episodeStatus) return { label: 'TELEOP', color: HUD_COLORS.dim };
    if (episodeStatus.state === 'recording') {
        return {
            label: `REC ${formatElapsed(collectionElapsedSeconds())}`,
            color: HUD_COLORS.recording,
        };
    }
    const recent = performance.now() - episodeStatusReceivedAtMs < HUD_OUTCOME_MS;
    if (recent && episodeStatus.last_event === 'save') {
        return { label: 'SAVED', color: HUD_COLORS.saved };
    }
    if (recent && episodeStatus.last_event === 'discard') {
        return { label: 'DISCARDED', color: HUD_COLORS.ready };
    }
    return { label: 'READY', color: HUD_COLORS.ready };
}

function exportLine() {
    if (!exportStatus) return { text: '', color: HUD_COLORS.dim };
    const name = exportStatus.dataset.split('/').filter(Boolean).pop() ?? '';
    switch (exportStatus.state) {
        case 'exporting':
            return { text: `LEROBOT EXPORTING ${name}`, color: HUD_COLORS.ready };
        case 'done':
            return { text: `LEROBOT ${exportStatus.episodes} EP OK  ${name}`, color: HUD_COLORS.saved };
        case 'failed':
            return { text: `LEROBOT FAILED  ${exportStatus.error ?? ''}`, color: HUD_COLORS.failed };
        default:
            return { text: `LEROBOT ${name}`, color: HUD_COLORS.dim };
    }
}

// Fit text to a width by trimming it with an ellipsis.
function fitText(context, text, maxWidth) {
    if (context.measureText(text).width <= maxWidth) return text;
    let trimmed = text;
    while (trimmed.length > 1 && context.measureText(`${trimmed}...`).width > maxWidth) {
        trimmed = trimmed.slice(0, -1);
    }
    return `${trimmed}...`;
}

// A 5x5 dotted ring like the reference HUD; the center dot blinks while recording.
function drawHudGlyph(context, x, y, size, color, centerOn) {
    const cell = size / 5;
    const radius = cell * 0.32;
    context.fillStyle = color;
    for (let row = 0; row < 5; row++) {
        for (let col = 0; col < 5; col++) {
            const edge = row === 0 || row === 4 || col === 0 || col === 4;
            const center = row === 2 && col === 2;
            if (!edge && !center) continue;
            if (center && !centerOn) continue;
            context.globalAlpha = edge ? 0.8 : 1.0;
            context.beginPath();
            context.arc(x + col * cell + cell / 2, y + row * cell + cell / 2, radius, 0, Math.PI * 2);
            context.fill();
        }
    }
    context.globalAlpha = 1.0;
}

function updateHudTexture() {
    if (!hudContext || !hudVisible()) return;
    const state = hudState();
    const recording = episodeStatus?.state === 'recording';
    const centerOn = !recording || collectionElapsedSeconds() % 2 === 0;
    const exportInfo = exportLine();
    const key = [state.label, exportInfo.text, centerOn, hudOffline].join('|');
    if (!hudDirty && key === hudDrawnKey) return;
    hudDirty = false;
    hudDrawnKey = key;

    const context = hudContext;
    context.clearRect(0, 0, HUD_WIDTH_PX, HUD_HEIGHT_PX);
    context.beginPath();
    context.roundRect(8, 8, HUD_WIDTH_PX - 16, HUD_HEIGHT_PX - 16, 28);
    context.fillStyle = 'rgba(4, 9, 17, 0.86)';
    context.fill();
    context.strokeStyle = 'rgba(188, 210, 235, 0.3)';
    context.lineWidth = 3;
    context.stroke();

    drawHudGlyph(context, 44, 48, 120, state.color, centerOn);

    const left = 200;
    const maxWidth = HUD_WIDTH_PX - left - 40;
    context.fillStyle = state.color;
    context.font = `700 78px ${HUD_FONT}`;
    context.fillText(fitText(context, state.label, maxWidth), left, 118);

    const hint = (webXRClientConfig?.hud_hint ?? '').toUpperCase();
    if (hint) {
        context.fillStyle = HUD_COLORS.dim;
        context.font = `500 28px ${HUD_FONT}`;
        context.fillText(fitText(context, hint, maxWidth), left, 162);
    }

    if (episodeStatus) {
        const task = (episodeStatus.task_label ?? '').toUpperCase();
        context.fillStyle = HUD_COLORS.text;
        context.font = `600 34px ${HUD_FONT}`;
        context.fillText(fitText(context, `TASK  ${task}`, maxWidth), left, 218);

        const saved = episodeStatus.episodes_saved;
        const discarded = episodeStatus.episodes_discarded;
        context.fillStyle = HUD_COLORS.dim;
        context.font = `500 30px ${HUD_FONT}`;
        context.fillText(
            `KEPT ${saved}   DISCARDED ${discarded}   TAKE ${saved + discarded + 1}`, left, 272,
        );
    }

    if (exportInfo.text) {
        context.fillStyle = exportInfo.color;
        context.font = `500 30px ${HUD_FONT}`;
        context.fillText(fitText(context, exportInfo.text, maxWidth), left, 326);
    }

    gl.bindTexture(gl.TEXTURE_2D, hudTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, hudCanvas);
}

function renderCollectionHud(view, viewport) {
    if (!panelsPlaced || !hudVisible() || hudDrawnKey === null) return;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    renderTexturedQuad(view, viewport, hudTexture, hudModelMatrix, view.transform.inverse.matrix);
    gl.disable(gl.BLEND);
}

function sendPose(handedness, pose) {
    const pos = pose.transform.position;
    const rot = pose.transform.orientation;
    const nowMs = Date.now();
    const poseStamped = new geometry_msgs.PoseStamped({
        header: new std_msgs.Header({
            stamp: new std_msgs.Time({ sec: Math.floor(nowMs / 1000), nsec: (nowMs % 1000) * 1_000_000 }),
            frame_id: handedness
        }),
        pose: new geometry_msgs.Pose({
            position: new geometry_msgs.Point({ x: pos.x, y: pos.y, z: pos.z }),
            orientation: new geometry_msgs.Quaternion({ x: rot.x, y: rot.y, z: rot.z, w: rot.w })
        })
    });
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(poseStamped.encode());
    }
}

function sendJoy(handedness, axes, buttons) {
    const nowMs = Date.now();
    const joyMsg = new sensor_msgs.Joy({
        header: new std_msgs.Header({
            stamp: new std_msgs.Time({ sec: Math.floor(nowMs / 1000), nsec: (nowMs % 1000) * 1_000_000 }),
            frame_id: handedness
        }),
        axes_length: axes.length,
        buttons_length: buttons.length,
        axes: axes,
        buttons: buttons
    });
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(joyMsg.encode());
    }
}

// Send raw controller and wrist tracking data (no processing - done in Python)
function processTracking(time, frame) {
    // Rate limit tracking data
    if (time - lastSendTime < sendInterval) {
        return;
    }
    lastSendTime = time;

    // Process controller and hand input sources.
    for (const inputSource of frame.session.inputSources) {
        const handedness = inputSource.handedness;
        if (handedness !== 'left' && handedness !== 'right') continue;

        const hand = inputSource.hand;
        if (hand) {
            const wristPose = frame.getJointPose(hand.get('wrist'), xrRefSpace);
            const thumbTipPose = frame.getJointPose(hand.get('thumb-tip'), xrRefSpace);
            const middleTipPose = frame.getJointPose(hand.get('middle-finger-tip'), xrRefSpace);
            if (!wristPose || !thumbTipPose || !middleTipPose) continue;

            const thumb = thumbTipPose.transform.position;
            const middle = middleTipPose.transform.position;
            const gripperPinched = Math.hypot(
                thumb.x - middle.x,
                thumb.y - middle.y,
                thumb.z - middle.z,
            ) < GRIPPER_PINCH_DISTANCE_METERS;

            sendPose(handedness, wristPose);
            // Index pinch selects arm tracking; middle pinch closes the gripper.
            sendJoy(
                handedness,
                [0, 0, gripperPinched ? 1 : 0, 0],
                [0, 0, 0, 0, handSelectActive.get(handedness) ? 1 : 0, 0, 0],
            );
            continue;
        }

        const trackingSpace = inputSource.gripSpace || inputSource.targetRaySpace;
        if (!trackingSpace) continue;
        const pose = frame.getPose(trackingSpace, xrRefSpace);
        if (!pose) continue;

        sendPose(handedness, pose);

        // Send Joy message with all buttons and axes
        const gamepad = inputSource.gamepad;
        if (gamepad) {
            const isXrStandard = gamepad.mapping === 'xr-standard';
            const stickX = (isXrStandard ? gamepad.axes[2] : gamepad.axes[0]) ?? 0.0;
            const stickY = (isXrStandard ? gamepad.axes[3] : gamepad.axes[1]) ?? 0.0;
            const axes = [
                stickX,
                stickY,
                gamepad.buttons[0]?.value ?? 0.0,
                gamepad.buttons[1]?.value ?? 0.0,
            ];

            // Buttons layout (int32, 0 or 1):
            // [0] = trigger (digital)
            // [1] = grip (digital)
            // [2] = touchpad press
            // [3] = thumbstick press
            // [4] = X/A button
            // [5] = Y/B button
            // [6] = menu (if exposed)
            // Pad to at least 7 entries: the Python side
            // (WebXRControllerState.from_joy) requires the full layout,
            // but browsers only report the buttons the controller has
            // (e.g. 6 when no menu/thumbrest is exposed).
            const buttons = [];
            const buttonCount = Math.max(gamepad.buttons.length, 7);
            for (let i = 0; i < buttonCount; i++) {
                buttons.push(gamepad.buttons[i]?.pressed ? 1 : 0);
            }
            // The grip is the arm deadman; the wrist panel on this side shows it.
            gripHeld[handedness] = buttons[1] === 1;

            sendJoy(handedness, axes, buttons);
        }
    }

    if (webXRClientConfig.body_tracking_mode !== 'off') {
        const joints = captureBody(frame, xrBodyRefSpace);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'body_tracking_snapshot',
                capture_time_s: (performance.timeOrigin + time) / 1000,
                frame_id: xrBodyRefSpaceType,
                joints,
            }));
        }
    }
}

// WebXR render loop
function onXRFrame(time, frame) {
    if (!xrSession) return;
    xrSession.requestAnimationFrame(onXRFrame);
    // Process and send tracking data
    processTracking(time, frame);

    const glLayer = xrSession.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Textures keep their last frame between uploads, so panels never blink.
    uploadPanelFrames();
    updateHudTexture();
    const pose = frame.getViewerPose(xrRefSpace);
    if (pose) {
        if (!panelsPlaced) placeCameraPanels(pose);
        for (const view of pose.views) {
            const viewport = glLayer.getViewport(view);
            renderCameraPanels(view, viewport);
            renderCollectionHud(view, viewport);
        }
    }
}

// Start an immersive WebXR session with passthrough when available.
async function startWebXRSession(clientConfig) {
    try {
        setStatus('Initializing WebGL...');
        initGL();
        setStatus('Requesting WebXR session...');

        let session = null;
        const failures = [];
        for (const mode of clientConfig.session_modes) {
            try {
                session = await navigator.xr.requestSession(mode, clientConfig.session_options);
                console.log(`Started ${mode} session`);
                break;
            } catch (error) {
                const failure = describeSessionRequestError(mode, error);
                failures.push(failure);
                console.warn(`WebXR session request failed: ${failure}`);
            }
        }
        if (!session) {
            throw new Error(`WebXR session request failed: ${failures.join('; ')}`);
        }

        xrSession = session;
        panelsPlaced = false;
        lastSendTime = 0;

        // Setup WebGL layer
        const glLayer = new XRWebGLLayer(session, gl);
        await session.updateRenderState({
            baseLayer: glLayer
        });

        // Get reference space
        xrRefSpace = await session.requestReferenceSpace('local-floor');

        if (clientConfig.body_tracking_mode !== 'off') {
            try {
                xrBodyRefSpace = await session.requestReferenceSpace('bounded-floor');
                xrBodyRefSpaceType = 'bounded-floor';
            } catch (error) {
                console.warn('bounded-floor unavailable; using local-floor for body poses', error);
                xrBodyRefSpace = xrRefSpace;
                xrBodyRefSpaceType = 'local-floor';
            }
        }

        setStatus(`WebXR active (${session.mode})`);

        // Session event handlers
        session.addEventListener('end', () => {
            setStatus('WebXR session ended');
            handSelectActive.clear();
            gripHeld.left = false;
            gripHeld.right = false;
            panelsPlaced = false;
            xrSession = null;
            xrBodyRefSpace = null;
            xrBodyRefSpaceType = null;
            window.disconnect();
        });

        session.addEventListener('selectstart', (event) => {
            const { handedness, hand } = event.inputSource;
            if (hand && (handedness === 'left' || handedness === 'right')) {
                handSelectActive.set(handedness, true);
            }
        });
        session.addEventListener('selectend', (event) => {
            const { handedness, hand } = event.inputSource;
            if (hand && (handedness === 'left' || handedness === 'right')) {
                handSelectActive.set(handedness, false);
            }
        });

        // Start render loop
        session.requestAnimationFrame(onXRFrame);

    } catch (error) {
        setStatus('WebXR failed: ' + error.message);
        console.error('WebXR session error:', error);
        throw error;
    }
}

// Connect button handler
window.connect = async function() {
    try {
        connectBtn.disabled = true;

        if (
            webXRClientConfig?.dataset_label_enabled
            && datasetLabelInput.value.trim()
            && datasetLabelInput.value.trim() !== selectedDatasetSuffix
        ) {
            throw new Error('Press Set label or Enter before connecting.');
        }

        // Check WebXR support
        if (!navigator.xr) {
            throw new Error('WebXR not supported. Use a WebXR-capable browser.');
        }
        if (!webXRClientConfig) {
            throw new Error('WebXR configuration is unavailable. Reload the page and try again.');
        }

        // Immersive sessions must be requested while this click still carries
        // transient user activation. In particular, required body tracking can
        // trigger a consent check, so do not await network setup first.
        await startWebXRSession(webXRClientConfig);

        // Connect the data channel after the browser grants the XR session.
        await setupWebSocket();

        // Update UI
        connectBtn.classList.add('hidden');
        disconnectBtn.classList.remove('hidden');

    } catch (error) {
        const message = error?.message || String(error);
        const failedSession = xrSession;
        xrSession = null;
        if (failedSession) await failedSession.end().catch(console.error);
        const failedWebSocket = ws;
        ws = null;
        if (failedWebSocket) failedWebSocket.close();
        setStatus(`Connection failed: ${message}`);
        console.error('Connection error:', error);
        connectBtn.disabled = false;
    }
};

// Disconnect button handler
window.disconnect = async function() {
    setStatus('Disconnecting...');

    if (xrSession) {
        await xrSession.end().catch(console.error);
        xrSession = null;
    }

    if (ws) {
        ws.close();
        ws = null;
    }

    // Update UI
    connectBtn.classList.remove('hidden');
    connectBtn.disabled = false;
    disconnectBtn.classList.add('hidden');
    setStatus('Disconnected');
};

// Check WebXR availability on load
window.addEventListener('load', async () => {
    if (!navigator.xr) {
        setStatus('WebXR not available');
        connectBtn.disabled = true;
        return;
    }

    try {
        webXRClientConfig = await loadWebXRClientConfig();
        if (webXRClientConfig.dataset_label_enabled) {
            datasetLabelForm.classList.remove('hidden');
            datasetLabelInput.maxLength = webXRClientConfig.dataset_suffix_max_length * 2;
            if (webXRClientConfig.dataset_suffix) {
                selectedDatasetSuffix = webXRClientConfig.dataset_suffix;
                datasetLabelInput.value = selectedDatasetSuffix;
                setDatasetLabelStatus(`Dataset folder will end in _${selectedDatasetSuffix}`);
            }
        }
        await Promise.all(webXRClientConfig.session_modes.map(async (mode) => {
            const supported = await navigator.xr.isSessionSupported(mode).catch(() => false);
            sessionModeSupport.set(mode, supported);
        }));

        const supported = webXRClientConfig.session_modes.some(
            (mode) => sessionModeSupport.get(mode),
        );
        if (!supported) {
            setStatus(`Session modes unsupported: ${webXRClientConfig.session_modes.join(', ')}`);
            connectBtn.disabled = true;
        }
    } catch (error) {
        setStatus(error?.message || String(error));
        connectBtn.disabled = true;
        console.error('WebXR setup failed:', error);
    }
});
