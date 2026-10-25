// Copyright 2026 Dimensional Inc. Licensed under the Apache License, Version 2.0.
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {ToolPinch} from './spatial_input.mjs';
import {decodeSurface, SurfaceGrid} from './spatial_surface.mjs';
import {SpatialUI} from './spatial_ui.mjs';
import {WorkspaceTransform} from './spatial_workspace.mjs';

const embedded = new URLSearchParams(location.search).has('embedded');
if (embedded) document.body.classList.add('embedded');
const apiBase = new URL('./spatial/', location.href);
const endpoint = path => new URL(path, apiBase).href;
const status = document.querySelector('#status');
const enter = document.querySelector('#enter');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.xr.enabled = !embedded;
renderer.xr.setReferenceSpaceType('local-floor');
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#11181d');
scene.add(new THREE.HemisphereLight(0xffffff, 0x536976, 2));
const light = new THREE.DirectionalLight(0xffffff, 2);
light.position.set(1, 3, 2); scene.add(light);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.03, 15);
camera.position.set(1.2, 1.6, 1.5);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 1, -0.4); orbit.update();

// World X forward / Y left / Z up -> XR X right / Y up / Z back.
// This SAME matrix is inverted for every hand command and surface selection.
const basis = new THREE.Matrix4().set(0,-1,0,0, 0,0,1,0, -1,0,0,0, 0,0,0,1);
const workspace = new WorkspaceTransform();
const stage = new THREE.Group(); stage.matrixAutoUpdate = false;
scene.add(stage); applyPlacement();
const inverseStage = new THREE.Matrix4();
const arms = {}, handles = {}, gestures = {}, handDots = {}, openingControls = {};
const colours = {left: 0x76e2c1, right: 0xfab972};
for (const side of ['left', 'right']) {
  arms[side] = new THREE.Group(); stage.add(arms[side]);
  const handle = new THREE.Mesh(new THREE.SphereGeometry(0.035, 16, 12),
    new THREE.MeshBasicMaterial({color: colours[side], transparent: true, opacity: 0.7, wireframe: true}));
  handles[side] = handle; stage.add(handle); handle.visible = false;
  gestures[side] = new ToolPinch();
  const control = new THREE.Group(); stage.add(control); control.visible = false;
  const material = new THREE.MeshBasicMaterial({color: colours[side], transparent:true, opacity:0.65});
  const rail = new THREE.Mesh(new THREE.BoxGeometry(1,0.003,0.003), material);
  control.add(rail);
  const measured = new THREE.Mesh(new THREE.SphereGeometry(0.012,16,12),material);
  const requested = new THREE.Mesh(new THREE.SphereGeometry(0.018,16,12),
    new THREE.MeshBasicMaterial({color:0xffffff, wireframe:true}));
  control.add(measured,requested); requested.visible=false;
  const canvas = document.createElement('canvas'); canvas.width=512; canvas.height=128;
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace=THREE.SRGBColorSpace;
  const label = new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false}));
  label.scale.set(0.19,0.0475,1); label.position.set(0,0.05,0); control.add(label);
  openingControls[side] = {group:control,rail,measured,requested,canvas,texture,label,lastText:''};
  handDots[side] = [];
  for (let i = 0; i < 25; i++) {
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.005, 8, 6), new THREE.MeshBasicMaterial({color: colours[side]}));
    dot.visible = false; scene.add(dot); handDots[side].push(dot);
  }
}
const cloud = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({side:THREE.DoubleSide}));
cloud.matrixAutoUpdate = false;
stage.add(cloud);
let surfaceGrid = null;
const detections = new THREE.Group(); stage.add(detections);
const focusMarker = new THREE.Mesh(new THREE.SphereGeometry(0.018, 12, 8), new THREE.MeshBasicMaterial({color: 0xffffff, wireframe: true}));
focusMarker.visible = false; stage.add(focusMarker);
let feedback = null, feedbackAt = 0, cloudAt = 0, cloudTS = 0, sequence = 0, socket = null;
let placeAhead = true, lastSend = 0, stopped = false, controlRequested = false;
let entryError = '';
let paused = false, inputGeneration=0, observationsReady=false;
const ui = new SpatialUI(stage,endpoint,{embedded,onAction:action=>{
  if(action==='hold') paused=!paused;
  if(action==='actual_size') { workspace.resize(1,ui.workspaceAnchor()); applyPlacement(); }
  if(action==='place_ahead') placeAhead=true;
  resetHands(); send({});
}});
const uiPointer=new THREE.Mesh(new THREE.SphereGeometry(0.007,8,6),new THREE.MeshBasicMaterial({color:0xffffff,depthTest:false}));
scene.add(uiPointer); uiPointer.visible=false;
const geometry = await fetch(endpoint('model')).then(check).then(r => r.json()).catch(error => {
  status.textContent = `Cannot load station: ${error.message}. Reload after the blueprint is running.`;
  throw error;
});
const meshes = {};
if(!geometry.finger_geometry) {
  const note=document.querySelector('#model-note'); note.hidden=false;
  note.textContent='Measured jaw opening appears on each tool. Fingertips are visible through the depth camera.';
}
for (const [id, mesh] of Object.entries(geometry.meshes)) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(mesh.vertices, 3));
  g.setIndex(mesh.faces); g.computeVertexNormals(); meshes[id] = g;
}
for (const side of geometry.sides) {
  for (const geom of geometry.arms[side]) {
    const mesh = new THREE.Mesh(meshes[geom.mesh], new THREE.MeshStandardMaterial({color: new THREE.Color(...geom.rgba.slice(0,3)), roughness: 0.65}));
    mesh.matrixAutoUpdate = false; arms[side].add(mesh);
  }
  arms[side].visible = false;
}

function check(response) { if (!response.ok) throw new Error(`Station HTTP ${response.status}`); return response; }
async function poll(path, interval, accept) {
  while (!stopped) {
    try { await accept(await fetch(path, {cache: 'no-store'}).then(check)); }
    catch (error) { console.warn(path, error.message); }
    await new Promise(resolve => setTimeout(resolve, interval));
  }
}
function connect() {
  if (!controlRequested) return;
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.onopen = () => { sequence = 0; entryError=''; resetHands(); };
  socket.onclose = event => { resetHands(); if(event.code===1008) entryError='Another operator currently owns control.'; if (!stopped && controlRequested) setTimeout(connect, 1500); };
}
poll(endpoint('state'), 35, async response => {
  feedback = await response.json(); feedbackAt = performance.now();
  for (const side of geometry.sides) {
    const arm = feedback.arms?.[side]; if (!arm) continue;
    arm.transforms.forEach((matrix, i) => arms[side].children[i].matrix.fromArray(matrix));
    handles[side].position.fromArray(arm.grasp);
    const c=openingControls[side], q=new THREE.Quaternion().fromArray(arm.orientation);
    c.group.position.copy(handles[side].position).add(new THREE.Vector3().fromArray(feedback.opening_handle_offset).applyQuaternion(q));
    c.group.quaternion.copy(q);
    c.rail.scale.x=feedback.opening_handle_travel_m;
    c.measured.position.x=(arm.opening-0.5)*feedback.opening_handle_travel_m;
    const text=`${side.toUpperCase()} · ${Math.round(arm.opening*100)}% OPEN`;
    if(text!==c.lastText) {
      c.lastText=text; const ctx=c.canvas.getContext('2d'); ctx.clearRect(0,0,512,128);
      ctx.fillStyle='#13242ee8'; ctx.fillRect(0,0,512,128);
      ctx.fillStyle=side==='left'?'#76e2c1':'#fab972'; ctx.font='bold 32px system-ui';
      ctx.textAlign='center'; ctx.fillText(text,256,47);
      ctx.fillStyle='#ffffff'; ctx.font='25px system-ui'; ctx.fillText('CLOSE   ←  PINCH & SLIDE  →   OPEN',256,96);
      c.texture.needsUpdate=true;
    }
  }
  document.querySelector('#perception').textContent = feedback.objects_active
    ? `Object perception connected · ${feedback.objects.length} current detections`
    : 'Object recognition: no detector connected.';
  for (const child of [...detections.children]) { child.geometry.dispose(); child.material.dispose(); detections.remove(child); }
  for (const object of feedback.objects) {
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(...object.size)), new THREE.LineBasicMaterial({color: 0x88ceff}));
    box.position.fromArray(object.position); box.quaternion.fromArray(object.orientation); detections.add(box);
  }
});
// Each request asks only for a newer acquisition. One in flight; no frame queue.
async function pollSurface() {
  while(!stopped) {
    try {
      const response=check(await fetch(endpoint(`surface?after=${cloudTS}`),{cache:'no-store'}));
      if(response.status!==204) {
        const {header:h,depth,jpeg}=decodeSurface(await response.arrayBuffer());
        const bitmap=await createImageBitmap(new Blob([jpeg],{type:'image/jpeg'}),{imageOrientation:'flipY'});
        if(!surfaceGrid || surfaceGrid.width!==h.width || surfaceGrid.height!==h.height) {
          surfaceGrid=new SurfaceGrid(h.width,h.height);
          cloud.geometry.dispose(); cloud.geometry=new THREE.BufferGeometry();
          cloud.geometry.setAttribute('position',new THREE.BufferAttribute(surfaceGrid.positions,3).setUsage(THREE.DynamicDrawUsage));
          cloud.geometry.setAttribute('uv',new THREE.BufferAttribute(surfaceGrid.uv,2));
          cloud.geometry.setIndex(new THREE.BufferAttribute(surfaceGrid.indices,1).setUsage(THREE.DynamicDrawUsage));
        }
        surfaceGrid.update(h,depth);
        for(const attribute of Object.values(cloud.geometry.attributes)) attribute.needsUpdate=true;
        cloud.geometry.index.needsUpdate=true; cloud.geometry.setDrawRange(0,surfaceGrid.count);
        cloud.geometry.computeBoundingSphere(); cloud.matrix.fromArray(h.world_from_optical);
        if(!cloud.material.map) {
          cloud.material.map=new THREE.Texture(); cloud.material.map.colorSpace=THREE.SRGBColorSpace;
          cloud.material.map.generateMipmaps=false; cloud.material.map.minFilter=THREE.LinearFilter;
          cloud.material.needsUpdate=true;
        }
        cloud.material.map.image?.close(); cloud.material.map.image=bitmap; cloud.material.map.needsUpdate=true;
        cloudTS=h.ts; cloudAt=performance.now();
      }
    } catch(error) { console.warn('RGB-D:',error.message); }
    await new Promise(resolve=>setTimeout(resolve,25));
  }
}
pollSurface();
function resetTools() { inputGeneration++; for (const state of Object.values(gestures)) state.reset(); }
function resetHands() { resetTools(); workspace.cancel(); }
function applyPlacement() {
  stage.matrix.makeRotationY(workspace.yaw).multiply(basis)
    .scale(new THREE.Vector3().setScalar(workspace.scale)).setPosition(...workspace.position);
  stage.updateMatrixWorld(true);
}
function fresh() {
  const now=feedback?.server_time+(performance.now()-feedbackAt)/1000;
  return feedback?.arms && performance.now()-feedbackAt < 350 && performance.now()-cloudAt < 500
    && now-feedback.ts < feedback.observation_timeout_s
    && now-cloudTS > -0.1 && now-cloudTS < feedback.observation_timeout_s;
}
function place(viewer) {
  const position = new THREE.Vector3().copy(viewer.transform.position);
  const heading = new THREE.Vector3(0,0,-1).applyQuaternion(new THREE.Quaternion().copy(viewer.transform.orientation));
  heading.y = 0; if (heading.length() < 0.1) return;
  heading.normalize();
  const yaw = Math.atan2(-heading.x, -heading.z);
  workspace.place([position.x + heading.x*0.15,Math.max(0.3,position.y-0.8),position.z + heading.z*0.15],yaw);
  applyPlacement(); resetHands(); send({}); placeAhead = false;
}
document.querySelector('#place').onclick = () => { placeAhead=true; resetHands(); send({}); };
enter.onclick = async () => {
  entryError = '';
  try {
    const session = await navigator.xr.requestSession('immersive-vr', {requiredFeatures: ['local-floor', 'hand-tracking']});
    controlRequested = true; paused=false;
    session.addEventListener('end', () => { controlRequested = false; resetHands(); send({}); socket?.close(); enter.disabled = false; orbit.enabled = true; });
    session.addEventListener('visibilitychange', () => { if (session.visibilityState !== 'visible') { resetHands(); send({}); } });
    await renderer.xr.setSession(session); connect(); placeAhead = true; orbit.enabled = false; enter.disabled = true;
  } catch (error) { controlRequested = false; socket?.close(); entryError = `Cannot enter WebXR: ${error.message}`; }
};
if (!embedded && navigator.xr && await navigator.xr.isSessionSupported('immersive-vr')) enter.disabled = false;
else enter.textContent = 'WebXR unavailable in this browser';

function send(hands, focus = null) {
  if (socket?.readyState !== WebSocket.OPEN) return;
  if (socket.bufferedAmount > 16384) { socket.close(); resetHands(); return; }
  socket.send(JSON.stringify({type: 'spatial_input', sequence: sequence++, observed_at: feedback?.ts || 0, hands, focus}));
}
function track(frame, reference, ready) {
  const samples={};
  ui.hovered=null; uiPointer.visible=false;
  for (const source of frame.session.inputSources) {
    // Vision Pro's gaze+pinch transient-pointer is NOT a hand pose source.
    if (!source.hand || source.targetRayMode !== 'tracked-pointer' || !gestures[source.handedness]) continue;
    const side = source.handedness;
    const poses = {};
    let index = 0;
    for (const [name, joint] of source.hand) {
      const pose = frame.getJointPose(joint, reference); poses[name] = pose;
      const dot = handDots[side][index++]; if (!dot) continue;
      dot.visible = !!pose; if (pose) dot.position.copy(pose.transform.position);
    }
    const thumb = poses['thumb-tip'], tip = poses['index-finger-tip'], wrist = poses.wrist;
    if (!thumb || !tip || !wrist) { gestures[side].reset(); continue; }
    const indexPosition = new THREE.Vector3().copy(tip.transform.position);
    const thumbPosition = new THREE.Vector3().copy(thumb.transform.position);
    samples[side]={indexPosition,thumbPosition,wrist,knuckle:poses['index-finger-metacarpal'],
      position:indexPosition.clone().add(thumbPosition).multiplyScalar(0.5),
      distance:indexPosition.distanceTo(thumbPosition)};
  }
  for(const side of ['left','right']) if(!samples[side]) {
    gestures[side].reset(); for(const dot of handDots[side]) dot.visible=false;
  }
  stage.updateMatrixWorld(true);
  const arranging=workspace.update(Object.fromEntries(Object.entries(samples).map(([side,hand])=>[
    side,{position:hand.position.toArray(),distance:hand.distance,nearRail:ui.nearWorkspace(hand.position)},
  ])));
  if(arranging) {
    applyPlacement(); resetTools(); send({}); return;
  }
  const inputs = {}; let focus = null;
  const generation=inputGeneration;
  inverseStage.copy(stage.matrixWorld).invert();
  // Uniform view scale affects positions, never the hand's orientation quaternion.
  const rotation = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().extractRotation(inverseStage));
  for(const [side,hand] of Object.entries(samples)) {
    const {indexPosition,thumbPosition,wrist,knuckle}=hand;
    const pinchPosition=hand.position.clone().applyMatrix4(inverseStage);
    const c=openingControls[side], arm=feedback?.arms?.[side];
    const axis=new THREE.Vector3(1,0,0).applyQuaternion(c.group.quaternion);
    const knob=c.measured.position.clone().applyQuaternion(c.group.quaternion).add(c.group.position);
    const direction=knuckle?indexPosition.clone().sub(new THREE.Vector3().copy(knuckle.transform.position)):null;
    const ray=direction?.length()>0.01?new THREE.Raycaster(indexPosition,direction.normalize(),0,3):null;
    const uiHit=ray?ui.hit(ray):null;
    const toolTarget=!ready||!arm||paused?null:pinchPosition.distanceTo(knob)<0.045 ? 'opening'
      : pinchPosition.distanceTo(handles[side].position)<0.10 ? 'move' : null;
    const target=toolTarget || (uiHit?`ui:${uiHit.action}`:null);
    if(!toolTarget && uiHit) { ui.hovered=uiHit.action; uiPointer.position.copy(uiHit.point); uiPointer.visible=true; }
    const gesture=gestures[side].update(indexPosition.distanceTo(thumbPosition),target,
      pinchPosition.toArray(),axis.toArray(),arm?.opening??0,feedback?.opening_handle_travel_m??1);
    const orientation = rotation.clone().multiply(new THREE.Quaternion().copy(wrist.transform.orientation));
    if(ready && arm) inputs[side] = {position: pinchPosition.toArray(), orientation: orientation.toArray(),
      pinched:gesture.pinched, action:['move','opening'].includes(gesture.action)?gesture.action:null};
    if(gesture.pressed && gesture.action?.startsWith('ui:')) ui.activate(gesture.action.slice(3));
    handles[side].material.opacity = gesture.action==='move' ? 1 : 0.45;
    c.measured.material.opacity=target==='opening' || gesture.action==='opening' ? 1 : 0.55;
    if (ready && gesture.pressed && target===null) {
      if (ray) {
        const hit = ray.intersectObject(cloud)[0];
        if (hit) { focus = hit.point.clone().applyMatrix4(inverseStage).toArray(); focusMarker.position.fromArray(focus); focusMarker.visible = true; }
      }
    }
  }
  send(generation===inputGeneration?inputs:{}, focus);
}
renderer.setAnimationLoop((time, frame) => {
  const ready = !!fresh();
  if(!ready && observationsReady) resetTools();
  observationsReady=ready;
  for (const side of ['left', 'right']) {
    arms[side].visible = !!feedback?.arms?.[side]; handles[side].visible = ready;
    const c=openingControls[side]; c.group.visible=ready;
    c.requested.visible=ready && gestures[side].action==='opening';
    if(c.requested.visible) c.requested.position.x=(gestures[side].opening-0.5)*feedback.opening_handle_travel_m;
  }
  cloud.visible = ready;
  detections.visible=ready;
  if(!ready) focusMarker.visible=false;
  status.textContent = entryError || (ready ? `Live · ${surfaceGrid?.valid || 0} depth samples · ${Math.round((feedback.server_time-cloudTS)*1000)} ms old · ${!controlRequested ? 'desktop preview' : paused ? 'control paused' : socket?.readyState === WebSocket.OPEN ? 'control connected' : 'control disconnected'}` : (feedback?.observation_error || 'Waiting for fresh arm feedback and RGB-D observations · control held'));
  ui.update({now:feedback?feedback.server_time+(performance.now()-feedbackAt)/1000:Date.now()/1000,
    ready,connected:socket?.readyState===WebSocket.OPEN,paused,actions:feedback?.actions,preview:!controlRequested,
    workspaceScale:workspace.scale,workspaceActive:workspace.active});
  if (frame) {
    const reference = renderer.xr.getReferenceSpace();
    const viewer = frame.getViewerPose(reference);
    if (placeAhead && viewer) place(viewer);
    if (time-lastSend > 20) {
      lastSend = time;
      if (frame.session.visibilityState === 'visible') track(frame, reference, ready);
      else { resetHands(); uiPointer.visible=false; send({}); }
    }
  } else orbit.update();
  renderer.render(scene, camera);
});
addEventListener('resize', () => { camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
addEventListener('pagehide', () => { stopped = true; resetHands(); send({}); socket?.close(); ui.dispose(); });
