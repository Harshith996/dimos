// Copyright 2026 Dimensional Inc. Licensed under the Apache License, Version 2.0.
import * as THREE from 'three';

const panelRotation = new THREE.Matrix4().makeBasis(
  new THREE.Vector3(0,-1,0), new THREE.Vector3(0,0,1), new THREE.Vector3(-1,0,0));
const cameraViews = [['left','LEFT WRIST',0.48],['workspace','WORKSPACE',0],['right','RIGHT WRIST',-0.48]];
const RAIL_LENGTH = 0.9;
const RAIL_GRAB_RADIUS = 0.05; // Maximum physical XR radius; shrink with miniature views.

function panel(parent, width, height, position, pixels) {
  const canvas=document.createElement('canvas'); [canvas.width,canvas.height]=pixels;
  const texture=new THREE.CanvasTexture(canvas); texture.colorSpace=THREE.SRGBColorSpace;
  texture.minFilter=THREE.LinearFilter; texture.generateMipmaps=false;
  const mesh=new THREE.Mesh(new THREE.PlaneGeometry(width,height),new THREE.MeshBasicMaterial({map:texture,side:THREE.DoubleSide}));
  mesh.position.set(...position); mesh.quaternion.setFromRotationMatrix(panelRotation); parent.add(mesh);
  return {canvas,texture,mesh,context:canvas.getContext('2d')};
}

function wrapped(context, text, x, y, width, lineHeight) {
  let line='';
  for(const word of text.split(' ')) {
    if(line && context.measureText(`${line} ${word}`).width>width) {
      context.fillText(line,x,y); line=word; y+=lineHeight;
    } else line+=`${line?' ':''}${word}`;
  }
  context.fillText(line,x,y);
}

// Owns camera/HUD resources. It only invokes episode RPCs; arm input stays in spatial.js.
export class SpatialUI {
  constructor(parent, endpoint, {embedded, onAction}) {
    this.endpoint=endpoint; this.embedded=embedded; this.onAction=onAction;
    this.abort=new AbortController(); this.stopped=false; this.pending=null; this.version=0;
    this.collection=null; this.collectionAt=0; this.error=''; this.pollError=''; this.hovered=null; this.drawn='';
    this.cameras=new Map(); this.buttons=[];
    this.workspaceButtons=[]; this.workspaceDrawn='';
    this.hud=panel(parent,0.84,0.315,[0.82,0,1.05],[1024,384]);
    this.workspacePanel=panel(parent,0.84,0.15,[0.2,0,-0.16],[1024,184]);
    // Keep arrangement affordances readable through the observed tabletop, like tool labels.
    this.workspacePanel.mesh.material.depthTest=false;
    this.workspacePanel.mesh.material.depthWrite=false;
    this.workspacePanel.mesh.renderOrder=10;
    this.rail=new THREE.Mesh(new THREE.CylinderGeometry(0.012,0.012,RAIL_LENGTH,12),
      new THREE.MeshBasicMaterial({color:0xb9cecf,depthTest:false,depthWrite:false}));
    this.rail.renderOrder=11;
    this.rail.position.set(0.2,0,-0.05); parent.add(this.rail);
    for(const y of [-RAIL_LENGTH/2,RAIL_LENGTH/2]) {
      const end=new THREE.Mesh(new THREE.SphereGeometry(0.018,12,8),this.rail.material);
      end.renderOrder=11;
      end.position.y=y; this.rail.add(end);
    }
    for(const [name,title,y] of cameraViews) {
      const view=panel(parent,0.42,0.3465,[0.78,y,0.68],[640,528]);
      Object.assign(view,{name,title,ts:0,url:null,stale:true}); this.cameras.set(name,view);
      this.cameraStatus(view,false); this.poll(()=>this.loadCamera(view),150);
    }
    this.poll(()=>this.loadCollection(),500);
    for(const button of document.querySelectorAll('[data-action]')) {
      button.onclick=()=>this.activate(button.dataset.action);
    }
  }

  async request(path, options={}) {
    const response=await fetch(this.endpoint(path),{cache:'no-store',signal:this.abort.signal,...options});
    if(!response.ok) {
      const data=await response.json().catch(()=>({}));
      throw new Error(typeof data.detail==='string'?data.detail:`Station HTTP ${response.status}`);
    }
    return response;
  }

  async poll(load, delay) {
    while(!this.stopped) {
      try { await load(); }
      catch(error) { if(!this.stopped) this.pollError=error.message; }
      if(!this.stopped) await new Promise(resolve=>setTimeout(resolve,delay));
    }
  }

  async loadCamera(view) {
    let response;
    try { response=await this.request(`camera/${view.name}`); }
    catch(error) { if(!this.stopped) this.cameraStatus(view,false); return; }
    const ts=Number(response.headers.get('X-Frame-Timestamp'));
    if(!Number.isFinite(ts) || ts<=view.ts) return;
    const blob=await response.blob(), bitmap=await createImageBitmap(blob);
    if(this.stopped) { bitmap.close(); return; }
    const c=view.context;
    c.fillStyle='#0c141b'; c.fillRect(0,48,640,480);
    const scale=Math.min(640/bitmap.width,480/bitmap.height);
    const w=bitmap.width*scale,h=bitmap.height*scale;
    c.drawImage(bitmap,(640-w)/2,48+(480-h)/2,w,h); bitmap.close();
    view.ts=ts; this.cameraStatus(view,true);
    if(!this.embedded) {
      const img=document.querySelector(`#${view.name}-camera`);
      const url=URL.createObjectURL(blob); img.src=url; img.hidden=false;
      if(view.url) URL.revokeObjectURL(view.url); view.url=url;
    }
  }

  cameraStatus(view, live) {
    const c=view.context; view.stale=!live;
    c.fillStyle='#13232d'; c.fillRect(0,0,640,48);
    c.fillStyle=view.name==='left'?'#76e2c1':view.name==='right'?'#fab972':'#c4dcf2';
    c.font='bold 23px system-ui'; c.textAlign='left'; c.fillText(view.title,18,32);
    c.fillStyle=live?'#76e2c1':'#f3bf77'; c.textAlign='right'; c.fillText(live?'LIVE':'UNAVAILABLE',622,32);
    if(!live) {
      c.fillStyle='#0c141b'; c.fillRect(0,48,640,480);
      c.textAlign='center'; c.fillStyle='#9db0b8'; c.font='24px system-ui';
      c.fillText('Waiting for camera',320,290);
      document.querySelector(`#${view.name}-camera`).hidden=true;
    }
    document.querySelector(`#${view.name}-caption`).textContent=`${view.title} · ${live?'Live':'Unavailable'}`;
    view.texture.needsUpdate=true;
  }

  async loadCollection() {
    if(this.pending) return;
    const version=this.version;
    const state=await this.request('collection').then(r=>r.json());
    if(version===this.version && !this.pending) {
      this.collection=state; this.collectionAt=performance.now(); this.pollError='';
    }
  }

  async activate(action) {
    if(![...this.buttons,...this.workspaceButtons].some(button=>button.action===action && button.enabled)) return;
    if(['hold','actual_size','place_ahead'].includes(action)) { this.onAction(action); return; }
    if(this.pending) return;
    this.pending=action; this.version++; this.error='';
    try {
      this.collection=await this.request('collection',{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({action,task:document.querySelector('#task').value.trim()}),
      }).then(r=>r.json());
      this.collectionAt=performance.now();
    } catch(error) { this.error=error.message; }
    finally { this.pending=null; }
  }

  update({now,ready,connected,paused,actions,preview,workspaceScale=1,workspaceActive=false}) {
    this.updateWorkspace(workspaceScale,workspaceActive,!preview);
    for(const view of this.cameras.values()) {
      if(!view.stale && now-view.ts>0.5) this.cameraStatus(view,false);
    }
    const state=this.collection;
    const available=state && performance.now()-this.collectionAt<2000;
    const recording=available && !!state.recording;
    const elapsed=recording?Math.max(0,Math.floor(now-state.started_at)):0;
    const timer=`${Math.floor(elapsed/60).toString().padStart(2,'0')}:${(elapsed%60).toString().padStart(2,'0')}`;
    const active=connected && !preview && !workspaceActive;
    this.buttons=[
      {action:'record',label:'Record',enabled:active && ready && available && state.collection_ready && !recording && !!document.querySelector('#task').value.trim()},
      {action:'save',label:'Save',enabled:active && recording},
      {action:'discard',label:'Discard',enabled:active && recording},
      {action:'hold',label:paused?'Resume':'Hold control',enabled:active && (paused?ready:true)},
    ].map((button,i)=>({...button,x:24+i*250,y:292,w:232,h:68,enabled:button.enabled && (!this.pending || button.action==='hold')}));
    const control=preview?'VIEW ONLY':workspaceActive?'ARRANGING WORKSPACE · ARMS HELD':!connected?'CONTROL DISCONNECTED':!ready?'OBSERVATIONS STALE · HELD':paused?'CONTROL PAUSED':'CONTROL CONNECTED';
    const take=this.pending?({record:'STARTING',save:'SAVING',discard:'DISCARDING'}[this.pending])
      :recording?`REC ${timer}`:!available?'RECORDER UNAVAILABLE':state.last_event==='save'?'SAVED':state.last_event==='discard'?'DISCARDED':'READY';
    const hint=this.error || this.pollError || state?.collection_error || (recording?state.task:state?.collection_waiting)
      || 'Point at a button and pinch. Tool handles move only their own arm.';
    const arms=['left','right'].map(side=>`${side.toUpperCase()} ${ready&&!paused&&!workspaceActive?(actions?.[side]==='move'?'MOVING':actions?.[side]==='opening'?'JAWS':'HELD'):'HELD'}`).join('   ·   ');
    const summary=`${take} · ${state?.episodes_saved??0} saved · ${state?.episodes_discarded??0} discarded`;
    document.querySelector('#episode-state').textContent=summary;
    document.querySelector('#episode-hint').textContent=hint;
    document.querySelector('#task').disabled=recording || !!this.pending;
    for(const button of this.buttons) {
      const element=document.querySelector(`[data-action="${button.action}"]`);
      element.disabled=!button.enabled; element.textContent=button.label;
    }
    const signature=JSON.stringify([control,summary,arms,hint,this.buttons,this.hovered]);
    if(signature===this.drawn) return; this.drawn=signature;
    const c=this.hud.context;
    c.fillStyle='#13232d'; c.fillRect(0,0,1024,384); c.textAlign='left';
    c.fillStyle=active&&ready&&!paused?'#76e2c1':'#f3bf77'; c.font='bold 28px system-ui'; c.fillText(control,24,46);
    c.fillStyle=recording?'#ff8888':'#edf5f4'; c.font='bold 35px system-ui'; c.fillText(summary,24,101);
    c.fillStyle='#c4dcf2'; c.font='25px system-ui'; c.fillText(arms,24,148);
    c.fillStyle='#b9cecf'; c.font='24px system-ui'; wrapped(c,hint,24,195,976,32);
    for(const button of this.buttons) {
      c.fillStyle=!button.enabled?'#25333e':button.action===this.hovered?'#c0f4e7':button.action==='discard'?'#f1b38f':'#76e2c1';
      c.fillRect(button.x,button.y,button.w,button.h);
      c.fillStyle=button.enabled?'#10202b':'#78909b'; c.font='bold 27px system-ui'; c.textAlign='center';
      c.fillText(button.label,button.x+button.w/2,button.y+44);
    }
    this.hud.texture.needsUpdate=true;
  }

  hit(ray) {
    const hit=ray.intersectObjects([this.hud.mesh,this.workspacePanel.mesh])[0];
    if(!hit) return null;
    const workspace=hit.object===this.workspacePanel.mesh;
    const x=hit.uv.x*1024,y=(1-hit.uv.y)*(workspace?184:384);
    const button=(workspace?this.workspaceButtons:this.buttons).find(b=>b.enabled && x>=b.x && x<=b.x+b.w && y>=b.y && y<=b.y+b.h);
    return button?{action:button.action,point:hit.point}:null;
  }

  nearWorkspace(point) {
    const start=new THREE.Vector3(0,-RAIL_LENGTH/2,0).applyMatrix4(this.rail.matrixWorld);
    const end=new THREE.Vector3(0,RAIL_LENGTH/2,0).applyMatrix4(this.rail.matrixWorld);
    const nearest=new THREE.Line3(start,end).closestPointToPoint(point,true,new THREE.Vector3());
    const scale=this.rail.matrixWorld.getMaxScaleOnAxis();
    return nearest.distanceTo(point)<RAIL_GRAB_RADIUS*Math.min(1,scale);
  }

  workspaceAnchor() { return this.rail.getWorldPosition(new THREE.Vector3()).toArray(); }

  updateWorkspace(scale,active,immersive) {
    this.workspaceButtons=[['actual_size','Actual size'],['place_ahead','Place ahead']].map(([action,label],i)=>({
      action,label,x:24+i*500,y:100,w:476,h:64,enabled:immersive&&!active,
    }));
    const signature=JSON.stringify([Math.round(scale*100),active,immersive,this.hovered]);
    if(signature===this.workspaceDrawn) return; this.workspaceDrawn=signature;
    this.rail.material.color.set(active?0x76e2c1:0xb9cecf);
    const c=this.workspacePanel.context;
    c.fillStyle='#13232d'; c.fillRect(0,0,1024,184); c.textAlign='left';
    c.fillStyle='#edf5f4'; c.font='bold 29px system-ui';
    c.fillText(`WORKSPACE · ${Math.round(scale*100)}%${active?' · ARMS HELD':''}`,24,38);
    c.fillStyle='#b9cecf'; c.font='24px system-ui';
    c.fillText('Pinch rail to move · Two hands to resize and turn',24,77);
    for(const button of this.workspaceButtons) {
      c.fillStyle=!button.enabled?'#25333e':this.hovered===button.action?'#c0f4e7':'#76e2c1';
      c.fillRect(button.x,button.y,button.w,button.h);
      c.fillStyle=button.enabled?'#10202b':'#78909b'; c.font='bold 27px system-ui'; c.textAlign='center';
      c.fillText(button.label,button.x+button.w/2,button.y+42);
    }
    this.workspacePanel.texture.needsUpdate=true;
  }

  dispose() {
    this.stopped=true; this.abort.abort();
    for(const view of [this.hud,this.workspacePanel,...this.cameras.values()]) {
      view.texture.dispose(); view.mesh.geometry.dispose(); view.mesh.material.dispose(); view.mesh.removeFromParent();
      if(view.url) URL.revokeObjectURL(view.url);
    }
    this.rail.traverse(mesh=>mesh.geometry.dispose()); this.rail.material.dispose(); this.rail.removeFromParent();
    for(const button of document.querySelectorAll('[data-action]')) button.onclick=null;
  }
}
