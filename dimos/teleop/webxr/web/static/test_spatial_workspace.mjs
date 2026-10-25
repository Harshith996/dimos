// Copyright 2026 Dimensional Inc. Licensed under the Apache License, Version 2.0.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {WorkspaceTransform} from './spatial_workspace.mjs';

const hand = (position, pinched = true, nearRail = true) => ({position, distance:pinched ? 0.01 : 0.05, nearRail});
const close = (a, b) => {
  assert.equal(a.length,b.length);
  a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-10,`${a} != ${b}`));
};
const pose = w => [...w.position,w.yaw,w.scale];
function both(w,left=[-0.5,1,-1],right=[0.5,1,-1]) {
  w.update({left:hand(left,false),right:hand(right,false)});
  w.update({left:hand(left),right:hand(right)});
}

test('one-hand rail grab translates without rotating or resizing and freezes on release',()=>{
  const w=new WorkspaceTransform({position:[1,2,3],yaw:0.7,scale:0.8});
  w.update({left:hand([0,1,-1],false)});
  assert.equal(w.update({left:hand([0,1,-1])}),true);
  close(pose(w),[1,2,3,0.7,0.8]);
  w.update({left:hand([0.3,1.2,-1.4],true,false)});
  close(pose(w),[1.3,2.2,2.6,0.7,0.8]);
  assert.equal(w.update({left:hand([0.3,1.2,-1.4],false)}),true);
  assert.equal(w.active,false);
  assert.equal(w.update({left:hand([3,4,5],false)}),false);
  close(pose(w),[1.3,2.2,2.6,0.7,0.8]);
});

test('two hands rotate and scale about their moving midpoint, keeping the world upright',()=>{
  const w=new WorkspaceTransform({position:[0,0,-1]});
  both(w);
  // Double separation, turn 90 degrees about XR up, and move midpoint by (1,1,1).
  w.update({left:hand([1,2,1]),right:hand([1,2,-1])});
  close(pose(w),[1,0,0,Math.PI/2,2]);
});

test('joining and releasing either hand reanchors without a jump',()=>{
  for(const released of ['left','right']) {
    const w=new WorkspaceTransform();
    w.update({left:hand([-.3,1,-1],false),right:hand([.3,1,-1],false)});
    w.update({left:hand([-.3,1,-1]),right:hand([.3,1,-1],false)});
    w.update({left:hand([-.2,1.1,-1.1]),right:hand([.3,1,-1],false)});
    const beforeJoin=pose(w);
    w.update({left:hand([-.2,1.1,-1.1]),right:hand([.3,1,-1])});
    close(pose(w),beforeJoin);
    w.update({left:hand([-.4,1,-1]),right:hand([.5,1,-1.2])});
    const beforeRelease=pose(w);
    const remaining=released==='left'?'right':'left';
    const positions={left:[-.4,1,-1],right:[.5,1,-1.2]};
    w.update({[remaining]:hand(positions[remaining]),[released]:hand(positions[released],false)});
    close(pose(w),beforeRelease);
    w.update({[remaining]:hand(positions[remaining].map(v=>v+.1))});
    close(w.position,beforeRelease.slice(0,3).map(v=>v+.1));
    close([w.yaw,w.scale],beforeRelease.slice(3));
  }
});

test('a pinch begun elsewhere cannot become a workspace grab until released',()=>{
  const w=new WorkspaceTransform();
  w.update({left:hand([0,1,-1],false,false)});
  assert.equal(w.update({left:hand([0,1,-1],true,false)}),false);
  assert.equal(w.update({left:hand([0,1,-1])}),false);
  w.update({left:hand([0,1,-1],false)});
  assert.equal(w.update({left:hand([0,1,-1])}),true);
});

test('tracking loss retains placement and cannot resume a held pinch',()=>{
  const w=new WorkspaceTransform();
  both(w);
  w.update({left:hand([-.6,1,-1]),right:hand([.6,1,-1])});
  const before=pose(w);
  assert.equal(w.update({}),true);
  assert.equal(w.update({left:hand([2,2,2])}),false);
  close(pose(w),before);
  w.update({left:hand([2,2,2],false)});
  assert.equal(w.update({left:hand([2,2,2])}),true);
  close(pose(w),before);
  w.cancel();
  assert.equal(w.update({left:hand([3,2,2])}),false);
});

test('scale is bounded and coincident or vertical hands cannot cause a rotation jump',()=>{
  const w=new WorkspaceTransform();
  both(w);
  w.update({left:hand([-5,1,-1]),right:hand([5,1,-1])});
  assert.equal(w.scale,2);
  w.update({left:hand([-.05,1,-1]),right:hand([.05,1,-1])});
  assert.equal(w.scale,.25);
  w.update({left:hand([0,1,-1]),right:hand([0,1,-1])});
  const before=pose(w);
  w.update({left:hand([0,.5,-1]),right:hand([0,1.5,-1])});
  close(pose(w),before);
  w.update({left:hand([-.3,1,-1]),right:hand([.3,1,-1])});
  close(pose(w),before);
  w.update({left:hand([-.6,1,-1]),right:hand([.6,1,-1])});
  assert.equal(w.scale,.5);
});

test('actual size keeps its anchor fixed and placement preserves chosen scale',()=>{
  const w=new WorkspaceTransform({position:[0,0,-1],yaw:.5,scale:2});
  w.resize(1,[1,1,-1]);
  close(pose(w),[.5,.5,-1,.5,1]);
  w.resize(.5,[1,1,-1]);
  w.place([4,5,6],1.2);
  close(pose(w),[4,5,6,1.2,.5]);
  assert.equal(w.active,false);
});

test('invalid tracking samples cannot corrupt placement',()=>{
  const w=new WorkspaceTransform();
  both(w);
  const before=pose(w);
  w.update({left:hand([NaN,0,0]),right:{...hand([1,1,1]),distance:NaN}});
  close(pose(w),before);
  assert.equal(w.active,false);
});
