// Copyright 2026 Dimensional Inc. Licensed under the Apache License, Version 2.0.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {interactionMode, sampleXRInputs} from './xr_input.mjs';
import {PinchGrab} from './spatial_input.mjs';

const pose = x => ({transform:{position:{x,y:1,z:-1},orientation:{x:0,y:0,z:0,w:1}}});
const controller = (side, trigger=0, grip=0) => ({handedness:side,targetRayMode:'tracked-pointer',
  gripSpace:pose(0.3),targetRaySpace:pose(0.3),gamepad:{mapping:'xr-standard',buttons:[{value:trigger},{value:grip}]}});
const frame = (...sources) => ({session:{inputSources:sources},getPose:space=>space,
  getJointPose:joint=>joint});

test('controllers without hands supply independent grip, trigger and tracked poses', () => {
  const samples=sampleXRInputs(frame(controller('left',.8,1),controller('right',0,0)),{});
  assert.equal(interactionMode('auto',samples),'clutch');
  assert.equal(samples.left.kind,'controller');
  assert.equal(samples.left.trigger,.8);
  assert.ok(samples.left.clutchDistance<.022);
  assert.ok(samples.right.clutchDistance>.035);
  assert.deepEqual(samples.left.position,[.3,1,-1]);
  assert.deepEqual(samples.left.ray.orientation,[0,0,0,1]);
  assert.equal(interactionMode('tools',samples),'tools');
});

test('tracked hands remain usable without a gamepad; gaze pointers never become hands', () => {
  const source={handedness:'left',targetRayMode:'tracked-pointer',targetRaySpace:pose(0),
    hand:new Map([['wrist',pose(0)],['thumb-tip',pose(0)],['index-finger-tip',pose(.02)]])};
  const samples=sampleXRInputs(frame(source),{});
  assert.equal(interactionMode('auto',samples),'tools');
  assert.equal(samples.left.distance,.02);
  assert.deepEqual(samples.left.position,[.01,1,-1]);
  assert.deepEqual(sampleXRInputs(frame({...source,targetRayMode:'transient-pointer'}),{}),{});
  source.hand.set('wrist',null);
  assert.deepEqual(sampleXRInputs(frame(source),{}),{});
});

test('missing controller tracking and missing squeeze cannot produce a clutch grab', () => {
  assert.deepEqual(sampleXRInputs(frame({...controller('left'),gripSpace:null}),{}),{});
  const source=controller('left',1,1); source.gamepad.buttons.pop();
  const sample=sampleXRInputs(frame(source),{}).left;
  assert.equal(interactionMode('auto',{left:sample}),'tools');
  const clutch=new PinchGrab(); clutch.update(.05,true);
  assert.equal(clutch.update(sample.clutchDistance,true).grab,false);
});

test('a controller arriving already squeezed must release before moving', () => {
  const source=controller('right',.6,1), clutch=new PinchGrab();
  const update=()=>clutch.update(sampleXRInputs(frame(source),{}).right.clutchDistance,true).grab;
  assert.equal(update(),false);
  source.gamepad.buttons[1].value=0; assert.equal(update(),false);
  source.gamepad.buttons[1].value=1; assert.equal(update(),true);
  clutch.reset(); assert.equal(update(),false);
});
