// Copyright 2026 Dimensional Inc. Licensed under the Apache License, Version 2.0.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PinchGrab} from './spatial_input.mjs';
test('release, grab, hold with hysteresis, release', () => {
  const hand = new PinchGrab();
  assert.equal(hand.update(.01, true).grab, false);
  hand.update(.05, true);
  assert.deepEqual(hand.update(.01, true), {grab:true, pressed:true});
  assert.equal(hand.update(.028, false).grab, true);
  assert.equal(hand.update(.05, false).grab, false);
});
test('lost tracking and pinch away from a handle cannot activate an arm', () => {
  const hand = new PinchGrab();
  hand.update(.05, true); hand.update(.01, true);
  hand.update(NaN, true);
  assert.equal(hand.update(.01, true).grab, false);
  hand.update(.05, false);
  assert.deepEqual(hand.update(.01, false), {grab:false, pressed:true});
  assert.equal(hand.update(.01, true).grab, false);
});
