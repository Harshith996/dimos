// Copyright 2026 Dimensional Inc. Licensed under the Apache License, Version 2.0.
import {PinchGrab} from './spatial_input.mjs';

const SIDES = ['left', 'right'];
const MIN_SPAN = 0.08; // XR metres; avoid unstable rotation when hands meet or stack vertically.
const add = (a, b) => a.map((v, i) => v + b[i]);
const subtract = (a, b) => a.map((v, i) => v - b[i]);
const multiply = (a, s) => a.map(v => v * s);
const rotate = (p, yaw) => [
  Math.cos(yaw) * p[0] + Math.sin(yaw) * p[2], p[1],
  -Math.sin(yaw) * p[0] + Math.cos(yaw) * p[2],
];

// Viewer placement in XR coordinates. No robot state, scene truth or command transport.
export class WorkspaceTransform {
  constructor({position = [0, 0.8, -0.15], yaw = 0, scale = 1, minScale = 0.25, maxScale = 2} = {}) {
    this.position = [...position]; this.yaw = yaw; this.scale = scale;
    this.minScale = minScale; this.maxScale = maxScale;
    this.pinches = Object.fromEntries(SIDES.map(side => [side, new PinchGrab()]));
    this.reference = null;
  }

  get active() { return this.reference !== null; }

  cancel() {
    this.reference = null;
    for (const pinch of Object.values(this.pinches)) pinch.reset();
  }

  place(position, yaw) {
    this.cancel(); this.position = [...position]; this.yaw = yaw;
  }

  resize(scale, anchor) {
    this.cancel();
    const next = Math.max(this.minScale, Math.min(this.maxScale, scale));
    this.position = add(anchor, multiply(subtract(this.position, anchor), next / this.scale));
    this.scale = next;
  }

  // True means this frame belongs to arrangement, including the final release frame.
  update(hands) {
    const wasActive = this.active;
    const held = [];
    for (const side of SIDES) {
      const hand = hands[side], pinch = this.pinches[side];
      if (!hand || hand.position.length !== 3 || !hand.position.every(Number.isFinite)) {
        pinch.reset(); continue;
      }
      if (pinch.update(hand.distance, hand.nearRail).grab) held.push([side, hand.position]);
    }
    if (!held.length) { this.reference = null; return wasActive; }
    const key = held.map(([side]) => side).join(',');
    const center = multiply(held.reduce((sum, [, p]) => add(sum, p), [0, 0, 0]), 1 / held.length);
    const vector = held.length === 2 ? subtract(held[1][1], held[0][1]) : null;
    const stable = vector !== null && Math.hypot(vector[0], vector[2]) >= MIN_SPAN;
    const span = vector ? Math.hypot(...vector) : 0;
    const angle = vector ? Math.atan2(-vector[2], vector[0]) : 0;
    const capture = () => ({key, center, span, angle, stable,
      position: [...this.position], yaw: this.yaw, scale: this.scale});
    if (this.reference?.key !== key) {
      this.reference = capture();
      return true;
    }
    const start = this.reference;
    if (held.length === 1 || !stable || !start.stable) {
      this.position = add(start.position, subtract(center, start.center));
      // Degenerate two-hand poses retain rotation/scale and establish a fresh baseline.
      if (held.length === 2) this.reference = capture();
    } else {
      const yaw = Math.atan2(Math.sin(angle - start.angle), Math.cos(angle - start.angle));
      this.scale = Math.max(this.minScale, Math.min(this.maxScale, start.scale * span / start.span));
      this.yaw = start.yaw + yaw;
      this.position = add(center, multiply(rotate(subtract(start.position, start.center), yaw), this.scale / start.scale));
    }
    return true;
  }
}
