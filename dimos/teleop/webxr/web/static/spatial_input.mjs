// Copyright 2026 Dimensional Inc. Licensed under the Apache License, Version 2.0.
// Hysteresis and release-before-grab survive tracking loss without a surprise resume.
export class PinchGrab {
  constructor() { this.reset(); }
  reset() { this.armed = false; this.pinched = false; this.grabbed = false; }
  update(distance, nearHandle) {
    const before = this.pinched;
    if (!Number.isFinite(distance)) { this.reset(); return {grab: false, pressed: false}; }
    if (distance > 0.035) { this.pinched = false; this.armed = true; this.grabbed = false; }
    if (distance < 0.022 && this.armed && !this.pinched) {
      this.pinched = true; this.armed = false; this.grabbed = nearHandle;
    }
    return {grab: this.grabbed, pressed: this.pinched && !before};
  }
}
