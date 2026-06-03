// Body attitude helpers for replay markers (vendor Sabatino / matrixB2I convention).
// Local frame: x = east, y = north, z = up (meters).

export const ATT_ARROW_SCALE_M_PER_RAD = 3;
export const ATT_ARROW_MIN_M = 0.3;

export const ATT_AXIS_COLORS = {
  roll: "#ff453a",
  pitch: "#30d158",
  yaw: "#bf5af2",
};

/** Arrow length (m) from Euler angle magnitude (rad). */
export function axisLength(angleRad, scale = ATT_ARROW_SCALE_M_PER_RAD, minLen = ATT_ARROW_MIN_M) {
  const a = Math.abs(Number(angleRad) || 0);
  return Math.max(minLen, scale * a);
}

/**
 * Body axis unit vectors in local ENU. R = Rz(yaw) * Ry(pitch) * Rx(roll);
 * columns are body +X, +Y, +Z in world/local frame.
 */
export function bodyAxesLocal(roll, pitch, yaw) {
  const cr = Math.cos(roll);
  const sr = Math.sin(roll);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);

  // R = Rz * Ry * Rx
  const r00 = cy * cp;
  const r01 = cy * sp * sr - sy * cr;
  const r02 = cy * sp * cr + sy * sr;
  const r10 = sy * cp;
  const r11 = sy * sp * sr + cy * cr;
  const r12 = sy * sp * cr - cy * sr;
  const r20 = -sp;
  const r21 = cp * sr;
  const r22 = cp * cr;

  return {
    x: [r00, r10, r20],
    y: [r01, r11, r21],
    z: [r02, r12, r22],
  };
}

/** Local ENU [e, n, u] -> Three.js world (x=east, y=up, z=-north). */
export function localToThree([ex, ey, ez]) {
  return [ex, ez, -ey];
}

/** Body axis arrows: { key, dir, angle, color, length }[] for roll/pitch/yaw. */
export function bodyAxisArrows(roll, pitch, yaw) {
  const axes = bodyAxesLocal(roll, pitch, yaw);
  return [
    { key: "roll", dir: axes.x, angle: roll, color: ATT_AXIS_COLORS.roll },
    { key: "pitch", dir: axes.y, angle: pitch, color: ATT_AXIS_COLORS.pitch },
    { key: "yaw", dir: axes.z, angle: yaw, color: ATT_AXIS_COLORS.yaw },
  ].map((a) => ({ ...a, length: axisLength(a.angle) }));
}

export const VEC_AXIS_COLORS = {
  vx: "#ff453a",
  vy: "#30d158",
  vz: "#64d2ff",
};

export const VEL_ARROW_SCALE_M_PER_MPS = 0.5;
export const ACC_ARROW_SCALE_M_PER_MPS2 = 0.1;

/** ENU component arrows for velocity or acceleration. */
export function vectorArrows(vx, vy, vz, { scale, minLen = ATT_ARROW_MIN_M, colors = VEC_AXIS_COLORS } = {}) {
  const comps = [
    { key: "vx", v: vx, dir: [1, 0, 0] },
    { key: "vy", v: vy, dir: [0, 1, 0] },
    { key: "vz", v: vz, dir: [0, 0, 1] },
  ];
  return comps.map(({ key, v, dir }) => ({
    key,
    dir,
    length: Math.max(minLen, scale * Math.abs(Number(v) || 0)),
    color: colors[key],
  }));
}

/** Replay overlay arrows for the selected axis mode. */
export function replayAxisArrows(mode, { attitude, velocity, acceleration } = {}) {
  if (!mode) return [];
  if (mode === "velocity" && velocity && velocity.length >= 3) {
    const [vx, vy, vz] = velocity;
    return vectorArrows(vx, vy, vz, { scale: VEL_ARROW_SCALE_M_PER_MPS });
  }
  if (mode === "acceleration" && acceleration && acceleration.length >= 3) {
    const [ax, ay, az] = acceleration;
    return vectorArrows(ax, ay, az, { scale: ACC_ARROW_SCALE_M_PER_MPS2 });
  }
  if (mode === "attitude" && attitude && attitude.length >= 3) {
    const [roll, pitch, yaw] = attitude;
    return bodyAxisArrows(roll, pitch, yaw);
  }
  return [];
}
