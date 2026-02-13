// frontend/src/lib/features.js

export const IDX = {
  upper_lip: 13,
  lower_lip: 14,
  mouth_left: 61,
  mouth_right: 291,
  left_eye_top: 159,
  left_eye_bottom: 145,
  right_eye_top: 386,
  right_eye_bottom: 374,
  left_eye_outer: 33,
  right_eye_outer: 263,
  left_brow: 105,
  right_brow: 334,
  forehead: 10,
  chin: 152,
  nose_tip: 1,
};

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function norm3(a) {
  return Math.sqrt(dot3(a, a));
}
function unit3(a) {
  const n = norm3(a) || 1e-9;
  return [a[0] / n, a[1] / n, a[2] / n];
}
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function dist2(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function extractFeatures(landmarks) {
  const L = (i) => landmarks[i];

  // Normalize by face height (2D, stable)
  const faceH = dist2(L(IDX.forehead), L(IDX.chin)) || 1e-6;

  const mouth_open = dist2(L(IDX.upper_lip), L(IDX.lower_lip)) / faceH;
  const mouth_w = dist2(L(IDX.mouth_left), L(IDX.mouth_right)) / faceH;

  const eye_open =
    ((dist2(L(IDX.left_eye_top), L(IDX.left_eye_bottom)) +
      dist2(L(IDX.right_eye_top), L(IDX.right_eye_bottom))) /
      2) /
    faceH;

  // 3D face frame for smile_up / brow_raise (pose robust)
  const le = L(IDX.left_eye_outer);
  const re = L(IDX.right_eye_outer);
  const nose = L(IDX.nose_tip);

  const origin = [(le.x + re.x) / 2, (le.y + re.y) / 2, (le.z + re.z) / 2];

  const x_axis = unit3([re.x - le.x, re.y - le.y, re.z - le.z]);
  const forward_hint = unit3([nose.x - origin[0], nose.y - origin[1], nose.z - origin[2]]);

  // Gram-Schmidt to get y_axis orthogonal to x_axis
  let y_axis = [
    forward_hint[0] - dot3(forward_hint, x_axis) * x_axis[0],
    forward_hint[1] - dot3(forward_hint, x_axis) * x_axis[1],
    forward_hint[2] - dot3(forward_hint, x_axis) * x_axis[2],
  ];
  y_axis = unit3(y_axis);

  // Not strictly needed but keeps the frame coherent
  const _z_axis = unit3(cross(x_axis, y_axis));

  // smile_up: corners vs mouth center along face y_axis
  const upper = L(IDX.upper_lip);
  const lower = L(IDX.lower_lip);
  const ml = L(IDX.mouth_left);
  const mr = L(IDX.mouth_right);

  const mouth_center = {
    x: (upper.x + lower.x) / 2,
    y: (upper.y + lower.y) / 2,
    z: (upper.z + lower.z) / 2,
  };
  const corners_avg = {
    x: (ml.x + mr.x) / 2,
    y: (ml.y + mr.y) / 2,
    z: (ml.z + mr.z) / 2,
  };

  const v_smile = [corners_avg.x - mouth_center.x, corners_avg.y - mouth_center.y, corners_avg.z - mouth_center.z];
  const smile_up = -dot3(v_smile, y_axis);

  // brow_raise: brow vs eye-top along face y_axis
  const lb = L(IDX.left_brow);
  const rb = L(IDX.right_brow);
  const leTop = L(IDX.left_eye_top);
  const reTop = L(IDX.right_eye_top);

  const vL = [lb.x - leTop.x, lb.y - leTop.y, lb.z - leTop.z];
  const vR = [rb.x - reTop.x, rb.y - reTop.y, rb.z - reTop.z];
  const brow_raise = -((dot3(vL, y_axis) + dot3(vR, y_axis)) / 2);

  return { mouth_open, eye_open, mouth_w, smile_up, brow_raise };
}
