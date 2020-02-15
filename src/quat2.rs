//! Dual Quaternion
//! 
//! Format: ```[real, dual]```
//! 
//! Quaternion format: ```XYZW```
//! 
//! Make sure to have normalized dual quaternions, 
//! otherwise the functions may not work as intended.

use super::common::{Quat2, Quat, Vec3, Mat4, hypot, EPSILON};
use super::{quat, mat4};

/// Creates a new identity dual quat
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat2.html)
pub fn create() -> Quat2 {
    let mut dq: Quat2 = [0_f32; 8];

    dq[3] = 1.;

    dq
}

pub fn clone(a: &Quat2) -> Quat2 {
    let mut dq: Quat2 = [0_f32; 8];

    dq[0] = a[0];
    dq[1] = a[1];
    dq[2] = a[2];
    dq[3] = a[3];
    dq[4] = a[4];
    dq[5] = a[5];
    dq[6] = a[6];
    dq[7] = a[7];

    dq
}

pub fn from_values(x1: f32, y1: f32, z1: f32, w1: f32,
                   x2: f32, y2: f32, z2: f32, w2: f32) -> Quat2 {
    let mut dq: Quat2 = [0_f32; 8];

    dq[0] = x1;
    dq[1] = y1;
    dq[2] = z1;
    dq[3] = w1;
    dq[4] = x2;
    dq[5] = y2;
    dq[6] = z2;
    dq[7] = w2;

    dq
}

pub fn from_rotation_translation_values(x1: f32, y1: f32, z1: f32, w1: f32,
                                        x2: f32, y2: f32, z2: f32) -> Quat2 {
    let mut dq: Quat2 = [0_f32; 8];

    dq[0] = x1;
    dq[1] = y1;
    dq[2] = z1;
    dq[3] = w1;

    let ax = x2 * 0.5;
    let ay = y2 * 0.5;
    let az = z2 * 0.5;

    dq[4] = ax * w1 + ay * z1 - az * y1;
    dq[5] = ay * w1 + az * x1 - ax * z1;
    dq[6] = az * w1 + ax * y1 - ay * x1;
    dq[7] = -ax * x1 - ay * y1 - az * z1;

    dq
}

pub fn from_rotation_translation(out: &mut Quat2, q: &Quat, t: &Vec3) {
    let ax = t[0] * 0.5;
    let ay = t[1] * 0.5;
    let az = t[2] * 0.5;
    let bx = q[0];
    let by = q[1];
    let bz = q[2];
    let bw = q[3];

    out[0] = bx;
    out[1] = by;
    out[2] = bz;
    out[3] = bw;
    out[4] = ax * bw + ay * bz - az * by;
    out[5] = ay * bw + az * bx - ax * bz;
    out[6] = az * bw + ax * by - ay * bx;
    out[7] = -ax * bx - ay * by - az * bz;
}

pub fn from_translation(out: &mut Quat2, t: &Vec3) {
    out[0] = 0.;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 1.;
    out[4] = t[0] * 0.5;
    out[5] = t[1] * 0.5;
    out[6] = t[2] * 0.5;
    out[7] = 0.;
}

pub fn from_rotation(out: &mut Quat2, q: &Quat) {
    out[0] = q[0];
    out[1] = q[1];
    out[2] = q[2];
    out[3] = q[3];
    out[4] = 0.;
    out[5] = 0.;
    out[6] = 0.;
    out[7] = 0.;
}

pub fn from_mat4(out: &mut Quat2, a: &Mat4) {
    //TODO Optimize this
    let outer = &mut quat::create();

    mat4::get_rotation(outer, a);

    let t = &mut [0f32; 3];

    mat4::get_translation(t, a);

    from_rotation_translation(out, outer, t);
}

pub fn copy(out: &mut Quat2, a: &Quat2) -> Quat2 {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
    out[6] = a[6];
    out[7] = a[7];

    *out
}

pub fn identity(out: &mut Quat2) -> Quat2 {
    out[0] = 0.;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 1.;
    out[4] = 0.;
    out[5] = 0.;
    out[6] = 0.;
    out[7] = 0.;

    *out
}

pub fn set(out: &mut Quat2, x1: f32, y1: f32, z1: f32, w1: f32,
                            x2: f32, y2: f32, z2: f32, w2: f32) -> Quat2 {
    out[0] = x1;
    out[1] = y1;
    out[2] = z1;
    out[3] = w1;
    out[4] = x2;
    out[5] = y2;
    out[6] = z2;
    out[7] = w2;

    *out
}

pub fn get_real(q: &mut Quat, dq: &Quat2) {
    let dq_real = [dq[0], dq[1], dq[2], dq[3]];
    quat::copy(q, &dq_real);
}

pub fn get_dual(out: &mut Quat, a: &Quat2) {
    out[0] = a[4];
    out[1] = a[5];
    out[2] = a[6];
    out[3] = a[7];
}

pub fn set_real(dq: &mut Quat2, q: &Quat) {
    dq[0] = q[0];
    dq[1] = q[1];
    dq[2] = q[2];
    dq[3] = q[3];
}

pub fn set_dual(out: &mut Quat2, q: &Quat) {
    out[4] = q[0];
    out[5] = q[1];
    out[6] = q[2];
    out[7] = q[3];
}

pub fn get_translation(out: &mut Vec3, a: &Quat2) {
    let ax = a[4];
    let ay = a[5];
    let az = a[6];
    let aw = a[7];

    let bx = -a[0];
    let by = -a[1];
    let bz = -a[2];
    let bw = a[3];

    out[0] = (ax * bw + aw * bx + ay * bz - az * by) * 2.;
    out[1] = (ay * bw + aw * by + az * bx - ax * bz) * 2.;
    out[2] = (az * bw + aw * bz + ax * by - ay * bx) * 2.;
}

pub fn translate(out: &mut Quat2, a: &Quat2, v: &Vec3) {
    let ax1 = a[0];
    let ay1 = a[1];
    let az1 = a[2];
    let aw1 = a[3];
    
    let bx1 = v[0] * 0.5;
    let by1 = v[1] * 0.5;
    let bz1 = v[2] * 0.5;

    let ax2 = a[4];
    let ay2 = a[5];
    let az2 = a[6];
    let aw2 = a[7];

    out[0] = ax1;
    out[1] = ay1;
    out[2] = az1;
    out[3] = aw1;
    out[4] = aw1 * bx1 + ay1 * bz1 - az1 * by1 + ax2;
    out[5] = aw1 * by1 + az1 * bx1 - ax1 * bz1 + ay2;
    out[6] = aw1 * bz1 + ax1 * by1 - ay1 * bx1 + az2;
    out[7] = -ax1 * bx1 - ay1 * by1 - az1 * bz1 + aw2;
}

pub fn rotate_x(out: &mut Quat2, a: &Quat2, rad: f32) {
    let mut bx = -a[0];
    let mut by = -a[1];
    let mut bz = -a[2];
    let mut bw = a[3];

    let ax = a[4];
    let ay = a[5];
    let az = a[6];
    let aw = a[7];

    let ax1 = ax * bw + aw * bx + ay * bz - az * by;
    let ay1 = ay * bw + aw * by + az * bx - ax * bz;
    let az1 = az * bw + aw * bz + ax * by - ay * bx;
    let aw1 = aw * bw - ax * bx - ay * by - az * bz;

    let out_real = &mut [out[1], out[2], out[3], out[4]];
    let a_real = &[a[1], a[2], a[3], a[4]];
    quat::rotate_x(out_real, a_real, rad);

    bx = out[0];
    by = out[1];
    bz = out[2];
    bw = out[3];

    out[4] = ax1 * bw + aw1 * bx + ay1 * bz - az1 * by;
    out[5] = ay1 * bw + aw1 * by + az1 * bx - ax1 * bz;
    out[6] = az1 * bw + aw1 * bz + ax1 * by - ay1 * bx;
    out[7] = aw1 * bw - ax1 * bx - ay1 * by - az1 * bz;
}

pub fn rotate_y(out: &mut Quat2, a: &Quat2, rad: f32) {
    let mut bx = -a[0];
    let mut by = -a[1];
    let mut bz = -a[2];
    let mut bw = a[3];
    
    let ax = a[4];
    let ay = a[5];
    let az = a[6];
    let aw = a[7];

    let ax1 = ax * bw + aw * bx + ay * bz - az * by;
    let ay1 = ay * bw + aw * by + az * bx - ax * bz;
    let az1 = az * bw + aw * bz + ax * by - ay * bx;
    let aw1 = aw * bw - ax * bx - ay * by - az * bz;

    let out_real = &mut [out[1], out[2], out[3], out[4]];
    let a_real = &[a[1], a[2], a[3], a[4]];
    quat::rotate_y(out_real, a_real, rad);

    bx = out[0];
    by = out[1];
    bz = out[2];
    bw = out[3];

    out[4] = ax1 * bw + aw1 * bx + ay1 * bz - az1 * by;
    out[5] = ay1 * bw + aw1 * by + az1 * bx - ax1 * bz;
    out[6] = az1 * bw + aw1 * bz + ax1 * by - ay1 * bx;
    out[7] = aw1 * bw - ax1 * bx - ay1 * by - az1 * bz;
}


pub fn rotate_z(out: &mut Quat2, a: &Quat2, rad: f32) -> Quat2 {
    let mut bx = -a[0];
    let mut by = -a[1];
    let mut bz = -a[2];
    let mut bw = a[3];
   
    let ax = a[4];
    let ay = a[5];
    let az = a[6];
    let aw = a[7];

    let ax1 = ax * bw + aw * bx + ay * bz - az * by;
    let ay1 = ay * bw + aw * by + az * bx - ax * bz;
    let az1 = az * bw + aw * bz + ax * by - ay * bx;
    let aw1 = aw * bw - ax * bx - ay * by - az * bz;

    let out_real = &mut [out[1], out[2], out[3], out[4]];
    let a_real = &[a[1], a[2], a[3], a[4]];
    quat::rotate_z(out_real, a_real, rad);

    bx = out[0];
    by = out[1];
    bz = out[2];
    bw = out[3];
    
    out[4] = ax1 * bw + aw1 * bx + ay1 * bz - az1 * by;
    out[5] = ay1 * bw + aw1 * by + az1 * bx - ax1 * bz;
    out[6] = az1 * bw + aw1 * bz + ax1 * by - ay1 * bx;
    out[7] = aw1 * bw - ax1 * bx - ay1 * by - az1 * bz;

    *out
}

pub fn rotate_by_quat_append(out: &mut Quat2, a: &Quat2, q: Quat) -> Quat2 {
    let qx = q[0];
    let qy = q[1];
    let qz = q[2];
    let qw = q[3];
    
    let mut ax = a[0];
    let mut ay = a[1];
    let mut az = a[2];
    let mut aw = a[3];

    out[0] = ax * qw + aw * qx + ay * qz - az * qy;
    out[1] = ay * qw + aw * qy + az * qx - ax * qz;
    out[2] = az * qw + aw * qz + ax * qy - ay * qx;
    out[3] = aw * qw - ax * qx - ay * qy - az * qz;

    ax = a[4];
    ay = a[5];
    az = a[6];
    aw = a[7];

    out[4] = ax * qw + aw * qx + ay * qz - az * qy;
    out[5] = ay * qw + aw * qy + az * qx - ax * qz;
    out[6] = az * qw + aw * qz + ax * qy - ay * qx;
    out[7] = aw * qw - ax * qx - ay * qy - az * qz;

    *out 
}

pub fn rotate_by_quat_prepend(out: &mut Quat2, q: &Quat, a: &Quat2) -> Quat2 {
    let qx = q[0];
    let qy = q[1];
    let qz = q[2];
    let qw = q[3];
    
    let mut bx = a[0];
    let mut by = a[1];
    let mut bz = a[2];
    let mut bw = a[3];

    out[0] = qx * bw + qw * bx + qy * bz - qz * by;
    out[1] = qy * bw + qw * by + qz * bx - qx * bz;
    out[2] = qz * bw + qw * bz + qx * by - qy * bx;
    out[3] = qw * bw - qx * bx - qy * by - qz * bz;

    bx = a[4];
    by = a[5];
    bz = a[6];
    bw = a[7];

    out[4] = qx * bw + qw * bx + qy * bz - qz * by;
    out[5] = qy * bw + qw * by + qz * bx - qx * bz;
    out[6] = qz * bw + qw * bz + qx * by - qy * bx;
    out[7] = qw * bw - qx * bx - qy * by - qz * bz;

    *out 
}

pub fn rotate_around_axis(out: &mut Quat2, a: &Quat2, axis: &Vec3, rad: f32) -> Quat2 {
    //Special case for rad = 0
    if f32::abs(rad) < EPSILON {
        return copy(out, a);
    }

    let axis_length = hypot(axis);
    let rad = rad * 0.5;

    let s = f32::sin(rad);
    let bx = s * axis[0] / axis_length;
    let by = s * axis[1] / axis_length;
    let bz = s * axis[2] / axis_length;
    let bw = f32::cos(rad);
    
    let ax1 = a[0];
    let ay1 = a[1];
    let az1 = a[2];
    let aw1 = a[3];

    out[0] = ax1 * bw + aw1 * bx + ay1 * bz - az1 * by;
    out[1] = ay1 * bw + aw1 * by + az1 * bx - ax1 * bz;
    out[2] = az1 * bw + aw1 * bz + ax1 * by - ay1 * bx;
    out[3] = aw1 * bw - ax1 * bx - ay1 * by - az1 * bz;

    let ax = a[4];
    let ay = a[5];
    let az = a[6];
    let aw = a[7];

    out[4] = ax * bw + aw * bx + ay * bz - az * by;
    out[5] = ay * bw + aw * by + az * bx - ax * bz;
    out[6] = az * bw + aw * bz + ax * by - ay * bx;
    out[7] = aw * bw - ax * bx - ay * by - az * bz;

    *out
}

pub fn add(out: &mut Quat2, a: &Quat2, b: &Quat2) -> Quat2 {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
    out[4] = a[4] + b[4];
    out[5] = a[5] + b[5];
    out[6] = a[6] + b[6];
    out[7] = a[7] + b[7];

    *out
}

pub fn multiply(out: &mut Quat2, a: &Quat2, b: &Quat2) -> Quat2 {
    let ax0 = a[0];
    let ay0 = a[1];
    let az0 = a[2];
    let aw0 = a[3];

    let bx1 = b[4];
    let by1 = b[5];
    let bz1 = b[6];
    let bw1 = b[7];

    let ax1 = a[4];
    let ay1 = a[5];
    let az1 = a[6];
    let aw1 = a[7];

    let bx0 = b[0];
    let by0 = b[1];
    let bz0 = b[2];
    let bw0 = b[3];

    out[0] = ax0 * bw0 + aw0 * bx0 + ay0 * bz0 - az0 * by0;
    out[1] = ay0 * bw0 + aw0 * by0 + az0 * bx0 - ax0 * bz0;
    out[2] = az0 * bw0 + aw0 * bz0 + ax0 * by0 - ay0 * bx0;
    out[3] = aw0 * bw0 - ax0 * bx0 - ay0 * by0 - az0 * bz0;
    out[4] = ax0 * bw1 + aw0 * bx1 + ay0 * bz1 - az0 * by1 + ax1 * bw0 + aw1 * bx0 + ay1 * bz0 - az1 * by0;
    out[5] = ay0 * bw1 + aw0 * by1 + az0 * bx1 - ax0 * bz1 + ay1 * bw0 + aw1 * by0 + az1 * bx0 - ax1 * bz0;
    out[6] = az0 * bw1 + aw0 * bz1 + ax0 * by1 - ay0 * bx1 + az1 * bw0 + aw1 * bz0 + ax1 * by0 - ay1 * bx0;
    out[7] = aw0 * bw1 - ax0 * bx1 - ay0 * by1 - az0 * bz1 + aw1 * bw0 - ax1 * bx0 - ay1 * by0 - az1 * bz0;

    *out
}

pub fn mul(out: &mut Quat2, a: &Quat2, b: &Quat2) -> Quat2 {
    multiply(out, a, b)
}

pub fn scale(out: &mut Quat2, a: &Quat2, b: f32) -> Quat2 {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
    out[3] = a[3] * b;
    out[4] = a[4] * b;
    out[5] = a[5] * b;
    out[6] = a[6] * b;
    out[7] = a[7] * b;

    *out
}

pub fn dot(a: &Quat2, b: &Quat2) -> f32 {
    let a_real = &[a[1], a[2], a[3], a[4]];
    let b_real = &[b[1], b[2], b[3], b[4]];

    quat::dot(a_real, b_real)
}

pub fn lerp(out: &mut Quat2, a: &Quat2, b: &Quat2, t: f32) -> Quat2 {
    let mut t = t;
    let mt = 1_f32 - t;

    if dot(a, b) < 0_f32 { 
         t = -t;
    }

    out[0] = a[0] * mt + b[0] * t;
    out[1] = a[1] * mt + b[1] * t;
    out[2] = a[2] * mt + b[2] * t;
    out[3] = a[3] * mt + b[3] * t;
    out[4] = a[4] * mt + b[4] * t;
    out[5] = a[5] * mt + b[5] * t;
    out[6] = a[6] * mt + b[6] * t;
    out[7] = a[7] * mt + b[7] * t;

    *out
}

pub fn invert(out: &mut Quat2, a: &Quat2) {
    let sqlen = squared_length(a);

    out[0] = -a[0] / sqlen;
    out[1] = -a[1] / sqlen;
    out[2] = -a[2] / sqlen;
    out[3] = a[3] / sqlen;
    out[4] = -a[4] / sqlen;
    out[5] = -a[5] / sqlen;
    out[6] = -a[6] / sqlen;
    out[7] = a[7] / sqlen;
}

pub fn conjugate(out: &mut Quat2, a: &Quat2) {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];
    out[3] = a[3];
    out[4] = -a[4];
    out[5] = -a[5];
    out[6] = -a[6];
    out[7] = a[7];
}

pub fn length(a: &Quat2) -> f32 { 
    let a_real = &[a[1], a[2], a[3], a[4]];
    
    quat::length(a_real)
}

pub fn len(a: &Quat2) -> f32 { 
    length(a)
}


pub fn squared_length(a: &Quat2) -> f32 { 
    let a_real = &[a[1], a[2], a[3], a[4]];
    
    quat::squared_length(a_real)
}

pub fn sqr_len(a: &Quat2) -> f32 { 
    squared_length(a)
}

pub fn normalize(out: &mut Quat2, a: &Quat2) {
    let mut magnitude = squared_length(a);

    if magnitude > 0_f32 {
        magnitude = f32::sqrt(magnitude);
        let a0 = a[0] / magnitude;
        let a1 = a[1] / magnitude;
        let a2 = a[2] / magnitude;
        let a3 = a[3] / magnitude;
        
        let b0 = a[4];
        let b1 = a[5];
        let b2 = a[6];
        let b3 = a[7];
       
        let a_dot_b = (a0 * b0) + (a1 * b1) + (a2 * b2) + (a3 * b3);

        out[0] = a0;
        out[1] = a1;
        out[2] = a2;
        out[3] = a3;
        out[4] = (b0 - (a0 * a_dot_b)) / magnitude;
        out[5] = (b1 - (a1 * a_dot_b)) / magnitude;
        out[6] = (b2 - (a2 * a_dot_b)) / magnitude;
        out[7] = (b3 - (a3 * a_dot_b)) / magnitude;
    }
}

pub fn quat2_string(a: &Quat2) -> String {
    let a0 = ["quat2(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = a[2].to_string(); 
    let a3 = a[3].to_string(); 
    let a4 = a[4].to_string(); 
    let a5 = a[5].to_string(); 
    let a6 = a[6].to_string(); 
    let a7 = [a[7].to_string(), ")".to_string()].join("");

    [a0, a1, a2, a3, a4, a5, a6, a7].join(", ")
}

pub fn exact_equals(a: &Quat2, b: &Quat2) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] &&
    a[4] == b[4] && a[5] == b[5] && a[6] == b[6] && a[7] == b[7]
}

pub fn equals(a: &Quat2, b: &Quat2) -> bool {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    let a4 = a[4];
    let a5 = a[5];
    let a6 = a[6];
    let a7 = a[7];

    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3];
    let b4 = b[4];
    let b5 = b[5];
    let b6 = b[6];
    let b7 = b[7];
    
    f32::abs(a0 - b0) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a0), f32::abs(b0))) &&
    f32::abs(a1 - b1) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a1), f32::abs(b1))) && 
    f32::abs(a2 - b2) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a2), f32::abs(b2))) && 
    f32::abs(a3 - b3) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a3), f32::abs(b3))) &&
    f32::abs(a4 - b4) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a4), f32::abs(b4))) &&
    f32::abs(a5 - b5) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a5), f32::abs(b5))) &&
    f32::abs(a6 - b6) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a6), f32::abs(b6))) &&
    f32::abs(a7 - b7) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a7), f32::abs(b7)))
}