use super::common::{Quat, Vec4, Vec3, Mat3, random_f32, EPSILON, PI};
use super::mat3;
use super::vec4;
use super::vec3;

pub fn create() -> Quat {
    let mut out: Quat = [0_f32; 4];

    out[3] = 1.;

    out
}

pub fn identity(out: &mut Quat) {
    out[0] = 0.; 
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 1.;
}

pub fn set_axis_angle(out: &mut Quat, axis: &Vec3, rad: f32) {
    let rad = rad * 0.5;
    let s = f32::sin(rad);
   
    out[0] = s * axis[0];
    out[1] = s * axis[1];
    out[2] = s * axis[2];
    out[3] = f32::cos(rad);
}

pub fn get_axis_angle(out_axis: &mut Vec3, q: &Quat) -> f32 {
    let rad = f32::acos(q[3]) * 2.0;
    let s = f32::sin(rad / 2.0);

    if s > EPSILON {
        out_axis[0] = q[0] / s;
        out_axis[1] = q[1] / s;
        out_axis[2] = q[2] / s;
    } else {
        // If s is zero, return any axis (no rotation - axis does not matter)
        out_axis[0] = 1.;
        out_axis[1] = 0.;
        out_axis[2] = 0.;
    }

    rad
}

pub fn get_angle(a: &Quat, b: &Quat) -> f32 {
    let dot_product = dot(a, b);

    f32::acos(2_f32 * dot_product * dot_product - 1_f32)
}

pub fn multiply(out: &mut Quat, a: &Quat, b: &Quat) {
    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    let aw = a[3];

    let bx = b[0];
    let by = b[1];
    let bz = b[2];
    let bw = b[3];

    out[0] = ax * bw + aw * bx + ay * bz - az * by;
    out[1] = ay * bw + aw * by + az * bx - ax * bz;
    out[2] = az * bw + aw * bz + ax * by - ay * bx;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
}

pub fn rotate_x(out: &mut Quat, a: &Quat, rad: f32) {
    let rad = rad * 0.5;

    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    let aw = a[3];

    let bx = f32::sin(rad);
    let bw = f32::cos(rad);

    out[0] = ax * bw + aw * bx;
    out[1] = ay * bw + az * bx;
    out[2] = az * bw - ay * bx;
    out[3] = aw * bw - ax * bx;
}

pub fn rotate_y(out: &mut Quat, a: &Quat, rad: f32) {
    let rad = rad * 0.5;

    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    let aw = a[3];

    let by = f32::sin(rad);
    let bw = f32::cos(rad);

    out[0] = ax * bw - az * by;
    out[1] = ay * bw + aw * by;
    out[2] = az * bw + ax * by;
    out[3] = aw * bw - ay * by;
}

pub fn rotate_z(out: &mut Quat, a: &Quat, rad: f32) {
    let rad = rad * 0.5;

    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    let aw = a[3];
    
    let bz = f32::sin(rad);
    let bw = f32::cos(rad);

    out[0] = ax * bw + ay * bz;
    out[1] = ay * bw - ax * bz;
    out[2] = az * bw + aw * bz;
    out[3] = aw * bw - az * bz;
}

pub fn calculate_w(out: &mut Quat, a: &Quat) {
    let x = a[0];
    let y = a[1];
    let z = a[2];

    out[0] = x;
    out[1] = y;
    out[2] = z;
    out[3] = f32::sqrt(f32::abs(1_f32 - x * x - y * y - z * z));
}

pub fn exp(out: &mut Quat, a: &Quat) {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    let w = a[3];

    let r = f32::sqrt(x*x + y*y + z*z);
    let et = f32::exp(w);

    let s = if r > 0_f32 { et * f32::sin(r) / r } else { 0_f32 };

    out[0] = x * s;
    out[1] = y * s;
    out[2] = z * s;
    out[3] = et * f32::cos(r);
}

pub fn ln(out: &mut Quat, a: &Quat) {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    let w = a[3];

    let r = f32::sqrt(x*x + y*y + z*z);

    let t = if r > 0_f32 { f32::atan2(r, w) / r } else { 0_f32 };

    out[0] = x * t;
    out[1] = y * t;
    out[2] = z * t;
    out[3] = 0.5 * f32::ln(x*x + y*y + z*z + w*w);
}

pub fn pow(out: &mut Quat, a: &Quat, b: f32) {
    ln(out, a);
    scale(out, &clone(out), b);
    exp(out, &clone(out));
}

pub fn slerp(out: &mut Quat, a: &Quat, b: &Quat, t: f32) {
    // benchmarks:
    // http://jsperf.com/quaternion-slerp-implementations
    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    let aw = a[3];

    let mut bx = b[0];
    let mut by = b[1];
    let mut bz = b[2];
    let mut bw = b[3];

    let omega;
    let mut cosom;
    let sinom;
    let scale0;
    let scale1;

    // calc cosine
    cosom = ax * bx + ay * by + az * bz + aw * bw;

    // adjust signs (if necessary)
    if cosom < 0_f32 {
        cosom = -cosom;
        bx = - bx;
        by = - by;
        bz = - bz;
        bw = - bw;
    }

    // calculate coefficients
    if (1_f32 - cosom) > EPSILON {
        // standard case (slerp)
        omega  = f32::acos(cosom);
        sinom  = f32::sin(omega);
        scale0 = f32::sin((1_f32 - t) * omega) / sinom;
        scale1 = f32::sin(t * omega) / sinom;
    } else {
        // "from" and "to" quaternions are very close
        //  ... so we can do a linear interpolation
        scale0 = 1_f32 - t;
        scale1 = t;
    }

    // calculate final values
    out[0] = scale0 * ax + scale1 * bx;
    out[1] = scale0 * ay + scale1 * by;
    out[2] = scale0 * az + scale1 * bz;
    out[3] = scale0 * aw + scale1 * bw;
}

pub fn random(out: &mut Quat) {
    // Implementation of http://planning.cs.uiuc.edu/node198.html
    // TODO: Calling random 3 times is probably not the fastest solution
    let u1 = random_f32();
    let u2 = random_f32();
    let u3 = random_f32();
    
    let sqrt1_minus_u1 = f32::sqrt(1. - u1);
    let sqrt_u1 = f32::sqrt(u1);

    out[0] = sqrt1_minus_u1 * f32::sin(2.0 * PI * u2);
    out[1] = sqrt1_minus_u1 * f32::cos(2.0 * PI * u2);
    out[2] = sqrt_u1 * f32::sin(2.0 * PI * u3);
    out[3] = sqrt_u1 * f32::cos(2.0 * PI * u3);
}

pub fn invert(out: &mut Quat, a: &Quat) {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];

    let dot = a0*a0 + a1*a1 + a2*a2 + a3*a3;

    let inv_dot = if dot != 0_f32 { 1_f32 / dot } else { 0_f32 };

    // TODO: Would be faster to return [0,0,0,0] immediately if dot == 0
    out[0] = -a0 * inv_dot;
    out[1] = -a1 * inv_dot;
    out[2] = -a2 * inv_dot;
    out[3] = a3 * inv_dot;
}

pub fn conjugate(out: &mut Quat, a: &Quat) {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];
    out[3] = a[3];
}

pub fn from_mat3(out: &mut Quat, m: &Mat3) -> Quat {
    // Algorithm in Ken Shoemake's article in 1987 SIGGRAPH course notes
    // article "Quaternion Calculus and Fast Animation".
    let f_trace = m[0] + m[4] + m[8];
    let mut f_root;

    if f_trace > 0.0 {
        // |w| > 1/2, may as well choose w > 1/2
        f_root = f32::sqrt(f_trace + 1.0);  // 2w
        out[3] = 0.5 * f_root;
        f_root = 0.5 / f_root;  // 1/(4w)
        out[0] = (m[5]-m[7]) * f_root;
        out[1] = (m[6]-m[2]) * f_root;
        out[2] = (m[1]-m[3]) * f_root;
    } else {
        // |w| <= 1/2
        let mut i = 0;

        if  m[4] > m[0] {
            i = 1;
        }
        if m[8] > m[i*3+i] {
            i = 2;
        }
        
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;

        f_root = f32::sqrt(m[i*3+i]-m[j*3+j]-m[k*3+k] + 1.0);
        out[i] = 0.5 * f_root;
        f_root = 0.5 / f_root;
        out[3] = (m[j*3+k] - m[k*3+j]) * f_root;
        out[j] = (m[j*3+i] + m[i*3+j]) * f_root;
        out[k] = (m[k*3+i] + m[i*3+k]) * f_root;
    }

    *out 
}

pub fn from_euler(out: &mut Quat, x: f32, y: f32, z: f32) {
    let half_to_rad = 0.5 * PI / 180.0;

    let x = x * half_to_rad;
    let y = y * half_to_rad;
    let z = z * half_to_rad;

    let sx = f32::sin(x);
    let cx = f32::cos(x);
    let sy = f32::sin(y);
    let cy = f32::cos(y);
    let sz = f32::sin(z);
    let cz = f32::cos(z);

    out[0] = sx * cy * cz - cx * sy * sz;
    out[1] = cx * sy * cz + sx * cy * sz;
    out[2] = cx * cy * sz - sx * sy * cz;
    out[3] = cx * cy * cz + sx * sy * sz;
}

pub fn quat_string(a: &Quat) -> String {
    let a0 = ["quat(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = a[2].to_string(); 
    let a3 = [a[3].to_string(), ")".to_string()].join("");

    [a0, a1, a2, a3].join(", ")
}

pub fn clone(a: &Quat) -> Quat { 
     vec4::clone(a)
}

pub fn from_values(x: f32, y: f32, z: f32, w: f32) -> Quat {
    vec4::from_values(x, y, z, w)
}

pub fn copy(out: &mut Quat, a: &Quat) {
    vec4::copy(out, a);
}

pub fn set(out: &mut Vec4, x: f32, y: f32, z: f32, w: f32) {
    vec4::set(out, x, y, z, w);
}

pub fn add(out: &mut Quat, a: &Quat, b: &Quat) {
    vec4::add(out, a, b);
}

pub fn subtract(out: &mut Quat, a: &Quat, b: &Quat) {
    vec4::subtract(out, a, b);
}

pub fn scale(out: &mut Quat, a: &Quat, b: f32) {
    vec4::scale(out, a, b);
}

pub fn dot(a: &Quat, b: &Quat) -> f32 { 
    vec4::dot(a, b)
}

pub fn lerp(out: &mut Quat, a: &Quat, b: &Quat, t: f32) {
    vec4::lerp(out, a, b, t);
}

pub fn length(a: &Quat) -> f32 {
    vec4::length(a)
}

pub fn len(a: &Quat) -> f32 {
    length(a)
}

pub fn squared_length(a: &Quat) -> f32 {
    vec4::squared_length(a)
}

pub fn sqr_len(a: &Quat) -> f32 {
    squared_length(a)
}

pub fn normalize(out: &mut Quat, a: &Quat) {
    vec4::normalize(out, a);
}

pub fn exact_equals(a: &Quat, b: &Quat) -> bool {
    vec4::exact_equals(a, b)
}

pub fn equals(a: &Quat, b: &Quat) -> bool {
    vec4::equals(a, b)
}

pub fn rotation_to(out: &mut Quat, a: &Vec3, b: &Vec3) { 
    let tmp_vec3 = &mut vec3::create();
    let x_unit_vec3 = &vec3::from_values(1.,0.,0.);
    let y_unit_vec3 = &vec3::from_values(0.,1.,0.);

    let dot = vec3::dot(a, b);
    if dot < -0.999999 {
        vec3::cross(tmp_vec3, x_unit_vec3, a);
        if vec3::len(tmp_vec3) < 0.000001 {
            vec3::cross(tmp_vec3, y_unit_vec3, a);
        }
        vec3::normalize(tmp_vec3, &vec3::clone(tmp_vec3));
        set_axis_angle(out, tmp_vec3, PI);
    } else if dot > 0.999999 {
        out[0] = 0.;
        out[1] = 0.;
        out[2] = 0.;
        out[3] = 1.;
    } else {
        vec3::cross(tmp_vec3, a, b);
        out[0] = tmp_vec3[0];
        out[1] = tmp_vec3[1];
        out[2] = tmp_vec3[2];
        out[3] = 1. + dot;

        normalize(out, &clone(out));
    }
}

pub fn sqlerp(out: &mut Quat, a: &Quat, b: &Quat, 
                              c: &Quat, d: &Quat, 
                              t: f32) { 
    let temp1 = &mut create();
    let temp2 = &mut create();

    slerp(temp1, a, d, t);
    slerp(temp2, b, c, t);
    slerp(out, temp1, temp2, 2. * t * (1. - t));
}

pub fn set_axis(out: &mut Quat, view: &Vec3, 
                right: &Vec3, up: &Vec3){
    let matr = &mut mat3::create();

    matr[0] = right[0];
    matr[3] = right[1];
    matr[6] = right[2];
    matr[1] = up[0];
    matr[4] = up[1];
    matr[7] = up[2];
    matr[2] = -view[0];
    matr[5] = -view[1];
    matr[8] = -view[2];
    
    
    normalize(out, &from_mat3(&mut clone(out), matr));
}
