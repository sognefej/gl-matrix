use super::common::{Vec3, Mat4, Mat3, Quat, hypot, random_f32, EPSILON, PI};

pub fn create() -> Vec3 {
    let out: Vec3 = [0_f32; 3];
    
    out
}

pub fn clone(a: &Vec3) -> Vec3 {
    let mut out: Vec3 = [0_f32; 3];
    
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    
    out
}

pub fn length(a: &Vec3) -> f32 {
    let mut len: Vec3 = [0_f32; 3];
    
    len[0] = a[0]; // x
    len[1] = a[1]; // y
    len[2] = a[2]; // z
    
    hypot(&len)
}

pub fn from_values(x: f32, y: f32, z: f32) ->  Vec3 { 
    let mut out: Vec3 = [0_f32; 3];
    
    out[0] = x;
    out[1] = y;
    out[2] = z;
    
    out
}

pub fn copy(out: &mut Vec3, a: &Vec3) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
}

pub fn set(out: &mut Vec3, x: f32, y: f32, z: f32) {
    out[0] = x;
    out[1] = y;
    out[2] = z;
}

pub fn add(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3{
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];

    *out
}

pub fn subtract(out: &mut Vec3, a: &Vec3, b: &Vec3) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

pub fn multiply(out: &mut Vec3, a: &Vec3, b: &Vec3) {
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
    out[2] = a[2] * b[2];
}


pub fn divide(out: &mut Vec3, a: &Vec3, b: &Vec3) {
    out[0] = a[0] / b[0];
    out[1] = a[1] / b[1];
    out[2] = a[2] / b[2];
}

pub fn ceil(out: &mut Vec3, a: &Vec3) {
    out[0] = f32::ceil(a[0]);
    out[1] = f32::ceil(a[1]);
    out[2] = f32::ceil(a[2]);
}

pub fn floor(out: &mut Vec3, a: &Vec3) {
    out[0] = f32::floor(a[0]);
    out[1] = f32::floor(a[1]);
    out[2] = f32::floor(a[2]);
}

pub fn min(out: &mut Vec3, a: &Vec3, b: &Vec3) {
    out[0] = f32::min(a[0], b[0]);
    out[1] = f32::min(a[1], b[1]);
    out[2] = f32::min(a[2], b[2]);
}

pub fn max(out: &mut Vec3, a: &Vec3, b: &Vec3) {
    out[0] = f32::max(a[0], b[0]);
    out[1] = f32::max(a[1], b[1]);
    out[2] = f32::max(a[2], b[2]);
}

pub fn round(out: &mut Vec3, a: &Vec3) {
    out[0] = f32::round(a[0]);
    out[1] = f32::round(a[1]);
    out[2] = f32::round(a[2]);
}

pub fn scale(out: &mut Vec3, a: &Vec3, b: f32) {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
}

pub fn scale_and_add(out: &mut Vec3, a: &Vec3, b: &Vec3, scale: f32) {
    out[0] = a[0] + (b[0] * scale);
    out[1] = a[1] + (b[1] * scale);
    out[2] = a[2] + (b[2] * scale);
}

pub fn distance(a: &Vec3, b: &Vec3) -> f32 {
    let mut dist: Vec3 = [0_f32; 3];
    
    dist[0] = b[0] - a[0]; // x
    dist[1] = b[1] - a[1]; // y
    dist[2] = b[2] - a[2]; // z
    
    hypot(&dist)
}

pub fn squared_distance(a: &Vec3, b: &Vec3) -> f32 {
    let x = b[0] - a[0];
    let y = b[1] - a[1];
    let z = b[2] - a[2];
    
    x*x + y*y + z*z
}

pub fn squared_length(a: &Vec3) -> f32 {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    
    x*x + y*y + z*z
}

pub fn negate(out: &mut Vec3, a: &Vec3) {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];
}

pub fn inverse(out: &mut Vec3, a: &Vec3) {
    out[0] = 1.0 / a[0];
    out[1] = 1.0 / a[1];
    out[2] = 1.0 / a[2];
}

pub fn normalize(out: &mut Vec3, a: &Vec3) {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    
    let mut len = x*x + y*y + z*z;
    if len > 0_f32 {
        //TODO: evaluate use of glm_invsqrt here?
        len = 1_f32 / f32::sqrt(len);
    }
    
    out[0] = a[0] * len;
    out[1] = a[1] * len;
    out[2] = a[2] * len;
}

pub fn dot(a: &Vec3, b: &Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(out: &mut Vec3, a: &Vec3, b: &Vec3) {
    let ax = a[0];
    let ay = a[1];
    let az = a[2];

    let bx = b[0];
    let by = b[1];
    let bz = b[2];
    
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
}

pub fn lerp(out: &mut Vec3, a: &Vec3, b: &Vec3, t: f32) {
    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    
    out[0] = ax + t * (b[0] - ax);
    out[1] = ay + t * (b[1] - ay);
    out[2] = az + t * (b[2] - az);
}

pub fn hermite(out: &mut Vec3, a: &Vec3, b: &Vec3, 
                               c: &Vec3, d: &Vec3, t: f32) {
    let factor_times2 = t * t;
    let factor1 = factor_times2 * (2. * t - 3.) + 1.;
    let factor2 = factor_times2 * (t - 2.) + t;
    let factor3 = factor_times2 * (t - 1.);
    let factor4 = factor_times2 * (3. - 2. * t);
    
    out[0] = a[0] * factor1 + b[0] * factor2 + c[0] * factor3 + d[0] * factor4;
    out[1] = a[1] * factor1 + b[1] * factor2 + c[1] * factor3 + d[1] * factor4;
    out[2] = a[2] * factor1 + b[2] * factor2 + c[2] * factor3 + d[2] * factor4;
}

pub fn bezier(out: &mut Vec3, a: &Vec3, b: &Vec3, 
                              c: &Vec3, d: &Vec3, t: f32) {
    let inverse_factor = 1_f32 - t;
    let inverse_factor_times_two = inverse_factor * inverse_factor;
    let factor_times2 = t * t;
    let factor1 = inverse_factor_times_two * inverse_factor;
    let factor2 = 3_f32 * t * inverse_factor_times_two;
    let factor3 = 3_f32 * factor_times2 * inverse_factor;
    let factor4 = factor_times2 * t;
    
    out[0] = a[0] * factor1 + b[0] * factor2 + c[0] * factor3 + d[0] * factor4;
    out[1] = a[1] * factor1 + b[1] * factor2 + c[1] * factor3 + d[1] * factor4;
    out[2] = a[2] * factor1 + b[2] * factor2 + c[2] * factor3 + d[2] * factor4;
}

pub fn random(out: &mut Vec3, scale: Option<f32>) {    
    let scale = match scale { 
        Some(scale) => scale, 
        None => 1_f32, 
    };
    let r = random_f32() * 2.0 * PI;
    let z = (random_f32() * 2.0) - 1.0;
    let z_scale = f32::sqrt(1.0-z*z) * scale;
    
    out[0] = f32::cos(r) * z_scale;
    out[1] = f32::sin(r) * z_scale;
    out[2] = z * scale;
}


pub fn transform_mat4(out: &mut Vec3, a: &Vec3, m: &Mat4) {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    let w = m[3] * x + m[7] * y + m[11] * z + m[15];
   
    let w = if w != 0_f32 { w } else { 1_f32 };
    
    out[0] = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
    out[1] = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
    out[2] = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;
}

pub fn transform_mat3(out: &mut Vec3, a: &Vec3, m: &Mat3) {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    
    out[0] = x * m[0] + y * m[3] + z * m[6];
    out[1] = x * m[1] + y * m[4] + z * m[7];
    out[2] = x * m[2] + y * m[5] + z * m[8];
}

pub fn transform_quat(out: &mut Vec3, a: &Vec3, q: &Quat) {
    // benchmarks: https://jsperf.com/quaternion-transform-vec3-implementations-fixed
    let qx = q[0];
    let qy = q[1];
    let qz = q[2];
    let qw = q[3];
    
    let x = a[0];
    let y = a[1];
    let z = a[2];
    
    // var qvec = [qx, qy, qz];
    // var uv = vec3.cross([], qvec, a);
    let mut uvx = qy * z - qz * y;
    let mut uvy = qz * x - qx * z;
    let mut uvz = qx * y - qy * x;
    
    // var uuv = vec3.cross([], qvec, uv);
    let mut uuvx = qy * uvz - qz * uvy;
    let mut uuvy = qz * uvx - qx * uvz;
    let mut uuvz = qx * uvy - qy * uvx;
    
    // vec3.scale(uv, uv, 2 * w);
    let w2 = qw * 2.;
    
    uvx *= w2;
    uvy *= w2;
    uvz *= w2;
    
    // vec3.scale(uuv, uuv, 2);
    uuvx *= 2.;
    uuvy *= 2.;
    uuvz *= 2.;
    
    // return vec3.add(out, a, vec3.add(out, uv, uuv));
    out[0] = x + uvx + uuvx;
    out[1] = y + uvy + uuvy;
    out[2] = z + uvz + uuvz;
}

pub fn rotate_x(out: &mut Vec3, a: &Vec3, b: &Vec3, c: f32) {
    let mut p: Vec3 = [0_f32; 3];
    let mut r: Vec3 = [0_f32; 3];
    
    //Translate point to the origin
    p[0] = a[0] - b[0];
    p[1] = a[1] - b[1];
    p[2] = a[2] - b[2];
    
    //perform rotation
    r[0] = p[0];
    r[1] = p[1]*f32::cos(c) - p[2]*f32::sin(c);
    r[2] = p[1]*f32::sin(c) + p[2]*f32::cos(c);
    
    //translate to correct position
    out[0] = r[0] + b[0];
    out[1] = r[1] + b[1];
    out[2] = r[2] + b[2];
}

pub fn rotate_y(out: &mut Vec3, a: &Vec3, b: &Vec3, c: f32) {
    let mut p: Vec3 = [0_f32; 3];
    let mut r: Vec3 = [0_f32; 3];
    
    //Translate point to the origin
    p[0] = a[0] - b[0];
    p[1] = a[1] - b[1];
    p[2] = a[2] - b[2];
    
    //perform rotation
    r[0] = p[2]*f32::sin(c) + p[0]*f32::cos(c);
    r[1] = p[1];
    r[2] = p[2]*f32::cos(c) - p[0]*f32::sin(c);
    
    //translate to correct position
    out[0] = r[0] + b[0];
    out[1] = r[1] + b[1];
    out[2] = r[2] + b[2];
}

pub fn rotate_z(out: &mut Vec3, a: &Vec3, b: &Vec3, c: f32) {
    let mut p: Vec3 = [0_f32; 3];
    let mut r: Vec3 = [0_f32; 3];
    
    //Translate point to the origin
    p[0] = a[0] - b[0];
    p[1] = a[1] - b[1];
    p[2] = a[2] - b[2];
    
    //perform rotation
    r[0] = p[0]*f32::cos(c) - p[1]*f32::sin(c);
    r[1] = p[0]*f32::sin(c) + p[1]*f32::cos(c);
    r[2] = p[2];
    
    //translate to correct position
    out[0] = r[0] + b[0];
    out[1] = r[1] + b[1];
    out[2] = r[2] + b[2];
}

pub fn angle(a: &Vec3, b: &Vec3) -> f32 {
    let temp_a = &mut from_values(a[0], a[1], a[2]);
    let temp_b = &mut from_values(b[0], b[1], b[2]);
    
    normalize(temp_a, &clone(temp_a));
    normalize(temp_b, &clone(temp_b));
    
    let cosine = dot(temp_a, temp_b);
    
    if cosine > 1_f32 {
        return 0_f32;
    } else if cosine < -1_f32 {
        return PI;
    } else {
        return f32::acos(cosine);
    }
}

pub fn zero(out: &mut Vec3) {
    out[0] = 0.0;
    out[1] = 0.0;
    out[2] = 0.0;
}

pub fn vec3_string(a: &Quat) -> String {
    let a0 = ["vec3(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = [a[2].to_string(), ")".to_string()].join("");

    [a0, a1, a2].join(", ")
}

pub fn exact_equals(a: &Vec3, b: &Vec3) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
}

pub fn equals(a: &Vec3, b: &Vec3) -> bool {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    
    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    
    f32::abs(a0 - b0) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a0), f32::abs(b0))) && 
    f32::abs(a1 - b1) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a1), f32::abs(b1))) && 
    f32::abs(a2 - b2) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a2), f32::abs(b2)))
}

pub fn sub(out: &mut Vec3, a: &Vec3, b: &Vec3) {
    subtract(out, a, b);
}

pub fn mul(out: &mut Vec3, a: &Vec3, b: &Vec3) {
    multiply(out, a, b);
}

pub fn div(out: &mut Vec3, a: &Vec3, b: &Vec3) {
    divide(out, a, b);
}

pub fn dist(a: &Vec3, b: &Vec3) -> f32 {
    distance(a, b)
}

pub fn sqr_dist(a: &Vec3, b: &Vec3) -> f32 {
    squared_distance(a, b)
}

pub fn len(a: &Vec3) -> f32 {
    length(a)
}

pub fn sqr_len(a: &Vec3) -> f32 {
    squared_length(a)
}

// /**
// * Perform some operation over an array of vec3s.
// *
// * @param {Array} a the array of vectors to iterate over
// * @param {Number} stride Number of elements between the start of each vec3. If 0 assumes tightly packed
// * @param {Number} offset Number of elements to skip at the beginning of the array
// * @param {Number} count Number of vec3s to iterate over. If 0 iterates over entire array
// * @param {fn} fn fn to call for each vector in the array
// * @param {Object} [arg] additional argument to pass to fn
// * @returns {Array} a
// * @fn
// */
// pub const forEach = (fn() {
//     let vec = create();
//     return fn(a, stride, offset, count, fn, arg) {
//         let i, l;
//         if(!stride) {
//             stride = 3;
//         }
        
//         if(!offset) {
//             offset = 0;
//         }
        
//         if(count) {
//             l = f32::min((count * stride) + offset, a.length);
//         } else {
//             l = a.length;
//         }
        
//         for(i = offset; i < l; i += stride) {
//             vec[0] = a[i]; vec[1] = a[i+1]; vec[2] = a[i+2];
//             fn(vec, vec, arg);
//             a[i] = vec[0]; a[i+1] = vec[1]; a[i+2] = vec[2];
//         }
//         return a;
//     };
//})();