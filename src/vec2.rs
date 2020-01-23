use super::common::{Vec2, Vec3, Vec4, Mat2, Mat2d, Mat3, Mat4, hypot, random_f32, EPSILON, PI};

pub fn create() -> Vec2 {
    let out: Vec2 = [0_f32; 2];

    out
}

pub fn clone(a: &Vec2) -> Vec2 {
    let mut out: Vec2 = [0_f32; 2];

    out[0] = a[0];
    out[1] = a[1];

    out
}

pub fn from_values(x: f32, y: f32) -> Vec2 {
    let mut out: Vec2 = [0_f32; 2];

    out[0] = x;
    out[1] = y;

    out
}

pub fn copy(out: &mut Vec2, a: &Vec2) {
    out[0] = a[0];
    out[1] = a[1];
}

pub fn set(out: &mut Vec2, x: f32, y: f32) {
    out[0] = x;
    out[1] = y;
}

pub fn add(out: &mut Vec2, a: &Vec2, b: &Vec2) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
}

pub fn subtract(out: &mut Vec2, a: &Vec2, b: &Vec2) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
}

pub fn multiply(out: &mut Vec2, a: &Vec2, b: &Vec2) {
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
}

pub fn divide(out: &mut Vec2, a: &Vec2, b: &Vec2) {
    out[0] = a[0] / b[0];
    out[1] = a[1] / b[1];
}

pub fn ceil(out: &mut Vec2, a: &Vec2) {
    out[0] = f32::ceil(a[0]);
    out[1] = f32::ceil(a[1]);
}

pub fn floor(out: &mut Vec2, a: &Vec2) {
    out[0] = f32::floor(a[0]);
    out[1] = f32::floor(a[1]);
}

pub fn min(out: &mut Vec2, a: &Vec2, b: &Vec2) {
    out[0] = f32::min(a[0], b[0]);
    out[1] = f32::min(a[1], b[1]);
}

pub fn max(out: &mut Vec2, a: &Vec2, b: &Vec2) {
    out[0] = f32::max(a[0], b[0]);
    out[1] = f32::max(a[1], b[1]);
}

pub fn round(out: &mut Vec2, a: &Vec2) {
    out[0] = f32::round(a[0]);
    out[1] = f32::round(a[1]);
}

pub fn scale(out: &mut Vec2, a: &Vec2, b: f32) {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
}

pub fn scale_and_add(out: &mut Vec2, a: &Vec2, b: &Vec2, scale: f32) {
    out[0] = a[0] + (b[0] * scale);
    out[1] = a[1] + (b[1] * scale);
}

pub fn distance(a: &Vec2, b: &Vec2) -> f32 {
    let mut dist: Vec2 = [0_f32; 2];
    
    dist[0] = b[0] - a[0]; // x
    dist[1] = b[1] - a[1]; // y

    hypot(&dist.to_vec())
}

pub fn squared_distance(a: &Vec2, b: &Vec2) -> f32 {
    let x = b[0] - a[0];
    let y = b[1] - a[1];

    x*x + y*y
}

pub fn length(a: &Vec2) -> f32 {
    let mut len: Vec2 = [0_f32; 2];
    
    len[0] = a[0]; // x
    len[1] = a[1]; // y

    hypot(&len.to_vec())
}

pub fn squared_length(a: &Vec2) -> f32 {
    let x = a[0];
    let y = a[1];

    x*x + y*y
}

pub fn negate(out: &mut Vec2, a: &Vec2) {
    out[0] = -a[0];
    out[1] = -a[1];
}

pub fn inverse(out: &mut Vec2, a: &Vec2) {
    out[0] = 1.0 / a[0];
    out[1] = 1.0 / a[1];
}

pub fn normalize(out: &mut Vec2, a: &Vec2) {
    let x = a[0];
    let y = a[1];

    let mut len = x*x + y*y;

    if len > 0_f32 {
        //TODO: evaluate use of glm_invsqrt here?
        len = 1_f32 / f32::sqrt(len);
    }    

    out[0] = a[0] * len;
    out[1] = a[1] * len;
}

pub fn dot(a: &Vec2, b: &Vec2) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

pub fn cross(out: &mut Vec3, a: &Vec2, b: &Vec2) {
    let z = a[0] * b[1] - a[1] * b[0];

    out[0] = 0.; 
    out[1] = 0.;
    out[2] = z;
}

pub fn lerp(out: &mut Vec2, a: &Vec2, b: &Vec2, t: f32) {
    let ax = a[0];
    let ay = a[1];

    out[0] = ax + t * (b[0] - ax);
    out[1] = ay + t * (b[1] - ay);
}

pub fn random(out: &mut Vec2, scale: Option<f32>) {
    let scale = match scale { 
        Some(scale) => scale, 
        None => 1_f32, 
    };

    let r = random_f32() * 2.0 * PI; 

    out[0] = f32::cos(r) * scale;
    out[1] = f32::sin(r) * scale;
}

pub fn transform_mat2(out: &mut Vec2, a: Vec2, m: &Mat2) {
    let x = a[0];
    let y = a[1];

    out[0] = m[0] * x + m[2] * y;
    out[1] = m[1] * x + m[3] * y;
}

pub fn transform_mat2d(out: &mut Vec2, a: &Vec2, m: &Mat2d) {
    let x = a[0];
    let y = a[1];

    out[0] = m[0] * x + m[2] * y + m[4];
    out[1] = m[1] * x + m[3] * y + m[5];
}

pub fn transform_mat3(out: &mut Vec2, a: &Vec2, m: &Mat3) {
    let x = a[0];
    let y = a[1];

    out[0] = m[0] * x + m[3] * y + m[6];
    out[1] = m[1] * x + m[4] * y + m[7];
}

pub fn transform_mat4(out: &mut Vec2, a: &Vec2, m: &Mat4) {
    let x = a[0];
    let y = a[1];

    out[0] = m[0] * x + m[4] * y + m[12];
    out[1] = m[1] * x + m[5] * y + m[13];
}

pub fn rotate(out: &mut Vec2, a: &Vec2, b: &Vec2, c: f32) {
    //Translate point to the origin
    let p0 = a[0] - b[0];
    let p1 = a[1] - b[1];

    let sin_c = f32::sin(c);
    let cos_c = f32::cos(c);
    
    //perform rotation and translate to correct position
    out[0] = p0 * cos_c - p1 * sin_c + b[0];
    out[1] = p0 * sin_c + p1 * cos_c + b[1];
}

pub fn angle(a: &Vec2, b: &Vec2) -> f32 {
    let x1 = a[0];
    let y1 = a[1];
    let x2 = b[0];
    let y2 = b[1];
    
    let mut len1 = x1*x1 + y1*y1;
    if len1 > 0_f32 {
        //TODO: evaluate use of glm_invsqrt here?
        len1 = 1_f32 / f32::sqrt(len1);
    }

    let mut len2 = x2*x2 + y2*y2;
    if len2 > 0_f32 {
        //TODO: evaluate use of glm_invsqrt here?
        len2 = 1_f32 / f32::sqrt(len2);
    }

    let cosine = (x1 * x2 + y1 * y2) * len1 * len2;
    if cosine > 1_f32 {
        return 0_f32;
    } else if cosine < -1_f32 {
        return PI;
    } else {
        return f32::acos(cosine);
    }
}

pub fn zero(out: &mut Vec2) {
    out[0] = 0.0;
    out[1] = 0.0;
}

pub fn mat3_string(a: &Vec4) -> String {
    let a0 = ["vec2(".to_string(), a[0].to_string()].join("");
    let a1 = [a[1].to_string(), ")".to_string()].join("");

    [a0, a1].join(", ")
}

pub fn exact_equals(a: &Vec2,  b: &Vec2) -> bool {
    a[0] == b[0] && a[1] == b[1]
}

pub fn equals(a: &Vec2, b: &Vec2) -> bool {
    let a0 = a[0];
    let a1 = a[1];
    let b0 = b[0];
    let b1 = b[1];

    f32::abs(a0 - b0) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a0), f32::abs(b0))) && 
    f32::abs(a1 - b1) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a1), f32::abs(b1)))
}

pub fn len(a: &Vec2) -> f32 {
    length(a)
}

pub fn sub(out: &mut Vec2, a: &Vec2, b: &Vec2) {
    subtract(out, a, b);
}

pub fn mul(out: &mut Vec2, a: &Vec2, b: &Vec2) {
    multiply(out, a, b);
}

pub fn div(out: &mut Vec2, a: &Vec2, b: &Vec2) {
    divide(out, a, b);
}

pub fn dist(a: &Vec2, b: &Vec2) -> f32 {
    distance(a, b)
}

pub fn sqr_dist(a: &Vec2, b: &Vec2) -> f32 {
    squared_distance(a, b)
}

pub fn sqr_len(a: &Vec2) -> f32 {
    squared_length(a)
}

// /**
//  * Perform some operation over an array of vec2s.
//  *
//  * @param {Array} a the array of vectors to iterate over
//  * @param {Number} stride Number of elements between the start of each vec2. If 0 assumes tightly packed
//  * @param {Number} offset Number of elements to skip at the beginning of the array
//  * @param {Number} count Number of vec2s to iterate over. If 0 iterates over entire array
//  * @param {fn} fn fn to call for each vector in the array
//  * @param {Object} [arg] additional argument to pass to fn
//  * @returns {Array} a
//  * @fn
//  */
// pub const forEach = (fn() {
//   let vec = create();
//   return fn(a, stride, offset, count, fn, arg) {
//     let i, l;
//     if(!stride) {
//       stride = 2;
//     }
//     if(!offset) {
//       offset = 0;
//     }
//     if(count) {
//       l = f32::min((count * stride) + offset, a.length);
//     } else {
//       l = a.length;
//     }
//     for(i = offset; i < l; i += stride) {
//       vec[0] = a[i]; vec[1] = a[i+1];
//       fn(vec, vec, arg);
//       a[i] = vec[0]; a[i+1] = vec[1];
//     }
//     return a;
//   };
// })();