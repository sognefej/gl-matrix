//! 3 Dimensional Vector

use super::common::{Vec3, Mat4, Mat3, Quat, hypot, random_f32, EPSILON, PI};

/// Creates a new, empty vec3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn create() -> Vec3 {
    let out: Vec3 = [0_f32; 3];
    
    out
}

/// Creates a new vec3 initialized with values from an existing vector.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn clone(a: &Vec3) -> Vec3 {
    let mut out: Vec3 = [0_f32; 3];
    
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    
    out
}

/// Calculates the length of a vec3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn length(a: &Vec3) -> f32 {
    let mut len: Vec3 = [0_f32; 3];
    
    len[0] = a[0]; // x
    len[1] = a[1]; // y
    len[2] = a[2]; // z
    
    hypot(&len)
}

/// Creates a new vec3 initialized with the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn from_values(x: f32, y: f32, z: f32) ->  Vec3 { 
    let mut out: Vec3 = [0_f32; 3];
    
    out[0] = x;
    out[1] = y;
    out[2] = z;
    
    out
}

/// Copy the values from one vec3 to another.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn copy(out: &mut Vec3, a: &Vec3) -> Vec3 {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];

    *out
}

/// Set the components of a vec3 to the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn set(out: &mut Vec3, x: f32, y: f32, z: f32) -> Vec3 {
    out[0] = x;
    out[1] = y;
    out[2] = z;

    *out
}

/// Adds two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn add(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];

    *out
}

/// Subtracts vector b from vector a.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn subtract(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];

    *out
}

/// Multiplies two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn multiply(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
    out[2] = a[2] * b[2];

    *out
}

/// Divides two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn divide(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    out[0] = a[0] / b[0];
    out[1] = a[1] / b[1];
    out[2] = a[2] / b[2];

    *out
}

/// f32::ceil the components of a vec3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn ceil(out: &mut Vec3, a: &Vec3) -> Vec3 {
    out[0] = f32::ceil(a[0]);
    out[1] = f32::ceil(a[1]);
    out[2] = f32::ceil(a[2]);

    *out
}

/// f32::ceil the components of a vec3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn floor(out: &mut Vec3, a: &Vec3) -> Vec3 {
    out[0] = f32::floor(a[0]);
    out[1] = f32::floor(a[1]);
    out[2] = f32::floor(a[2]);

    *out
}

/// Returns the minimum of two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn min(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    out[0] = f32::min(a[0], b[0]);
    out[1] = f32::min(a[1], b[1]);
    out[2] = f32::min(a[2], b[2]);

    *out
}

/// Returns the maximum of two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn max(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    out[0] = f32::max(a[0], b[0]);
    out[1] = f32::max(a[1], b[1]);
    out[2] = f32::max(a[2], b[2]);

    *out
}

/// f32::round the components of a vec3
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn round(out: &mut Vec3, a: &Vec3) -> Vec3 {
    out[0] = f32::round(a[0]);
    out[1] = f32::round(a[1]);
    out[2] = f32::round(a[2]);

    *out
}

/// Scales a vec3 by a scalar number.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn scale(out: &mut Vec3, a: &Vec3, b: f32) -> Vec3 {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;

    *out
}

/// Adds two vec3's after scaling the second operand by a scalar value.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn scale_and_add(out: &mut Vec3, a: &Vec3, b: &Vec3, scale: f32) -> Vec3 {
    out[0] = a[0] + (b[0] * scale);
    out[1] = a[1] + (b[1] * scale);
    out[2] = a[2] + (b[2] * scale);

    *out
}

/// Calculates the euclidian distance between two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn distance(a: &Vec3, b: &Vec3) -> f32 {
    let mut dist: Vec3 = [0_f32; 3];
    
    dist[0] = b[0] - a[0]; // x
    dist[1] = b[1] - a[1]; // y
    dist[2] = b[2] - a[2]; // z
    
    hypot(&dist)
}

/// Calculates the squared euclidian distance between two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn squared_distance(a: &Vec3, b: &Vec3) -> f32 {
    let x = b[0] - a[0];
    let y = b[1] - a[1];
    let z = b[2] - a[2];
    
    x*x + y*y + z*z
}

/// Calculates the squared length of a vec3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn squared_length(a: &Vec3) -> f32 {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    
    x*x + y*y + z*z
}

/// Negates the components of a vec3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn negate(out: &mut Vec3, a: &Vec3) -> Vec3 {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];

    *out
}

/// Returns the inverse of the components of a vec3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn inverse(out: &mut Vec3, a: &Vec3) -> Vec3 {
    out[0] = 1.0 / a[0];
    out[1] = 1.0 / a[1];
    out[2] = 1.0 / a[2];

    *out
}

/// Normalize a vec3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn normalize(out: &mut Vec3, a: &Vec3) -> Vec3 {
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

    *out
}

/// Calculates the dot product of two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn dot(a: &Vec3, b: &Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Computes the cross product of two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn cross(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    let ax = a[0];
    let ay = a[1];
    let az = a[2];

    let bx = b[0];
    let by = b[1];
    let bz = b[2];
    
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;

    *out
}

/// Performs a linear interpolation between two vec3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn lerp(out: &mut Vec3, a: &Vec3, b: &Vec3, t: f32) -> Vec3 {
    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    
    out[0] = ax + t * (b[0] - ax);
    out[1] = ay + t * (b[1] - ay);
    out[2] = az + t * (b[2] - az);

    *out
}

/// Performs a hermite interpolation with two control points.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn hermite(out: &mut Vec3, a: &Vec3, b: &Vec3, 
                               c: &Vec3, d: &Vec3, t: f32) -> Vec3 {
    let factor_times2 = t * t;
    let factor1 = factor_times2 * (2. * t - 3.) + 1.;
    let factor2 = factor_times2 * (t - 2.) + t;
    let factor3 = factor_times2 * (t - 1.);
    let factor4 = factor_times2 * (3. - 2. * t);
    
    out[0] = a[0] * factor1 + b[0] * factor2 + c[0] * factor3 + d[0] * factor4;
    out[1] = a[1] * factor1 + b[1] * factor2 + c[1] * factor3 + d[1] * factor4;
    out[2] = a[2] * factor1 + b[2] * factor2 + c[2] * factor3 + d[2] * factor4;

    *out
}

/// Performs a bezier interpolation with two control points.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn bezier(out: &mut Vec3, a: &Vec3, b: &Vec3, 
                              c: &Vec3, d: &Vec3, t: f32) -> Vec3 {
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

    *out
}

/// Generates a random vector with the given scale.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn random(out: &mut Vec3, scale: Option<f32>) -> Vec3 {    
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

    *out
}


/// Transforms the vec3 with a mat4.
/// 4th vector component is implicitly '1'.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn transform_mat4(out: &mut Vec3, a: &Vec3, m: &Mat4) -> Vec3 {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    let w = m[3] * x + m[7] * y + m[11] * z + m[15];
   
    let w = if w != 0_f32 { w } else { 1_f32 };
    
    out[0] = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
    out[1] = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
    out[2] = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;

    *out
}

/// Transforms the vec3 with a mat3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn transform_mat3(out: &mut Vec3, a: &Vec3, m: &Mat3) -> Vec3 {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    
    out[0] = x * m[0] + y * m[3] + z * m[6];
    out[1] = x * m[1] + y * m[4] + z * m[7];
    out[2] = x * m[2] + y * m[5] + z * m[8];

    *out
}

/// Transforms the vec3 with a quat.
/// Can also be used for dual quaternions. (Multiply it with the real part)
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn transform_quat(out: &mut Vec3, a: &Vec3, q: &Quat) -> Vec3 {
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

    *out
}

/// Rotate a 3D vector around the x-axis.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn rotate_x(out: &mut Vec3, a: &Vec3, b: &Vec3, c: f32) -> Vec3 {
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

    *out
}

/// Rotate a 3D vector around the y-axis.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn rotate_y(out: &mut Vec3, a: &Vec3, b: &Vec3, c: f32) -> Vec3 {
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

    *out
}

/// Rotate a 3D vector around the z-axis.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn rotate_z(out: &mut Vec3, a: &Vec3, b: &Vec3, c: f32) -> Vec3 {
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

    *out
}

/// Get the angle between two 3D vectors.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
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

/// Set the components of a vec3 to zero.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn zero(out: &mut Vec3) -> Vec3 {
    out[0] = 0.0;
    out[1] = 0.0;
    out[2] = 0.0;

    *out
}

/// Returns a string representation of a vector.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn string(a: &Vec3) -> String {
    let a0 = ["vec3(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = [a[2].to_string(), ")".to_string()].join("");

    [a0, a1, a2].join(", ")
}

/// Returns whether or not the vectors have exactly the same elements in the same position (when compared with ==)
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn exact_equals(a: &Vec3, b: &Vec3) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
}

/// Returns whether or not the vectors have approximately the same elements in the same position.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
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

/// Alias for vec3::subtract
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn sub(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    subtract(out, a, b)
}

/// Alias for vec3::multiply
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn mul(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    multiply(out, a, b)
}

/// Alias for vec3::divide
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn div(out: &mut Vec3, a: &Vec3, b: &Vec3) -> Vec3 {
    divide(out, a, b)
}

/// Alias for vec3::distance
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn dist(a: &Vec3, b: &Vec3) -> f32 {
    distance(a, b)
}

/// Alias for vec3::squared_distance
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn sqr_dist(a: &Vec3, b: &Vec3) -> f32 {
    squared_distance(a, b)
}

/// Alias for vec3::length
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
pub fn len(a: &Vec3) -> f32 {
    length(a)
}

/// Alias for vec3::squared_length
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec3.js.html)
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


#[cfg(test)] 
mod tests {
    use super::*; 

    #[test] 
    fn create_a_vec3() { 
        let out = create();
        
        assert_eq!([0., 0., 0.], out);
    }

    #[test] 
    fn clone_a_vec3() {
        let vec_a: Vec3 = [1., 2., 3.];
  
        let out = clone(&vec_a);
       
        assert_eq!(vec_a, out);
    }

    #[test]
    fn length_of_vec3() {
        use super::super::common;

        let vec_a: Vec3 = [1., 2., 3.];

        let result = length(&vec_a);

        assert!(common::equals(result, 3.741657));
    }

    #[test]
    fn len_of_vec3() { 
        use super::super::common;

        let vec_a: Vec3 = [1., 2., 3.];

        let result = len(&vec_a);

        assert!(common::equals(result, 3.741657));
    }

    #[test]
    fn length_is_equal_to_len() {
        let vec_a: Vec3 = [1., 2., 3.];

        let result_a = length(&vec_a);
        let result_b = len(&vec_a);

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn vec3_from_values() { 
        let result = from_values(1., 2., 3.);

        assert_eq!([1., 2., 3.], result);
    }

    #[test]
    fn create_vec3_from_values() { 
        let out = from_values(1., 2., 3.);
    
        assert_eq!([1., 2., 3.], out); 
    }
   
    #[test] 
    fn copy_values_from_a_mat2d_to_another() {
        let mut out =  [0., 0., 0.];
        let mat_a: Vec3 = [1., 2., 3.];
   
        let result = copy(&mut out, &mat_a);
      
        assert_eq!(mat_a, out);
        assert_eq!(result, out);
    }

    #[test]
    fn set_vec2_with_values() { 
        let mut out: Vec3 = [0., 0., 0.];
     
        let result = set(&mut out, 1., 2., 3.);

        assert_eq!([1., 2., 3.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn add_two_vec3s() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = add(&mut out, &vec_a, &vec_b);

        assert_eq!([5., 7., 9.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn subtract_two_vec3s() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = subtract(&mut out, &vec_a, &vec_b);

        assert_eq!([-3., -3., -3.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_two_vec3s() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = sub(&mut out, &vec_a, &vec_b);

        assert_eq!([-3., -3., -3.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_is_equal_to_subtract() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result_a = subtract(&mut out, &vec_a, &vec_b);
        let result_b = sub(&mut out, &vec_a, &vec_b);
        assert_eq!(result_a, result_b);
    }

    #[test]
    fn multiply_two_vec3s() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = multiply(&mut out, &vec_a, &vec_b);

        assert_eq!([4., 10., 18.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mul_two_vec3s() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = mul(&mut out, &vec_a, &vec_b);

        assert_eq!([4., 10., 18.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mul_is_equal_to_multiply() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result_a = multiply(&mut out, &vec_a, &vec_b);
        let result_b = mul(&mut out, &vec_a, &vec_b);
        assert_eq!(result_a, result_b);
    }

    #[test]
    fn divide_two_vec3s() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = divide(&mut out, &vec_a, &vec_b);

        assert_eq!([0.25, 0.4, 0.5], out);
        assert_eq!(result, out);
    }

    #[test]
    fn div_two_vec3s() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = div(&mut out, &vec_a, &vec_b);

        assert_eq!([0.25, 0.4, 0.5], out);
        assert_eq!(result, out);
    }

    #[test]
    fn div_is_equal_to_divide() { 
        let mut out =  [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result_a = divide(&mut out, &vec_a, &vec_b);
        let result_b = div(&mut out, &vec_a, &vec_b);
        assert_eq!(result_a, result_b);
    }

    #[test]
    fn ceil_of_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [core::f32::consts::E, PI, core::f32::consts::SQRT_2];

        let result = ceil(&mut out, &vec_a);

        assert_eq!([3., 4., 2.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn floor_of_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [core::f32::consts::E, PI, core::f32::consts::SQRT_2];

        let result = floor(&mut out, &vec_a);

        assert_eq!([2., 3., 1.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn min_of_two_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 3., 1.];
        let vec_b: Vec3 = [3., 1., 3.];

        let result = min(&mut out, &vec_a, &vec_b);

        assert_eq!([1., 1., 1.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn max_of_two_vec2() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 3., 1.];
        let vec_b: Vec3 = [3., 1., 3.];

        let result = max(&mut out, &vec_a, &vec_b);

        assert_eq!([3., 3., 3.], out); 
        assert_eq!(result, out);
    }
    
    #[test]
    fn round_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [core::f32::consts::E, PI, core::f32::consts::SQRT_2];

        let result = round(&mut out, &vec_a);

        assert_eq!([3., 3., 1.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn scale_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];

        let result = scale(&mut out, &vec_a, 2.);

        assert_eq!([2., 4., 6.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn scale_and_add_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = scale_and_add(&mut out, &vec_a, &vec_b, 0.5);

        assert_eq!([3., 4.5, 6.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn distance_between_vec3s() {  
        use super::super::common;

        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = distance(&vec_a, &vec_b);

        assert!(common::equals(result, 5.196152));
    }

    #[test]
    fn dist_between_vec3s() {  
        use super::super::common;

        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = dist(&vec_a, &vec_b);

        assert!(common::equals(result, 5.196152));
    }

    #[test]
    fn dist_is_equal_to_distance() {  
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result_a = distance(&vec_a, &vec_b);
        let result_b = dist(&vec_a, &vec_b);

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn squared_distance_between_vec2s() {  
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = squared_distance(&vec_a, &vec_b);

        assert_eq!(result, 27.);
    }

    #[test]
    fn sqr_dist_between_vec2s() {  
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = sqr_dist(&vec_a, &vec_b);

        assert_eq!(result, 27.);
    }

    #[test]
    fn sqr_dist_is_equal_to_squared_distance() {  
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result_a = squared_distance(&vec_a, &vec_b);
        let result_b = sqr_dist(&vec_a, &vec_b);

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn squared_length_vec3() {  
        let vec_a: Vec3 = [1., 2., 3.];

        let result = squared_length(&vec_a);

        assert_eq!(result, 14.);
    }

    #[test]
    fn sqr_len_vec3() {  
        let vec_a: Vec3 = [1., 2., 3.];

        let result = sqr_len(&vec_a);

        assert_eq!(result, 14.);
    }

    #[test]
    fn sqr_len_is_equal_to_sqr_dist() {  
        let vec_a: Vec3 = [1., 2., 3.];

        let result_a = squared_length(&vec_a);
        let result_b = sqr_len(&vec_a);

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn negate_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];

        let result = negate(&mut out, &vec_a);

        assert_eq!(out, [-1., -2., -3.]);
        assert_eq!(result, out);

    }

    #[test]
    fn invert_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];

        let result = inverse(&mut out, &vec_a);

        assert_eq!(out, [1., 0.5, 0.33333333333333]);
        assert_eq!(result, out);
    }

    #[test]
    fn normalize_vec3() {
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [5., 0., 0.];

        let result = normalize(&mut out, &vec_a);

        assert_eq!(out, [1., 0., 0.]);
        assert_eq!(result, out);
    }

    #[test]
    fn dot_product_of_two_vec3() {
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = dot(&vec_a, &vec_b);

        assert_eq!(result, 32.);
    }

    #[test]
    fn cross_product_of_two_vec3() {
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = cross(&mut out, &vec_a, &vec_b);

        assert_eq!(out, [-3., 6., -3.]);
        assert_eq!(result, out)
    }

    #[test]
    fn lerp_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = lerp(&mut out, &vec_a, &vec_b, 0.5);
        

        assert_eq!(out, [2.5, 3.5, 4.5]);
        assert_eq!(result, out);
    }

    #[test]
    fn hermite_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];
        let vec_c: Vec3 = [1., 2., 3.];
        let vec_d: Vec3 = [4., 5., 6.];

        let result = hermite(&mut out, &vec_a, &vec_b,
                                       &vec_c, &vec_d, 0.5);

        assert_eq!(out, [2.875, 3.875, 4.875]);
        assert_eq!(result, out);
    } 

    #[test]
    fn bezier_vec3() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];
        let vec_c: Vec3 = [1., 2., 3.];
        let vec_d: Vec3 = [4., 5., 6.];

        let result = bezier(&mut out, &vec_a, &vec_b,
                                       &vec_c, &vec_d, 0.5);

        assert_eq!(out, [2.5, 3.5, 4.5]);
        assert_eq!(result, out);
    } 
   
    #[test]
    fn random_vec3_no_scale() { 
        let mut out: Vec3 = [0., 0., 0.];
        let result = random(&mut out, None);
        
        assert!(out[0] >= -1_f32 && out[0] <= 1_f32);
        assert!(out[1] >= -1_f32 && out[1] <= 1_f32);
        assert_eq!(result, out);
    }

    #[test]
    fn random_vec3_scaled() { 
        let scale = 2_f32;
        let mut out: Vec3 = [0., 0., 0.]; 
        let result = random(&mut out, Some(scale));
        
        assert!(out[0] >= -1_f32 * scale && out[0] <= 1_f32 * scale);
        assert!(out[1] >= -1_f32 * scale && out[1] <= 1_f32 * scale);
        assert_eq!(result, out);
    }
    
    #[test]
    fn transform_mat4_to_vec3() {
        let mut out: Vec3 = [0., 0., 0.];
        let mat_r: Mat4 = [1., 0., 0., 0.,
                          0., 1., 0., 0., 
                          0., 0., 1., 0., 
                          0., 0., 0., 1.];
        let vec_a: Vec3 = [1., 2., 3.];
        
        let result = transform_mat4(&mut out, &vec_a, &mat_r);

        assert_eq!([1., 2., 3.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn transform_mat3_to_vec3() {
        let mut out: Vec3 = [0., 0., 0.];
        let mat_r: Mat3 = [1., 0., 0.,
                           0., 1., 0., 
                           0., 0., 1.];
        let vec_a: Vec3 = [1., 2., 3.];
        
        let result = transform_mat3(&mut out, &vec_a, &mat_r);

        assert_eq!([1., 2., 3.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn transform_quat_to_vec3() {
        let mut out: Vec3 = [0., 0., 0.];
        let quat_r: Quat = [0.18257418567011074, 0.3651483713402215, 0.5477225570103322, 0.730296742680443];
        let vec_a: Vec3 = [1., 2., 3.];
        
        let result = transform_quat(&mut out, &vec_a, &quat_r);

        assert_eq!([1., 2., 3.], out);
        assert_eq!(result, out);
    }
   
    #[test]
    fn rotate_vec3_x_same() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [0., 1., 0.];
        let vec_b: Vec3 = [0., 0., 0.];
        
        let result = rotate_x(&mut out, &vec_a, &vec_b, PI);

        assert!(equals(&out, &[0., -1., 0.]));
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_vec3_x_different() { 
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [2., 7., 0.];
        let vec_b: Vec3 = [2., 5., 0.];
        
        let result = rotate_x(&mut out, &vec_a, &vec_b, PI);

        assert!(equals(&out, &[2., 3., 0.]));
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_mat4_y_same() {
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [1., 0., 0.];
        let vec_b: Vec3 = [0., 0., 0.];
        
        let result = rotate_y(&mut out, &vec_a, &vec_b, PI);

        assert!(equals(&out, &[-1., 0., 0.]));
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_mat4_y_different() {
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [-2., 3., 10.];
        let vec_b: Vec3 = [-4., 3., 10.];
        
        let result = rotate_y(&mut out, &vec_a, &vec_b, PI);

        assert!(equals(&out, &[-6., 3., 10.]));
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_mat4_z_same() {
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [0., 1., 0.];
        let vec_b: Vec3 = [0., 0., 0.];
        
        let result = rotate_z(&mut out, &vec_a, &vec_b, PI);

        assert!(equals(&out, &[0., -1., 0.]));
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_mat4_z_different() {
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec3 = [0., 6., -5.];
        let vec_b: Vec3 = [0., 0., -5.];
        
        let result = rotate_z(&mut out, &vec_a, &vec_b, PI);

        assert!(equals(&out, &[-0., -6., -5.]));
        assert_eq!(result, out);
    }
   
    #[test]
    fn angle_of_vec3() { 
        use super::super::common;

        let vec_a: Vec3 = [1., 2., 3.];
        let vec_b: Vec3 = [4., 5., 6.];

        let result = angle(&vec_a, &vec_b);
        
        assert!(common::equals(result, 0.225726));
    }

    #[test]
    fn zero_out_vec3() { 
        let mut vec_a: Vec3 = [1., 2., 3.];

        let result = zero(&mut vec_a);
        
        assert_eq!(vec_a, [0., 0., 0.]);
        assert_eq!(result, vec_a);
    }

    #[test]
    fn get_vec3_string() { 
        let vec_a: Vec3 = [1., 2., 3.];
        
        let str_a = string(&vec_a);

        assert_eq!("vec3(1, 2, 3)".to_string(), str_a);
    }


    #[test]
    fn vec2_are_exact_equal() { 
        let vec_a: Vec3 = [0., 1., 2.];
        let vec_b: Vec3 = [0., 1., 2.];

        let r0 = exact_equals(&vec_a, &vec_b);

        assert!(r0);  
    }

    #[test]
    fn vec2s_are_not_exact_equal() { 
        let vec_a: Vec3 = [0., 1., 2.];
        let vec_b: Vec3 = [1., 2., 3.];

        let r0 = exact_equals(&vec_a, &vec_b);

        assert!(!r0);  
    }

    #[test]
    fn vec2s_are_equal() { 
        let vec_a: Vec3 = [0., 1., 2.];
        let vec_b: Vec3 = [0., 1., 2.];

        let r0 = equals(&vec_a, &vec_b);

        assert!(r0);  
    }

    #[test]
    fn vec2s_are_equal_enough() { 
        let vec_a: Vec3 = [0., 1., 2.];
        let vec_b: Vec3 = [1_f32*10_f32.powi(-16), 1., 2.];

        let r0 = equals(&vec_a, &vec_b);

        assert!(r0);  
    }

    #[test]
    fn vec2s_are_not_equal() { 
        let vec_a: Vec3 = [0., 1., 2.];
        let vec_b: Vec3 = [1., 2., 3.];

        let r0 = equals(&vec_a, &vec_b);

        assert!(!r0);  
    }

}
