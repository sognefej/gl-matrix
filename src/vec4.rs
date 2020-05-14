//! 4 Dimensional Vector
 
use super::common::{Vec4, Mat4, Quat, random_f32, hypot, EPSILON};

/// Creates a new, empty vec3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn create() -> Vec4 {
    let out: Vec4 = [0_f32; 4];

    out
}

/// Creates a new vec4 initialized with values from an existing vector.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn clone(a: &Vec4) -> Vec4 {
    let mut out: Vec4 = [0_f32; 4];

    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];

    out
}

/// Creates a new vec4 initialized with the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn from_values(x: f32, y: f32, z: f32, w: f32) -> Vec4{
    let mut out: Vec4 = [0_f32; 4];

    out[0] = x;
    out[1] = y;
    out[2] = z;
    out[3] = w;

    out
}

/// Copy the values from one vec4 to another.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn copy(out: &mut Vec4, a: &Vec4) -> Vec4 {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];

    *out
}

/// Set the components of a vec4 to the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn set(out: &mut Vec4, x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    out[0] = x;
    out[1] = y;
    out[2] = z;
    out[3] = w;

    *out
}

/// Adds two vec4's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn add(out: &mut Vec4, a: &Vec4, b: &Vec4) -> Vec4 {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];

    *out
}

/// Subtracts vector b from vector a.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn subtract(out: &mut Vec4, a: &Vec4, b: &Vec4) -> Vec4 {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    out[3] = a[3] - b[3];

    *out
}

/// Multiplies two vec4's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn multiply(out: &mut Vec4, a: &Vec4, b: &Vec4) -> Vec4 {
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
    out[2] = a[2] * b[2];
    out[3] = a[3] * b[3];

    *out
}

/// Divides two vec4's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn divide(out: &mut Vec4, a: &Vec4, b: &Vec4) -> Vec4 {
    out[0] = a[0] / b[0];
    out[1] = a[1] / b[1];
    out[2] = a[2] / b[2];
    out[3] = a[3] / b[3];

    *out
}

/// f32::ceil the components of a vec4.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn ceil(out: &mut Vec4, a: &Vec4) -> Vec4 {
    out[0] = f32::ceil(a[0]);
    out[1] = f32::ceil(a[1]);
    out[2] = f32::ceil(a[2]);
    out[3] = f32::ceil(a[3]);

    *out
}

/// f32::floor the components of a vec4.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn floor(out: &mut Vec4, a: &Vec4) -> Vec4 {
    out[0] = f32::floor(a[0]);
    out[1] = f32::floor(a[1]);
    out[2] = f32::floor(a[2]);
    out[3] = f32::floor(a[3]);

    *out
}

/// Returns the minimum of two vec4's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn min(out: &mut Vec4, a: &Vec4, b: &Vec4) -> Vec4 {
    out[0] = f32::min(a[0], b[0]);
    out[1] = f32::min(a[1], b[1]);
    out[2] = f32::min(a[2], b[2]);
    out[3] = f32::min(a[3], b[3]);

    *out
}

/// Returns the maximum of two vec4's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn max(out: &mut Vec4, a: &Vec4, b: &Vec4) -> Vec4 {
    out[0] = f32::max(a[0], b[0]);
    out[1] = f32::max(a[1], b[1]);
    out[2] = f32::max(a[2], b[2]);
    out[3] = f32::max(a[3], b[3]);

    *out
}

/// f32::round the components of a vec4.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn round(out: &mut Vec4, a: &Vec4) -> Vec4 {
    out[0] = f32::round(a[0]);
    out[1] = f32::round(a[1]);
    out[2] = f32::round(a[2]);
    out[3] = f32::round(a[3]);

    *out
}

/// Scales a vec4 by a scalar number.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn scale(out: &mut Vec4, a: &Vec4, b: f32) -> Vec4 {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
    out[3] = a[3] * b;

    *out
}

/// Adds two vec4's after scaling the second operand by a scalar value.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn scale_and_add(out: &mut Vec4, a: &Vec4, b: &Vec4, scale: f32) -> Vec4 {
    out[0] = a[0] + (b[0] * scale);
    out[1] = a[1] + (b[1] * scale);
    out[2] = a[2] + (b[2] * scale);
    out[3] = a[3] + (b[3] * scale);

    *out
}

/// Calculates the euclidian distance between two vec4's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn distance(a: &Vec4, b: &Vec4) -> f32 {
    let mut dist: Vec4 = [0_f32; 4];
   
    dist[0] = b[0] - a[0]; // x
    dist[1] = b[1] - a[1]; // y 
    dist[2] = b[2] - a[2]; // z
    dist[3] = b[3] - a[3]; // w

    hypot(&dist.to_vec())
}

/// Calculates the squared euclidian distance between two vec4's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn squared_distance(a: &Vec4, b: &Vec4) -> f32 {
    let x = b[0] - a[0];
    let y = b[1] - a[1];
    let z = b[2] - a[2];
    let w = b[3] - a[3];

    x*x + y*y + z*z + w*w
}

/// Calculates the length of a vec4.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn length(a: &Vec4) -> f32 {
    let mut len: Vec4  = [0_f32; 4];
    
    len[0] = a[0]; // x
    len[1] = a[1]; // y 
    len[2] = a[2]; // z
    len[3] = a[3]; // w

    hypot(&len.to_vec())
}

/// calculates the squared length of a vec4.
/// 
/// [glmatrix documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn squared_length(a: &[f32]) -> f32 {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    let w = a[3];

    x*x + y*y + z*z + w*w
}

/// Negates the components of a vec4.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn negate(out: &mut Vec4, a: &Vec4) -> Vec4 {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];
    out[3] = -a[3];

    *out
}

/// Returns the inverse of the components of a vec4.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn inverse(out: &mut Vec4, a: &Vec4) -> Vec4 {
    out[0] = 1.0 / a[0];
    out[1] = 1.0 / a[1];
    out[2] = 1.0 / a[2];
    out[3] = 1.0 / a[3];

    *out
}

/// Normalize a vec4.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn normalize(out: &mut Vec4, a: &Vec4) -> Vec4 {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    let w = a[3];

    let mut len = x*x + y*y + z*z + w*w;
    if len > 0_f32 {
        len = 1_f32 / f32::sqrt(len);
    }

    out[0] = x * len;
    out[1] = y * len;
    out[2] = z * len;
    out[3] = w * len;

    *out
}

/// Calculates the dot product of two vec4's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn dot(a: &Vec4, b: &Vec4) -> f32 { 
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Returns the cross-product of three vectors in a 4-dimensional space.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn cross (out: &mut Vec4, u: &Vec4, v: &Vec4, w: &Vec4) -> Vec4 {
    let a = (v[0] * w[1]) - (v[1] * w[0]);
    let b = (v[0] * w[2]) - (v[2] * w[0]);
    let c = (v[0] * w[3]) - (v[3] * w[0]);
    let d = (v[1] * w[2]) - (v[2] * w[1]);
    let e = (v[1] * w[3]) - (v[3] * w[1]);
    let f = (v[2] * w[3]) - (v[3] * w[2]);

    let g = u[0];
    let h = u[1];
    let i = u[2];
    let j = u[3];

    out[0] = (h * f) - (i * e) + (j * d);
    out[1] = -(g * f) + (i * c) - (j * b);
    out[2] = (g * e) - (h * c) + (j * a);
    out[3] = -(g * d) + (h * b) - (i * a);

    *out
}

/// Performs a linear interpolation between two vec4's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn lerp(out: &mut Vec4, a: &Vec4, b: &Vec4, t: f32) -> Vec4 {
    let ax = a[0];
    let ay = a[1];
    let az = a[2];
    let aw = a[3];

    out[0] = ax + t * (b[0] - ax);
    out[1] = ay + t * (b[1] - ay);
    out[2] = az + t * (b[2] - az);
    out[3] = aw + t * (b[3] - aw);

    *out
}

/// Generates a random vector with the given scale.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn random(out: &mut Vec4, scale: Option<f32>) -> Vec4 {
    let scale = match scale { 
        Some(scale) => scale, 
        None => 1_f32, 
    };
    // Marsaglia, George. Choosing a Point from the Surface of a
    // Sphere. Ann. f32:: Statist. 43 (1972), no. 2, 645--646.
    // http://projecteuclid.org/euclid.aoms/1177692644;
    let mut v1 = 0_f32;
    let mut v2 = 0_f32;
    let mut v3 = 0_f32;
    let mut v4 = 0_f32;

    let mut s1 = 2_f32;
    let mut s2 = 2_f32;

    while s1 > 1_f32 {
        v1 = random_f32() * 2. - 1.;
        v2 = random_f32() * 2. - 1.;
        s1 = v1 * v1 + v2 * v2;
    }
    while s2 > 1_f32 {
        v3 = random_f32() * 2. - 1.;
        v4 = random_f32() * 2. - 1.;
        s2 = v3 * v3 + v4 * v4;
    } 

    let d = f32::sqrt((1_f32 - s1) / s2);

    out[0] = scale * v1;
    out[1] = scale * v2;
    out[2] = scale * v3 * d;
    out[3] = scale * v4 * d;

    *out
}

/// Transforms the vec4 with a mat4.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn transform_mat4(out: &mut Vec4, a: &Vec4, m: &Mat4) -> Vec4 {
    let x = a[0];
    let y = a[1];
    let z = a[2];
    let w = a[3];

    out[0] = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
    out[1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
    out[2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
    out[3] = m[3] * x + m[7] * y + m[11] * z + m[15] * w;

    *out
}

/// Transforms the vec4 with a quat.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn transform_quat(out: &mut Vec4, a: &Vec4 , q: &Quat) -> Vec4 {
    let x = a[0];
    let y = a[1];
    let z = a[2];

    let qx = q[0];
    let qy = q[1];
    let qz = q[2];
    let qw = q[3];

    // calculate quat * vec
    let ix = qw * x + qy * z - qz * y;
    let iy = qw * y + qz * x - qx * z;
    let iz = qw * z + qx * y - qy * x;
    let iw = -qx * x - qy * y - qz * z;

    // calculate result * inverse quat
    out[0] = ix * qw + iw * -qx + iy * -qz - iz * -qy;
    out[1] = iy * qw + iw * -qy + iz * -qx - ix * -qz;
    out[2] = iz * qw + iw * -qz + ix * -qy - iy * -qx;
    out[3] = a[3];

    *out
}

/// Set the components of a vec4 to zero.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn zero(out: &mut Vec4) -> Vec4 {
    out[0] = 0.0;
    out[1] = 0.0;
    out[2] = 0.0;
    out[3] = 0.0;

    *out
}

/// Returns a string representation of a vector.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn string(a: &Vec4) -> String {
    let a0 = ["vec4(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = a[2].to_string(); 
    let a3 = [a[3].to_string(), ")".to_string()].join("");

    [a0, a1, a2, a3].join(", ")
}

/// Returns whether or not the vectors have exactly the same elements in the same position (when compared with ==).
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn exact_equals(a: &Vec4, b: &Vec4) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
}

/// Returns whether or not the vectors have approximately the same elements in the same position..
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn equals(a: &Vec4, b: &Vec4) -> bool {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];

    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3];

    f32::abs(a0 - b0) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a0), f32::abs(b0))) && 
    f32::abs(a1 - b1) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a1), f32::abs(b1))) && 
    f32::abs(a2 - b2) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a2), f32::abs(b2))) && 
    f32::abs(a3 - b3) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a3), f32::abs(b3)))
}

/// Alias for vec4::subtract.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn sub(out: &mut Vec4, a: &Vec4, b: &Vec4) -> Vec4 {
    subtract(out, a, b)
}

/// Alias for vec4::multiply.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn mul(out: &mut Vec4, a: &Vec4, b: &Vec4) -> Vec4 {
    multiply(out, a, b)
}

/// Alias for vec4::divide.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn div(out: &mut Vec4, a: &Vec4, b: &Vec4) -> Vec4 {
    divide(out, a, b)
}

/// Alias for vec4::distance.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn dist(a: &Vec4, b: &Vec4) -> f32 {
    distance(a, b)
}

/// Alias for vec4::squared_distance.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn sqr_dist(a: &Vec4, b: &Vec4) -> f32 {
    squared_distance(a, b)
}

/// Alias for vec4::length.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn len(a: &Vec4) -> f32 {
    length(a)
}

/// Alias for vec4::squred_length.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec4.js.html)
pub fn sqr_len(a: &Vec4) -> f32 {
    squared_length(a)
}

// /**
//  * Perform some operation over an array of vec4s.
//  *
//  * @param {Array} a the array of vectors to iterate over
//  * @param {Number} stride Number of elements between the start of each vec4. If 0 assumes tightly packed
//  * @param {Number} offset Number of elements to skip at the beginning of the array
//  * @param {Number} count Number of vec4s to iterate over. If 0 iterates over entire array
//  * @param {fn} fn fn to call for each vector in the array
//  * @param {Object} [arg] additional argument to pass to fn
//  * @returns {Array} a
//  * @fn
//  */

// pub fn for_each(a: &[f32], stride: f32, offset: f32, count: f32, f: fn(), arg){
// }
// pub const forEach = (fn() {
//   let vec = create();
//   return fn(a, stride, offset, count, fn, arg) {
//     let i, l;
//     if(!stride) {
//       stride = 4;
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
//       vec[0] = a[i]; vec[1] = a[i+1]; vec[2] = a[i+2]; vec[3] = a[i+3];
//       fn(vec, vec, arg);
//       a[i] = vec[0]; a[i+1] = vec[1]; a[i+2] = vec[2]; a[i+3] = vec[3];
//     }

//     return a;
//   };
// })();


#[cfg(test)]
mod tests { 
    use super::*;

    #[test]
    fn create_a_vec4() { 
        let out = create();

        assert_eq!([0., 0., 0., 0.], out);
    }

    #[test]
    fn clone_a_vec4() { 
        let vec4_a: Vec4 = [1., 2., 3., 4.];

        let out = clone(&vec4_a);

        assert_eq!([1., 2., 3., 4.], out);
    }

    #[test] 
    fn create_vec4_from_values() { 
        let out = from_values(1., 2., 3., 4.);

        assert_eq!([1., 2., 3., 4.], out);
    }

    #[test]
    fn copy_values_from_a_vec4_it_another() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec4_a: Vec4 = [1., 2., 3., 4.];

        let result = copy(&mut out, &vec4_a);

        assert_eq!([1., 2., 3., 4.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn set_vec4_with_values() { 
        let mut out: Vec4 = [0., 0., 0., 0.];

        let result = set(&mut out, 1., 2., 3., 4.);

        assert_eq!([1., 2., 3., 4.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn add_two_vec4s() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec4_a: Vec4 = [1., 2., 3., 4.];
        let vec4_b: Vec4 = [5., 6., 7., 8.];

        let result = add(&mut out, &vec4_a, &vec4_b);

        assert_eq!([6., 8., 10., 12.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn subtract_two_vec4s() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec4_a: Vec4 = [1., 2., 3., 4.];
        let vec4_b: Vec4 = [5., 6., 7., 8.];

        let result = subtract(&mut out, &vec4_a, &vec4_b);

        assert_eq!([-4., -4., -4., -4.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_two_vec4s() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec4_a: Vec4 = [1., 2., 3., 4.];
        let vec4_b: Vec4 = [5., 6., 7., 8.];

        let result = sub(&mut out, &vec4_a, &vec4_b);

        assert_eq!([-4., -4., -4., -4.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_is_equal_to_subtract() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result_a = subtract(&mut out, &vec_a, &vec_b);
        let result_b = sub(&mut out, &vec_a, &vec_b);
        assert_eq!(result_a, result_b);
    }
   
    #[test]
    fn multiply_two_vec4s() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result = multiply(&mut out, &vec_a, &vec_b);

        assert_eq!([5., 12., 21., 32.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mul_two_vec4s() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result = mul(&mut out, &vec_a, &vec_b);

        assert_eq!([5., 12., 21., 32.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mul_is_equal_to_multiply() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result_a = multiply(&mut out, &vec_a, &vec_b);
        let result_b = mul(&mut out, &vec_a, &vec_b);
        assert_eq!(result_a, result_b);
    }

    #[test]
    fn divide_two_vec4s() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result = divide(&mut out, &vec_a, &vec_b);

        assert_eq!([0.2, 0.33333334, 0.42857143, 0.5], out);
        assert_eq!(result, out);
    }

    #[test]
    fn div_two_vec4s() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result = div(&mut out, &vec_a, &vec_b);

        assert_eq!([0.2, 0.33333334, 0.42857143, 0.5], out);
        assert_eq!(result, out);
    }

    #[test]
    fn div_is_equal_to_divide() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result_a = divide(&mut out, &vec_a, &vec_b);
        let result_b = div(&mut out, &vec_a, &vec_b);
        assert_eq!(result_a, result_b);
    }

    #[test]
    fn ceil_of_vec4() {
        use super::super::common::PI; 

        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [core::f32::consts::E, PI, 
                           core::f32::consts::SQRT_2,
                           core::f32::consts::FRAC_1_SQRT_2];

        let result = ceil(&mut out, &vec_a);

        assert_eq!([3., 4., 2., 1.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn floor_of_vec4() { 
        use super::super::common::PI; 

        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [core::f32::consts::E, PI, 
                           core::f32::consts::SQRT_2,
                           core::f32::consts::FRAC_1_SQRT_2];

        let result = floor(&mut out, &vec_a);

        assert_eq!([2., 3., 1., 0.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn min_of_two_vec4() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 3., 1., 3.];
        let vec_b: Vec4 = [3., 1., 3., 1.];

        let result = min(&mut out, &vec_a, &vec_b);

        assert_eq!([1., 1., 1., 1.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn max_of_two_vec4() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 3., 1., 3.];
        let vec_b: Vec4 = [3., 1., 3., 1.];

        let result = max(&mut out, &vec_a, &vec_b);

        assert_eq!([3., 3., 3., 3.], out); 
        assert_eq!(result, out);
    }
    
    #[test]
    fn round_vec4() { 
        use super::super::common::PI; 

        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [core::f32::consts::E, PI, 
                           core::f32::consts::SQRT_2,
                           core::f32::consts::FRAC_1_SQRT_2];

        let result = round(&mut out, &vec_a);

        assert_eq!([3., 3., 1., 1.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn scale_vec4() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];

        let result = scale(&mut out, &vec_a, 2.);

        assert_eq!([2., 4., 6., 8.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn scale_and_add_vec4() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result = scale_and_add(&mut out, &vec_a, &vec_b, 0.5);

        assert_eq!([3.5, 5., 6.5, 8.], out); 
        assert_eq!(result, out);
    }
    #[test]
    fn distance_between_vec4s() {  
        use super::super::common;

        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result = distance(&vec_a, &vec_b);

        assert!(common::equals(result, 8.));
    }

    #[test]
    fn dist_between_vec4s() {  
        use super::super::common;

        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result = dist(&vec_a, &vec_b);

        assert!(common::equals(result, 8.));
    }

    #[test]
    fn dist_is_equal_to_distance() {  
        let vec_a: Vec4 = [1., 2., 3., 4.];
        let vec_b: Vec4 = [5., 6., 7., 8.];

        let result_a = distance(&vec_a, &vec_b);
        let result_b = dist(&vec_a, &vec_b);

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn squared_length_vec4() {  
        let vec_a: Vec4 = [1., 2., 3., 4.];

        let result = squared_length(&vec_a);

        assert_eq!(result, 30.);
    }

    #[test]
    fn sqr_len_vec4() {  
        let vec_a: Vec4 = [1., 2., 3., 4.];

        let result = sqr_len(&vec_a);

        assert_eq!(result, 30.);
    }

    #[test]
    fn sqr_len_is_equal_to_sqr_dist() {  
        let vec_a: Vec4 = [1., 2., 3., 4.];

        let result_a = squared_length(&vec_a);
        let result_b = sqr_len(&vec_a);

        assert_eq!(result_a, result_b);
    }
    
    #[test]
    fn length_of_vec4() {
        let vec4_a: Vec4 = [1., 2., 3., 4.];

        let out = length(&vec4_a);

        // they get 5.477225
        assert_eq!(5.477226, out);
    }

    #[test]
    fn len_of_vec4() { 
        let vec4_a: Vec4 = [1., 2., 3., 4.];

        let out = len(&vec4_a);

        // they get 5.477225
        assert_eq!(5.477226, out);
    }

    #[test]
    fn length_is_equal_to_len() {
        let vec4_a: Vec4 = [1., 2., 3., 4.];

        let out_a = length(&vec4_a);
        let out_b = len(&vec4_a);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn squared_length_of_vec4() { 
        let vec4_a: Vec4 = [1., 2., 3., 4.];

        let out = squared_length(&vec4_a);

        // they get 5.477225
        assert_eq!(30_f32, out);
    }

    #[test]
    fn sqr_len_of_vec4() { 
        let vec4_a: Vec4 = [1., 2., 3., 4.];

        let out = sqr_len(&vec4_a);

        // they get 5.477225
        assert_eq!(30_f32, out);
    }

    #[test]
    fn squared_length_is_equal_to_sqr_len() { 
        let vec4_a: Vec4 = [1., 2., 3., 4.];

        let out_a = squared_length(&vec4_a);
        let out_b = sqr_len(&vec4_a);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn negate_vec4() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];

        let result = negate(&mut out, &vec_a);

        assert_eq!(out, [-1., -2., -3., -4.]);
        assert_eq!(result, out);

    }

    #[test]
    fn invert_vec4() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 2., 3., 4.];

        let result = inverse(&mut out, &vec_a);

        assert_eq!(out, [1., 0.5, 0.33333333333333, 0.25]);
        assert_eq!(result, out);
    }

    #[test]
    fn normalize_vec4() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec4_a: Vec4 = [5., 0., 0., 0.];

        let result = normalize(&mut out, &vec4_a);

        assert_eq!([1., 0., 0., 0.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn dot_vec4() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec4_a: Vec4 = [1., 2., 3., 4.];

        let result = scale(&mut out, &vec4_a, 2_f32);

        assert_eq!([2., 4., 6., 8.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn cross_product_of_two_vec4() {
        let mut out: Vec4= [0., 0., 0., 0.];
        let vec_a: Vec4 = [1., 0., 0., 0.];
        let vec_b: Vec4 = [0., 1., 0., 0.];
        let vec_c: Vec4 = [0., 0., 1., 0.];

        let result = cross(&mut out, &vec_a, &vec_b, &vec_c);

        assert_eq!(out, [0., 0., 0., -1.]);
        assert_eq!(result, out)
    }

    #[test]
    fn lerp_vec4() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let vec4_a: Vec4 = [1., 2., 3., 4.];
        let vec4_b: Vec4 = [5., 6., 7., 8.];

        let result = add(&mut out, &vec4_a, &vec4_b);

        assert_eq!([6., 8., 10., 12.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn random_vec3_no_scale() { 
        let mut out: Vec4 = [0., 0., 0., 0.];
        let result = random(&mut out, None);
        
        assert!(out[0] >= -1_f32 && out[0] <= 1_f32);
        assert!(out[1] >= -1_f32 && out[1] <= 1_f32);
        assert_eq!(result, out);
    }

    #[test]
    fn random_vec3_scaled() { 
        let scale = 2_f32;
        let mut out: Vec4 = [0., 0., 0., 0.]; 
        let result = random(&mut out, Some(scale));
        
        assert!(out[0] >= -1_f32 * scale && out[0] <= 1_f32 * scale);
        assert!(out[1] >= -1_f32 * scale && out[1] <= 1_f32 * scale);
        assert_eq!(result, out);
    }

    #[test]
    fn transform_mat4_to_vec4() {
        let mut out: Vec4 = [0., 0., 0., 0.];
        let mat_r: Mat4 = [1., 0., 0., 0.,
                          0., 1., 0., 0., 
                          0., 0., 1., 0., 
                          0., 0., 0., 1.];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        
        let result = transform_mat4(&mut out, &vec_a, &mat_r);

        assert_eq!([1., 2., 3., 4.], out);
        assert_eq!(result, out);
    }
    #[test]
    fn transform_quat_to_vec3() {
        let mut out: Vec4 = [0., 0., 0., 0.];
        let quat_r: Quat = [0.18257418567011074, 0.3651483713402215, 0.5477225570103322, 0.730296742680443];
        let vec_a: Vec4 = [1., 2., 3., 4.];
        
        let result = transform_quat(&mut out, &vec_a, &quat_r);

        assert_eq!([1., 2., 3., 4.], out);
        assert_eq!(result, out);
    }
   
    #[test]
    fn zero_out_vec3() { 
        let mut vec_a: Vec4 = [1., 2., 3., 4.];

        let result = zero(&mut vec_a);
        
        assert_eq!(vec_a, [0., 0., 0., 0.]);
        assert_eq!(result, vec_a);
    }

    #[test]
    fn get_vec4_string() { 
        let vec_a: Vec4 = [1., 2., 3., 4.];
        
        let str_a = string(&vec_a);

        assert_eq!("vec4(1, 2, 3, 4)".to_string(), str_a);
    }
    
    #[test]
    fn vec4_are_exact_equal() { 
        let vec4_a: Vec4 = [0., 1., 2., 3.];
        let vec4_b: Vec4 = [0., 1., 2., 3.];

        let r0 = exact_equals(&vec4_a, &vec4_b);

        assert!(r0);  
    }

    #[test]
    fn vec4_are_not_exact_equal() { 
        let vec4_a: Vec4 = [0., 1., 2., 3.];
        let vec4_b: Vec4 = [1., 2., 3., 4.];

        let r0 = exact_equals(&vec4_a, &vec4_b);

        assert!(!r0); 
    }

    #[test]
    fn vec4_are_equal() { 
        let vec4_a: Vec4 = [0., 1., 2., 3.];
        let vec4_b: Vec4 = [0., 1., 2., 3.];

        let r0 = equals(&vec4_a, &vec4_b);

        assert!(r0);  
    }

    #[test]
    fn vec4_are_equal_enough() { 
        let vec4_a: Vec4 = [0., 1., 2., 3.];
        let vec4_b: Vec4 = [1_f32*10_f32.powi(-16), 1., 2., 3.];

        let r0 = equals(&vec4_a, &vec4_b);

        assert!(r0);  
    }

    #[test]
    fn vec4_are_not_equal() { 
        let vec4_a: Vec4 = [0., 1., 2., 3.];
        let vec4_b: Vec4 = [1., 2., 3., 4.];

        let r0 = equals(&vec4_a, &vec4_b);

        assert!(!r0);  
    }
}