//! 2 Dimensional Vector

use super::common::{Vec2, Vec3, Mat2, Mat2d, Mat3, Mat4, hypot, random_f32, EPSILON, PI};

/// Creates a new, empty vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn create() -> Vec2 {
    let out: Vec2 = [0_f32; 2];

    out
}

/// Creates a new vec2 initialized with values from an existing vector.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn clone(a: &Vec2) -> Vec2 {
    let mut out: Vec2 = [0_f32; 2];

    out[0] = a[0];
    out[1] = a[1];

    out
}

/// Creates a new vec2 initialized with the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn from_values(x: f32, y: f32) -> Vec2 {
    let mut out: Vec2 = [0_f32; 2];

    out[0] = x;
    out[1] = y;

    out
}

/// Creates a new vec2 initialized with the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn copy(out: &mut Vec2, a: &Vec2) -> Vec2 {
    out[0] = a[0];
    out[1] = a[1];

    *out
}

/// Set the components of a vec2 to the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn set(out: &mut Vec2, x: f32, y: f32) -> Vec2 {
    out[0] = x;
    out[1] = y;

    *out
}

/// Adds two vec2's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn add(out: &mut Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];

    *out
}

/// Subtracts vector b from vector a.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn subtract(out: &mut Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];

    *out
}

/// Multiplies two vec2's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn multiply(out: &mut Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];

    *out
}

/// Divides two vec2's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn divide(out: &mut Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
    out[0] = a[0] / b[0];
    out[1] = a[1] / b[1];

    *out
}

/// f32::ceil the components of a vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn ceil(out: &mut Vec2, a: &Vec2) -> Vec2 {
    out[0] = f32::ceil(a[0]);
    out[1] = f32::ceil(a[1]);

    *out
}

/// f32::floor the components of a vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn floor(out: &mut Vec2, a: &Vec2) -> Vec2 {
    out[0] = f32::floor(a[0]);
    out[1] = f32::floor(a[1]);

    *out
}

/// Returns the minimum of two vec2's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn min(out: &mut Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
    out[0] = f32::min(a[0], b[0]);
    out[1] = f32::min(a[1], b[1]);

    *out
}

/// Returns the maximum of two vec2's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn max(out: &mut Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
    out[0] = f32::max(a[0], b[0]);
    out[1] = f32::max(a[1], b[1]);

    *out
}

/// f32::round the components of a vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn round(out: &mut Vec2, a: &Vec2) -> Vec2 {
    out[0] = f32::round(a[0]);
    out[1] = f32::round(a[1]);

    *out
}

/// Scales a vec2 by a scalar number.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn scale(out: &mut Vec2, a: &Vec2, b: f32) -> Vec2 {
    out[0] = a[0] * b;
    out[1] = a[1] * b;

    *out
}

/// Adds two vec2's after scaling the second operand by a scalar value.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn scale_and_add(out: &mut Vec2, a: &Vec2, b: &Vec2, scale: f32) -> Vec2 {
    out[0] = a[0] + (b[0] * scale);
    out[1] = a[1] + (b[1] * scale);

    *out
}

/// Calculates the euclidian distance between two vec2's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn distance(a: &Vec2, b: &Vec2) -> f32 {
    let mut dist: Vec2 = [0_f32; 2];
    
    dist[0] = b[0] - a[0]; // x
    dist[1] = b[1] - a[1]; // y

    hypot(&dist.to_vec())
}

/// Calculates the squared euclidian distance between two vec2's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn squared_distance(a: &Vec2, b: &Vec2) -> f32 {
    let x = b[0] - a[0];
    let y = b[1] - a[1];

    x*x + y*y
}

/// Calculates the length of a vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn length(a: &Vec2) -> f32 {
    let mut len: Vec2 = [0_f32; 2];
    
    len[0] = a[0]; // x
    len[1] = a[1]; // y

    hypot(&len.to_vec())
}

/// Calculates the squared length of a vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn squared_length(a: &Vec2) -> f32 {
    let x = a[0];
    let y = a[1];

    x*x + y*y
}

/// Negates the components of a vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn negate(out: &mut Vec2, a: &Vec2) -> Vec2 {
    out[0] = -a[0];
    out[1] = -a[1];

    *out
}

/// Returns the inverse of the components of a vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn inverse(out: &mut Vec2, a: &Vec2) -> Vec2 {
    out[0] = 1.0 / a[0];
    out[1] = 1.0 / a[1];

    *out
}

/// Normalize a vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn normalize(out: &mut Vec2, a: &Vec2) -> Vec2 {
    let x = a[0];
    let y = a[1];

    let mut len = x*x + y*y;

    if len > 0_f32 {
        //TODO: evaluate use of glm_invsqrt here?
        len = 1_f32 / f32::sqrt(len);
    }    

    out[0] = a[0] * len;
    out[1] = a[1] * len;

    *out
}

/// Calculates the dot product of two vec2's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn dot(a: &Vec2, b: &Vec2) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

/// Computes the cross product of two vec2's.
/// Note that the cross product must by definition produce a 3D vector.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn cross(out: &mut Vec3, a: &Vec2, b: &Vec2) -> Vec3 {
    let z = a[0] * b[1] - a[1] * b[0];

    out[0] = 0.; 
    out[1] = 0.;
    out[2] = z;

    *out
}

/// Performs a linear interpolation between two vec2's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn lerp(out: &mut Vec2, a: &Vec2, b: &Vec2, t: f32) -> Vec2 {
    let ax = a[0];
    let ay = a[1];

    out[0] = ax + t * (b[0] - ax);
    out[1] = ay + t * (b[1] - ay);

    *out
}

/// Generates a random vector with the given scale.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn random(out: &mut Vec2, scale: Option<f32>) -> Vec2 {
    let scale = match scale { 
        Some(scale) => scale, 
        None => 1_f32, 
    };

    let r = random_f32() * 2.0 * PI; 

    out[0] = f32::cos(r) * scale;
    out[1] = f32::sin(r) * scale;

    *out
}

/// Transforms the vec2 with a mat2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn transform_mat2(out: &mut Vec2, a: &Vec2, m: &Mat2) -> Vec2 {
    let x = a[0];
    let y = a[1];

    out[0] = m[0] * x + m[2] * y;
    out[1] = m[1] * x + m[3] * y;

    *out
}

/// Transforms the vec2 with a mat2d.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn transform_mat2d(out: &mut Vec2, a: &Vec2, m: &Mat2d) -> Vec2 {
    let x = a[0];
    let y = a[1];

    out[0] = m[0] * x + m[2] * y + m[4];
    out[1] = m[1] * x + m[3] * y + m[5];

    *out
}

/// Transforms the vec2 with a mat3.
/// 3rd vector component is implicitly '1'.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn transform_mat3(out: &mut Vec2, a: &Vec2, m: &Mat3) -> Vec2 {
    let x = a[0];
    let y = a[1];

    out[0] = m[0] * x + m[3] * y + m[6];
    out[1] = m[1] * x + m[4] * y + m[7];

    *out
}

/// Transforms the vec2 with a mat4.
/// 3rd vector component is implicitly '0'.
/// 4th vector component is implicitly '1'.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn transform_mat4(out: &mut Vec2, a: &Vec2, m: &Mat4) -> Vec2 {
    let x = a[0];
    let y = a[1];

    out[0] = m[0] * x + m[4] * y + m[12];
    out[1] = m[1] * x + m[5] * y + m[13];

    *out
}

/// Rotate a 2D vector.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn rotate(out: &mut Vec2, a: &Vec2, b: &Vec2, c: f32) -> Vec2 {
    //Translate point to the origin
    let p0 = a[0] - b[0];
    let p1 = a[1] - b[1];

    let sin_c = f32::sin(c);
    let cos_c = f32::cos(c);
    
    //perform rotation and translate to correct position
    out[0] = p0 * cos_c - p1 * sin_c + b[0];
    out[1] = p0 * sin_c + p1 * cos_c + b[1];

    *out
}

/// Get the angle between two 2D vectors.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
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

/// Set the components of a vec2 to zero.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn zero(out: &mut Vec2) -> Vec2 {
    out[0] = 0.0;
    out[1] = 0.0;

    *out
}

/// Returns a string representation of a vector.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn string(a: &Vec2) -> String {
    let a0 = ["vec2(".to_string(), a[0].to_string()].join("");
    let a1 = [a[1].to_string(), ")".to_string()].join("");

    [a0, a1].join(", ")
}

/// Returns whether or not the vectors exactly have the same elements in the same position (when compared with ===).
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn exact_equals(a: &Vec2,  b: &Vec2) -> bool {
    a[0] == b[0] && a[1] == b[1]
}

/// Returns whether or not the vectors have approximately the same elements in the same position.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn equals(a: &Vec2, b: &Vec2) -> bool {
    let a0 = a[0];
    let a1 = a[1];
    let b0 = b[0];
    let b1 = b[1];

    f32::abs(a0 - b0) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a0), f32::abs(b0))) && 
    f32::abs(a1 - b1) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a1), f32::abs(b1)))
}

/// Alias for vec2::length
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn len(a: &Vec2) -> f32 {
    length(a)
}

/// Alias for vec2::subtract
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn sub(out: &mut Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
    subtract(out, a, b)
}

/// Alias for vec2::multiply
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn mul(out: &mut Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
    multiply(out, a, b)
}

/// Alias for vec2::divide}
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn div(out: &mut Vec2, a: &Vec2, b: &Vec2) -> Vec2 {
    divide(out, a, b)
}

/// Alias for vec2::distance
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn dist(a: &Vec2, b: &Vec2) -> f32 {
    distance(a, b)
}

/// Alias for vec2::squared_distance
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
pub fn sqr_dist(a: &Vec2, b: &Vec2) -> f32 {
    squared_distance(a, b)
}

/// Alias for vec2::squared_length
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/vec2.js.html)
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


#[cfg(test)] 
mod tests {
    use super::*; 

    #[test] 
    fn create_a_vec2() { 
        let out = create();
        
        assert_eq!([0., 0.], out);
    }

    #[test] 
    fn clone_a_vec2() {
        let vec_a: Vec2 = [1., 2.];
  
        let out = clone(&vec_a);
       
        assert_eq!(vec_a, out);
    }

    #[test]
    fn create_vec2_from_values() { 
        let out = from_values(1., 2.);

        assert_eq!([1., 2.], out);
    }

    #[test] 
    fn copy_values_from_a_vec2_to_another() {
        let mut out =  [0., 0.];
        let vec_a: Vec2 = [1., 2.];
   
        let result = copy(&mut out, &vec_a);
      
        assert_eq!(vec_a, out);
        assert_eq!(result, out);
    }
    
    #[test]
    fn set_vec2_with_values() { 
        let mut out: Vec2 = [0., 0.];
     
        let result = set(&mut out, 1., 2.);

        assert_eq!([1., 2.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn add_two_vec2s() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = add(&mut out, &vec_a, &vec_b);

        assert_eq!([4., 6.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn subtract_two_vec2s() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = subtract(&mut out, &vec_a, &vec_b);

        assert_eq!([-2., -2.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_two_mat2s() {  
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = sub(&mut out, &vec_a, &vec_b);

        assert_eq!([-2., -2.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_is_equal_to_subtract() {  
        let mut out_a: Vec2 = [0., 0.];
        let mut out_b: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        subtract(&mut out_a, &vec_a, &vec_b);
        sub(&mut out_b, &vec_a, &vec_b);

        assert_eq!(out_a, out_b);
    }
    #[test]
    fn multiply_two_vec2s() {  
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = multiply(&mut out, &vec_a, &vec_b);

        assert_eq!([3., 8.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn mul_two_mat2s() {  
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = mul(&mut out, &vec_a, &vec_b);

        assert_eq!([3., 8.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn mul_is_equal_to_multiply() {  
        let mut out_a: Vec2 = [0., 0.];
        let mut out_b: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        multiply(&mut out_a, &vec_a, &vec_b);
        mul(&mut out_b, &vec_a, &vec_b);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn divide_two_vec2s() {  
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = divide(&mut out, &vec_a, &vec_b);

        assert!(equals(&[0.3333333, 0.5], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn div_two_vec2s() {  
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = div(&mut out, &vec_a, &vec_b);

        assert!(equals(&[0.3333333, 0.5], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn div_is_equal_to_divide() {  
        let mut out_a: Vec2 = [0., 0.];
        let mut out_b: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        divide(&mut out_a, &vec_a, &vec_b);
        div(&mut out_b, &vec_a, &vec_b);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn ceil_of_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [core::f32::consts::E, PI];

        let result = ceil(&mut out, &vec_a);

        assert_eq!([3., 4.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn floor_of_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [core::f32::consts::E, PI];

        let result = floor(&mut out, &vec_a);

        assert_eq!([2., 3.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn min_of_two_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 4.];
        let vec_b: Vec2 = [3., 2.];

        let result = min(&mut out, &vec_a, &vec_b);

        assert_eq!([1., 2.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn max_of_two_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 4.];
        let vec_b: Vec2 = [3., 2.];

        let result = max(&mut out, &vec_a, &vec_b);

        assert_eq!([3., 4.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn round_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [core::f32::consts::E, PI];

        let result = round(&mut out, &vec_a);

        assert_eq!([3., 3.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn scale_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];

        let result = scale(&mut out, &vec_a, 2.);

        assert_eq!([2., 4.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn scale_and_add_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = scale_and_add(&mut out, &vec_a, &vec_b, 0.5);

        assert_eq!([2.5, 4.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn distance_between_vec2s() {  
        use super::super::common;

        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = distance(&vec_a, &vec_b);

        assert!(common::equals(result, 2.828427));
    }

    #[test]
    fn dist_between_vec2s() {  
        use super::super::common;

        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = dist(&vec_a, &vec_b);

        assert!(common::equals(result, 2.828427));
    }

    #[test]
    fn dist_is_equal_to_distance() {  
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result_a = distance(&vec_a, &vec_b);
        let result_b = dist(&vec_a, &vec_b);

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn squared_distance_between_vec2s() {  
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = squared_distance(&vec_a, &vec_b);

        assert_eq!(result, 8.);
    }

    #[test]
    fn sqr_dist_between_vec2s() {  
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = sqr_dist(&vec_a, &vec_b);

        assert_eq!(result, 8.);
    }

    #[test]
    fn sqr_dist_is_equal_to_squared_distance() {  
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result_a = squared_distance(&vec_a, &vec_b);
        let result_b = sqr_dist(&vec_a, &vec_b);

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn length_of_vec2() {
        use super::super::common;

        let vec_a: Vec2 = [1., 2.];

        let result = length(&vec_a);

        assert!(common::equals(result, 2.236067));
    }

    #[test]
    fn len_of_vec2() { 
        use super::super::common;

        let vec_a: Vec2 = [1., 2.];

        let result = len(&vec_a);

        assert!(common::equals(result, 2.236067));
    }

    #[test]
    fn length_is_equal_to_len() {
        let vec_a: Vec2 = [1., 2.];

        let result_a = length(&vec_a);
        let result_b = len(&vec_a);

        assert_eq!(result_a, result_b);
    }
    
    #[test]
    fn squared_length_vec2() {  
        let vec_a: Vec2 = [1., 2.];

        let result = squared_length(&vec_a);

        assert_eq!(result, 5.);
    }

    #[test]
    fn sqr_len_vec2() {  
        let vec_a: Vec2 = [1., 2.];

        let result = sqr_len(&vec_a);

        assert_eq!(result, 5.);
    }

    #[test]
    fn sqr_len_is_equal_to_sqr_dist() {  
        let vec_a: Vec2 = [1., 2.];

        let result_a = squared_length(&vec_a);
        let result_b = sqr_len(&vec_a);

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn negate_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];

        let result = negate(&mut out, &vec_a);

        assert_eq!(out, [-1., -2.]);
        assert_eq!(result, out);

    }

    #[test]
    fn invert_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];

        let result = inverse(&mut out, &vec_a);

        assert_eq!(out, [1., 0.5]);
        assert_eq!(result, out);
    }

    #[test]
    fn normalize_vec2() {
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [5., 0.];

        let result = normalize(&mut out, &vec_a);

        assert_eq!(out, [1., 0.]);
        assert_eq!(result, out);
    }

    #[test]
    fn dot_product_of_two_vec2() {
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = dot(&vec_a, &vec_b);

        assert_eq!(result, 11.);
    }

    #[test]
    fn cross_product_of_two_vec2() {
        let mut out: Vec3 = [0., 0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = cross(&mut out, &vec_a, &vec_b);

        assert_eq!(out, [0., 0., -2.]);
        assert_eq!(result, out)
    }

    #[test]
    fn lerp_vec2() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [1., 2.];
        let vec_b: Vec2 = [3., 4.];

        let result = lerp(&mut out, &vec_a, &vec_b, 0.5);
        

        assert_eq!(out, [2., 3.]);
        assert_eq!(result, out);
    }

    #[test]
    fn random_vec2_no_scale() { 
        let mut out: Vec2 = [0., 0.];
        let result = random(&mut out, None);
        
        assert!(out[0] >= -1_f32 && out[0] <= 1_f32);
        assert!(out[1] >= -1_f32 && out[1] <= 1_f32);
        assert_eq!(result, out);
    }

    #[test]
    fn random_vec2_scaled() { 
        let scale = 2_f32;
        let mut out: Vec2 = [0., 0.]; 
        let result = random(&mut out, Some(scale));
        
        assert!(out[0] >= -1_f32 * scale && out[0] <= 1_f32 * scale);
        assert!(out[1] >= -1_f32 * scale && out[1] <= 1_f32 * scale);
        assert_eq!(result, out);
    }

    #[test]
    fn transform_vec2_with_mat2() { 
        let mut out: Vec2 = [0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let vec_a: Vec2 = [1., 2.];

        let result = transform_mat2(&mut out, &vec_a, &mat_a);

        assert_eq!(out, [7., 10.]);
        assert_eq!(result, out);
    }

    #[test]
    fn transform_vec2_with_mat2d() { 
        let mut out: Vec2 = [0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let vec_a: Vec2 = [1., 2.];

        let result = transform_mat2d(&mut out, &vec_a, &mat_a);

        assert_eq!(out, [12., 16.]);
        assert_eq!(result, out);
    }

    #[test]
    fn transform_vec2_with_mat3() { 
        let mut out: Vec2 = [0., 0.];
        let mat_a: Mat3 = [1., 2., 3.,
                           4., 5., 6.,
                           7., 8., 9.];
        let vec_a: Vec2 = [1., 2.];

        let result = transform_mat3(&mut out, &vec_a, &mat_a);

        assert_eq!(out, [16., 20.]);
        assert_eq!(result, out);
    }

    #[test]
    fn transform_vec2_with_mat4() { 
        let mut out: Vec2 = [0., 0.];
        let mat_a: Mat4 = [1., 2., 3., 4.,
                           5., 6., 7., 8.,
                           9., 10., 11., 12.,
                           13., 14., 15., 16.];
        let vec_a: Vec2 = [1., 2.];

        let result = transform_mat4(&mut out, &vec_a, &mat_a);

        assert_eq!(out, [24., 28.]);
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_vec2_around_origin() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [0., 1.];
        let vec_b: Vec2 = [0., 0.];

        let result = rotate(&mut out, &vec_a, &vec_b, PI);
        
        assert!(equals(&out, &[0., -1.]));
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_vec2_arbitrary_origin() { 
        let mut out: Vec2 = [0., 0.];
        let vec_a: Vec2 = [6., -5.];
        let vec_b: Vec2 = [0., -5.];

        let result = rotate(&mut out, &vec_a, &vec_b, PI);
        
        assert!(equals(&out, &[-6., -5.]));
        assert_eq!(result, out);
    }

    #[test]
    fn angle_of_vec2() { 
        use super::super::common;

        let vec_a: Vec2 = [1., 0.];
        let vec_b: Vec2 = [1., 2.];

        let result = angle(&vec_a, &vec_b);
        
        assert!(common::equals(result, 1.107148));
    }

    #[test]
    fn zero_out_vec2() { 
        let mut vec_a: Vec2 = [1., 2.];

        let result = zero(&mut vec_a);
        
        assert_eq!(vec_a, [0., 0.]);
        assert_eq!(result, vec_a);
    }

    #[test]
    fn get_vec2_string() { 
        let vec_a: Vec2 = [1., 2.];
        
        let str_a = string(&vec_a);

        assert_eq!("vec2(1, 2)".to_string(), str_a);
    }


    #[test]
    fn vec2_are_exact_equal() { 
        let vec_a: Vec2 = [0., 1.];
        let vec_b: Vec2 = [0., 1.];

        let r0 = exact_equals(&vec_a, &vec_b);

        assert!(r0);  
    }

    #[test]
    fn vec2s_are_not_exact_equal() { 
        let vec_a: Vec2 = [0., 1.];
        let vec_b: Vec2 = [1., 2.];

        let r0 = exact_equals(&vec_a, &vec_b);

        assert!(!r0);  
    }

    #[test]
    fn vec2s_are_equal() { 
        let vec_a: Vec2 = [0., 1.];
        let vec_b: Vec2 = [0., 1.];

        let r0 = equals(&vec_a, &vec_b);

        assert!(r0);  
    }

    #[test]
    fn vec2s_are_equal_enough() { 
        let vec_a: Vec2 = [0., 1.];
        let vec_b: Vec2 = [1_f32*10_f32.powi(-16), 1.];

        let r0 = equals(&vec_a, &vec_b);

        assert!(r0);  
    }

    #[test]
    fn vec2s_are_not_equal() { 
        let vec_a: Vec2 = [0., 1.];
        let vec_b: Vec2 = [1., 2.];

        let r0 = equals(&vec_a, &vec_b);

        assert!(!r0);  
    }
}