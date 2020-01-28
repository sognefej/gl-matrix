//! 2x3 Matrix
//! 
//! A mat2d contains six elements defined as:
//! ```
//! use gl_matrix::common::Mat2d;
//! 
//! let a: f32 = 0.; 
//! let b: f32 = 0.; 
//! let c: f32 = 0.;
//! let d: f32 = 0.; 
//! let tx: f32 = 0.;
//! let ty: f32 = 0.;
//! 
//! let ma: Mat2d = [a, b, c,
//!                  d, tx, ty];
//! ```
//! This is a short form for the 3x3 matrix:
//! ```
//! use gl_matrix::common::Mat3;
//! 
//! let a: f32 = 0.; 
//! let b: f32 = 0.; 
//! let c: f32 = 0.;
//! let d: f32 = 0.; 
//! let tx: f32 = 0.;
//! let ty: f32 = 0.;
//! 
//! let mat: Mat3 = [a, b, 0.,
//!                  c, d, 0.,
//!                  tx, ty, 1.];
//! ```
//! The last column is ignored so the array is shorter and operations are faster.

use super::common::{Mat2d, Vec2, hypot, EPSILON};

/// Creates a new identity mat2d
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn create() -> Mat2d {
    let mut out: Mat2d = [0_f32; 6];
  
    out[0] = 1_f32;  
    out[3] = 1_f32;

    out
}

/// Creates a new mat2d initialized with values from an existing matrix
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn clone(a: &Mat2d) -> Mat2d {
    let mut out: Mat2d = [0_f32; 6];
  
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
  
    out
}

/// Copy the values from one mat2d to another
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn copy(out: &mut Mat2d, a: &Mat2d) -> Mat2d {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];  
    out[5] = a[5];

    *out
}

/// Set a mat2d to the identity matrix
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn identity(out: &mut Mat2d) -> Mat2d {
    out[0] = 1.;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 1.;
    out[4] = 0.;
    out[5] = 0.;

    *out
}

/// Create a new mat2d with the given values
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn from_values(a: f32, b: f32, 
                   c: f32, d: f32, 
                   tx: f32, ty: f32) -> Mat2d {
    let mut out: Mat2d = [0_f32; 6];
  
    out[0] = a;
    out[1] = b;
    out[2] = c;
    out[3] = d;
    out[4] = tx;
    out[5] = ty;
  
    out
}

/// Set the components of a mat2d to the given values
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn set(out: &mut Mat2d, a: f32, b: f32, 
                            c: f32, d: f32, 
                            tx: f32, ty: f32) -> Mat2d {
    out[0] = a;
    out[1] = b;
    out[2] = c;
    out[3] = d;
    out[4] = tx;  
    out[5] = ty;

    *out
}

/// Inverts a mat2d
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn invert(out: &mut Mat2d, a: &Mat2d) -> Option<Mat2d> {
    let aa = a[0];
    let ab = a[1];
    let ac = a[2];
    let ad = a[3];
    let atx = a[4];
    let aty = a[5];
    // Calculate the determinant
    let det = aa * ad - ab * ac;
  
    // Make sure matrix is not singular
    if det == 0_f32 {  
        return None;
    }
  
    let det = 1_f32 / det;
  
    out[0] = ad * det;
    out[1] = -ab * det;
    out[2] = -ac * det;
    out[3] = aa * det;
    out[4] = (ac * aty - ad * atx) * det;
    out[5] = (ab * atx - aa * aty) * det;
  
    Some(*out)
  }

/// Calculates the determinant of a mat2d
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn determinant(a: &Mat2d) -> f32 { 
    a[0] * a[3] - a[1] * a[2]
}

/// Multiplies two mat2d's
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn multiply(out: &mut Mat2d, a: &Mat2d, b: &Mat2d) -> Mat2d {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2]; 
    let a3 = a[3]; 
    let a4 = a[4]; 
    let a5 = a[5];
    
    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3]; 
    let b4 = b[4]; 
    let b5 = b[5];

    out[0] = a0 * b0 + a2 * b1;
    out[1] = a1 * b0 + a3 * b1;
    out[2] = a0 * b2 + a2 * b3;
    out[3] = a1 * b2 + a3 * b3;
    out[4] = a0 * b4 + a2 * b5 + a4;
    out[5] = a1 * b4 + a3 * b5 + a5;

    *out
}

/// Rotates a mat2d by the given angle
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn rotate(out: &mut Mat2d, a: &Mat2d, rad: f32) -> Mat2d {
    let a0 = a[0]; 
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    let a4 = a[4];
    let a5 = a[5];

    let s = f32::sin(rad);
    let c = f32::cos(rad);
  
    out[0] = a0 *  c + a2 * s;
    out[1] = a1 *  c + a3 * s;
    out[2] = a0 * -s + a2 * c;
    out[3] = a1 * -s + a3 * c;
    out[4] = a4;
    out[5] = a5;

    *out
}

/// Scales the mat2d by the dimensions in the given vec2
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn scale(out: &mut Mat2d, a: &Mat2d, v: &Vec2) -> Mat2d {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3]; 
    let a4 = a[4];
    let a5 = a[5];
    
    let v0 = v[0];
    let v1 = v[1];
  
    out[0] = a0 * v0;
    out[1] = a1 * v0;
    out[2] = a2 * v1;
    out[3] = a3 * v1;
    out[4] = a4;
    out[5] = a5;

    *out
}

/// Translates the mat2d by the dimensions in the given vec2
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn translate(out: &mut Mat2d, a: &Mat2d, v: &Vec2) -> Mat2d {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3]; 
    let a4 = a[4];
    let a5 = a[5];
    
    let v0 = v[0];
    let v1 = v[1];
  
    out[0] = a0;
    out[1] = a1;
    out[2] = a2;
    out[3] = a3;
    out[4] = a0 * v0 + a2 * v1 + a4;
    out[5] = a1 * v0 + a3 * v1 + a5;

    *out
}

/// Creates a matrix from a given angle
///
/// This is equivalent to (but much faster than):
/// ```
/// use gl_matrix::common::*;
/// use gl_matrix::mat2d;
/// 
/// let dest = &mut [0., 0., 0., 0., 0., 0.];
/// let rad = PI * 0.5;
/// 
/// mat2d::identity(dest);
/// mat2d::rotate(dest, &mat2d::clone(dest), rad);
/// ```
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn from_rotation(out: &mut Mat2d, rad: f32) -> Mat2d {
    let s = f32::sin(rad);
    let c = f32::cos(rad);
  
    out[0] = c;
    out[1] = s;
    out[2] = -s;
    out[3] = c;
    out[4] = 0.;  
    out[5] = 0.;

    *out
}

/// Creates a matrix from a vector scaling
/// 
/// This is equivalent to (but much faster than):
/// ```
/// use gl_matrix::common::*;
/// use gl_matrix::mat2d;
///
/// let dest = &mut [0., 0., 0., 0., 0., 0.];
/// let vec: Vec2 = [2., 3.];
/// 
/// mat2d::identity(dest);
/// mat2d::scale(dest, &mat2d::clone(dest), &vec);
/// ```
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn from_scaling(out: &mut Mat2d, v: &Vec2) -> Mat2d {
    out[0] = v[0];
    out[1] = 0.;
    out[2] = 0.;
    out[3] = v[1];
    out[4] = 0.;
    out[5] = 0.;  

    *out
}

/// Creates a matrix from a vector translation
/// 
/// This is equivalent to (but much faster than):
/// ```
/// use gl_matrix::common::*;
/// use gl_matrix::mat2d;
///
/// let dest = &mut [0., 0., 0., 0., 0., 0.];
/// let vec: Vec2 = [2., 3.];
/// 
/// mat2d::identity(dest);
/// mat2d::translate(dest, &mat2d::clone(dest), &vec);
/// ```
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn from_translation(out: &mut Mat2d, v: &Vec2) -> Mat2d {
    out[0] = 1.;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 1.;
    out[4] = v[0];
    out[5] = v[1];  

    *out
}

/// Returns a string representation of a mat2d
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn string(a: &Mat2d) -> String {
    let a0 = ["mat2d(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = a[2].to_string(); 
    let a3 = a[3].to_string(); 
    let a4 = a[4].to_string(); 
    let a5 = [a[5].to_string(), ")".to_string()].join("");

    [a0, a1, a2, a3, a4, a5].join(", ")
}

/// Returns Frobenius norm of a mat2d
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn frob(a: &Mat2d) -> f32 {
    hypot(a)
}

/// Adds two mat2d's
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn add(out: &mut Mat2d, a: &Mat2d, b: &Mat2d) -> Mat2d {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
    out[4] = a[4] + b[4];
    out[5] = a[5] + b[5];   
    
    *out
}

/// Subtracts matrix b from matrix a
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn subtract(out: &mut Mat2d, a: &Mat2d, b: &Mat2d) -> Mat2d {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    out[3] = a[3] - b[3];
    out[4] = a[4] - b[4];
    out[5] = a[5] - b[5]; 
    
    *out
}

/// Multiply each element of the matrix by a scalar.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn multiply_scalar(out: &mut Mat2d, a: &Mat2d, b: f32) -> Mat2d {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
    out[3] = a[3] * b;
    out[4] = a[4] * b;
    out[5] = a[5] * b; 
    
    *out
}

/// Adds two mat2d's after multiplying each element of the second operand by a scalar value.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn multiply_scalar_and_add(out: &mut Mat2d, a: &Mat2d, b: &Mat2d, scale: f32) -> Mat2d {
    out[0] = a[0] + (b[0] * scale);
    out[1] = a[1] + (b[1] * scale);
    out[2] = a[2] + (b[2] * scale);
    out[3] = a[3] + (b[3] * scale);
    out[4] = a[4] + (b[4] * scale);
    out[5] = a[5] + (b[5] * scale);

    *out
}

/// Returns whether or not the matrices have exactly the same elements in the same position (when compared with ==)
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn exact_equals(a: &Mat2d, b: &Mat2d) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] && a[4] == b[4] && a[5] == b[5]
}

/// Returns whether or not the matrices have approximately the same elements in the same position.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn equals(a: &Mat2d, b: &Mat2d) -> bool {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    let a4 = a[4];
    let a5 = a[5];
    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3];
    let b4 = b[4];
    let b5 = b[5];

    f32::abs(a0 - b0) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a0), f32::abs(b0))) && 
    f32::abs(a1 - b1) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a1), f32::abs(b1))) && 
    f32::abs(a2 - b2) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a2), f32::abs(b2))) && 
    f32::abs(a3 - b3) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a3), f32::abs(b3))) && 
    f32::abs(a4 - b4) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a4), f32::abs(b4))) && 
    f32::abs(a5 - b5) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a5), f32::abs(b5)))
}

///  Alias for mat2d::multiply
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn mul(out: &mut Mat2d, a: &Mat2d, b: &Mat2d) -> Mat2d {
    multiply(out, a, b)
}

/// Alias for link mat2d::subtract
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat2d.html)
pub fn sub(out: &mut Mat2d, a: &Mat2d, b: &Mat2d) -> Mat2d {
    subtract(out, a, b)
}


#[cfg(test)] 
mod tests {
    use super::*; 

    #[test] 
    fn create_a_mat2d() {
        let ident: Mat2d = [1., 0., 0., 1., 0., 0.];
 
        let out = create();
        
        assert_eq!(ident, out);
    }

    #[test] 
    fn clone_a_mat2d() {
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
  
        let out = clone(&mat_a);
       
        assert_eq!(mat_a, out);
    }

    #[test] 
    fn copy_values_from_a_mat2d_to_another() {
        let mut out =  [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
   
        let result = copy(&mut out, &mat_a);
      
        assert_eq!(mat_a, out);
        assert_eq!(result, out);
    }

    #[test] 
    fn set_a_mat2d_to_identity() {  
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];  
        let ident = [1., 0., 0., 1., 0., 0.];
    
        let result = identity(&mut out);
     
        assert_eq!(out, ident);
        assert_eq!(result, out);
    }

    #[test]
    fn create_mat2d_from_values() { 
        let out = from_values(1., 2., 3. ,4., 5., 6.); 
    
        assert_eq!([1., 2., 3., 4., 5., 6.], out); 
    }

    #[test]
    fn set_mat2d_with_values() { 
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
     
        let result = set(&mut out, 1., 2., 3., 4., 5., 6.);

        assert_eq!([1., 2., 3., 4., 5., 6.], out); 
        assert_eq!(result, out);
    }
    
    #[test] 
    fn invert_mat2d() {  
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];

        let result = invert(&mut out, &mat_a).unwrap();

        assert_eq!([ -2., 1., 1.5, -0.5, 1., -2.], out);
        assert_eq!(result, out);
    } 

    #[test] 
    fn invert_singular_mat2d() {  
        let mut out: Mat2d =  [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [-1., 3./2., 2./3., -1., 0., 0.]; 

        let result = invert(&mut out, &mat_a); 
       
        assert_eq!([0., 0., 0., 0., 0., 0.], out);
        assert_eq!(None, result);
    } 
    
    #[test]
    fn get_mat2d_determinant() { 
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];

        let det: f32 = determinant(&mat_a);

        assert_eq!(-2_f32, det); 
    }

    #[test]
    fn multiply_two_mat2ds() {  
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let mat_b: Mat2d = [7., 8., 9., 10., 11., 12.];

        let result = multiply(&mut out, &mat_a, &mat_b);

        assert_eq!([31., 46., 39., 58., 52., 76.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mul_two_mat2ds() {  
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let mat_b: Mat2d = [7., 8., 9., 10., 11., 12.];

        let result = mul(&mut out, &mat_a, &mat_b);

        assert_eq!([31., 46., 39., 58., 52., 76.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn mul_is_equal_to_multiply() {  
        let mut out_a: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mut out_b: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let mat_b: Mat2d = [7., 8., 9., 10., 11., 12.];

        multiply(&mut out_a, &mat_a, &mat_b);
        mul(&mut out_b, &mat_a, &mat_b);

        assert_eq!(out_a, out_b);
    }
    
    #[test]
    fn rotate_a_mat2d() { 
        use super::super::common::{PI};
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        
        let result = rotate(&mut out, &mat_a, PI * 0.5);
        
        assert!(equals(&[3.0, 4.0, -1.0, -2.0, 5., 6.], &out));
        assert_eq!(result, out);
    }
    
    #[test]
    fn scale_mat2d() { 
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let vec_a: Vec2 = [2., 3.];

        let result = scale(&mut out, &mat_a, &vec_a);

        assert_eq!([2., 4., 9., 12., 5., 6.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn translate_mat2d() { 
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let vec_a: Vec2 = [2., 3.];

        let result = translate(&mut out, &mat_a, &vec_a);

        assert_eq!([1., 2., 3., 4., 16., 22.], out);
        assert_eq!(result, out);
    }
    
    #[test]
    fn mat2d_from_rotation() { 
        use super::super::common::{PI};
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];

        let result = from_rotation(&mut out, PI);

        assert!(equals(&[-1.0, -0.0, 0.0, -1.0, 0.0, 0.0], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn mat2d_from_scaling() { 
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let vec_a: Vec2 = [2., 3.];

        let result = from_scaling(&mut out, &vec_a);

        assert_eq!([2., 0., 0., 3., 0., 0.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mat2d_from_translation() {
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let vec_a: Vec2 = [2., 3.];

        from_translation(&mut out, &vec_a);

        assert_eq!([1., 0., 0., 1., 2., 3.], out)
    }

    #[test]
    fn get_mat2d_string() { 
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        
        let str_a = string(&mat_a);

        assert_eq!("mat2d(1, 2, 3, 4, 5, 6)".to_string(), str_a);
    }

    #[test]
    fn calc_frob_norm_of_mat2d() {
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];

        let frob_a = frob(&mat_a);

        assert_eq!((1_f32.powi(2) + 2_f32.powi(2) + 
                    3_f32.powi(2) + 4_f32.powi(2) + 
                    5_f32.powi(2) + 6_f32.powi(2)).sqrt(), frob_a);
    }

    #[test]
    fn add_two_mat2ds() { 
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let mat_b: Mat2d = [7., 8., 9., 10., 11., 12.];

        let result = add(&mut out, &mat_a, &mat_b);

        assert_eq!([8., 10., 12., 14., 16., 18.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn subtract_two_mat2ds() { 
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let mat_b: Mat2d = [7., 8., 9., 10., 11., 12.];

        let result = subtract(&mut out, &mat_a, &mat_b);

        assert_eq!([-6., -6., -6., -6., -6., -6.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_two_mat2ds() { 
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let mat_b: Mat2d = [7., 8., 9., 10., 11., 12.];

        let result = sub(&mut out, &mat_a, &mat_b);

        assert_eq!([-6., -6., -6., -6., -6., -6.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_is_equal_to_subtract() { 
        let mut out_a: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mut out_b: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let mat_b: Mat2d = [5., 6., 7., 8., 9., 10.];

        sub(&mut out_a, &mat_a, &mat_b);
        subtract(&mut out_b, &mat_a, &mat_b);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn multiply_mat2d_by_scalar() { 
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        
        multiply_scalar(&mut out, &mat_a, 2.);

        assert_eq!([2., 4., 6., 8., 10., 12.], out);
    }

    #[test]
    fn multiply_mat2d_by_scalar_and_add() { 
        let mut out: Mat2d = [0., 0., 0., 0., 0., 0.];
        let mat_a: Mat2d = [1., 2., 3., 4., 5., 6.];
        let mat_b: Mat2d = [7., 8., 9., 10., 11., 12.];

        multiply_scalar_and_add(&mut out, &mat_a, &mat_b, 0.5);

        assert_eq!([4.5, 6., 7.5, 9., 10.5, 12.], out);
    } 
   
    #[test]
    fn mat2ds_are_exact_equal() { 
        let mat_a: Mat2d = [0., 1., 2., 3., 5., 6.];
        let mat_b: Mat2d = [0., 1., 2., 3., 5., 6.];

        let r0 = exact_equals(&mat_a, &mat_b);

        assert!(r0);  
    }

    #[test]
    fn mat2ds_are_not_exact_equal() { 
        let mat_a: Mat2d = [0., 1., 2., 3., 4., 5.];
        let mat_b: Mat2d = [1., 2., 3., 4., 5., 6.];

        let r0 = exact_equals(&mat_a, &mat_b);

        assert!(!r0); 
    }

    #[test]
    fn mat2ds_are_equal() { 
        let mat_a: Mat2d = [0., 1., 2., 3., 4., 5.];
        let mat_b: Mat2d = [0., 1., 2., 3., 4., 5.];

        let r0 = equals(&mat_a, &mat_b);

        assert!(r0);  
    }

    #[test]
    fn mat2ds_are_equal_enough() { 
        let mat_a: Mat2d = [0., 1., 2., 3., 4., 5.];
        let mat_b: Mat2d = [1_f32*10_f32.powi(-16), 1., 2., 3., 4., 5.];

        let r0 = equals(&mat_a, &mat_b);

        assert!(r0);  
    }

    #[test]
    fn mat2ds_are_not_equal() { 
        let mat_a: Mat2d = [0., 1., 2., 3., 4., 5.];
        let mat_b: Mat2d = [1., 2., 3., 4., 5., 6.];

        let r0 = equals(&mat_a, &mat_b);

        assert!(!r0);  
    }
}
