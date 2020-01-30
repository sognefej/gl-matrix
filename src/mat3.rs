//! 3x3 Matrix

use super::common::{Mat3, Mat4, Mat2d, Quat, Vec2, hypot, EPSILON};

/// Creates a new identity mat3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn create() -> Mat3 {
    let mut out: Mat3 = [0_f32; 9];

    out[0] = 1.;
    out[4] = 1.;
    out[8] = 1.;
    
    out
}

/// Copies the upper-left 3x3 values into the given mat3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn from_mat4(out: &mut Mat3, a: &Mat4) -> Mat3 {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[4];
    out[4] = a[5];
    out[5] = a[6];
    out[6] = a[8];
    out[7] = a[9];
    out[8] = a[10];

    *out
}

/// Creates a new mat3 initialized with values from an existing matrix.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn clone(a: &Mat3) -> Mat3 {
    let mut out: Mat3 = [0_f32; 9];

    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];

    out
}
  
/// Copy the values from one mat3 to another.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn copy(out: &mut Mat3, a: &Mat3) -> Mat3 {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];

    *out
}

/// Create a new mat3 with the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn from_values(m00: f32, m01: f32, m02: f32, 
                   m10: f32, m11: f32, m12: f32, 
                   m20: f32, m21: f32, m22: f32) -> Mat3 {
    let mut out: Mat3 = [0_f32; 9];

    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m10;
    out[4] = m11;
    out[5] = m12;
    out[6] = m20;
    out[7] = m21;
    out[8] = m22;

    out
}

/// Set the components of a mat3 to the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn set(out: &mut Mat3, m00: f32, m01: f32, m02: f32,
                           m10: f32, m11: f32, m12: f32,
                           m20: f32, m21: f32, m22: f32) -> Mat3 {
    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m10;
    out[4] = m11;
    out[5] = m12;
    out[6] = m20;
    out[7] = m21;
    out[8] = m22;

    *out
}


/// Set a mat3 to the identity matrix.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn identity(out: &mut Mat3) -> Mat3 {
    out[0] = 1.;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = 1.;
    out[5] = 0.;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = 1.;

    *out 
}

/// Copies the upper-left 3x3 values into the given mat3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn transpose(out: &mut Mat3, a: &Mat3) -> Mat3 {
    // If we are transposing ourselves we can skip a few steps but have to cache some values
    if out.eq(&a) {
        let a01 = a[1];
        let a02 = a[2];
        let a12 = a[5];
        
        out[1] = a[3];
        out[2] = a[6];
        out[3] = a01;
        out[5] = a[7];
        out[6] = a02;
        out[7] = a12;
    } else {
        out[0] = a[0];
        out[1] = a[3];
        out[2] = a[6];
        out[3] = a[1];
        out[4] = a[4];
        out[5] = a[7];
        out[6] = a[2];
        out[7] = a[5];
        out[8] = a[8];
    }

    *out
}
  
/// Inverts a mat3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn invert(out: &mut Mat3, a: &Mat3) -> Option<Mat3> {
    let a00 = a[0];
    let a01 = a[1];
    let a02 = a[2];
    let a10 = a[3];
    let a11 = a[4];
    let a12 = a[5];
    let a20 = a[6];
    let a21 = a[7];
    let a22 = a[8];
    let b01 = a22 * a11 - a12 * a21;
    let b11 = -a22 * a10 + a12 * a20;
    let b21 = a21 * a10 - a11 * a20;
    // Calculate the determinant
    let det = a00 * b01 + a01 * b11 + a02 * b21;
  
    // Make sure matrix is not singular
    if det == 0_f32 {  
        return None;
    }

    let det = 1_f32 / det;
  
    out[0] = b01 * det;
    out[1] = (-a22 * a01 + a02 * a21) * det;
    out[2] = (a12 * a01 - a02 * a11) * det;
    out[3] = b11 * det;
    out[4] = (a22 * a00 - a02 * a20) * det;
    out[5] = (-a12 * a00 + a02 * a10) * det;
    out[6] = b21 * det;
    out[7] = (-a21 * a00 + a01 * a20) * det;
    out[8] = (a11 * a00 - a01 * a10) * det;

    Some(*out)
}

/// Calculates the adjugate of a mat3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn adjoint(out: &mut Mat3, a: &Mat3) -> Mat3 {
    let a00 = a[0];
    let a01 = a[1];
    let a02 = a[2];
    let a10 = a[3];
    let a11 = a[4];
    let a12 = a[5];
    let a20 = a[6];
    let a21 = a[7];
    let a22 = a[8];

    out[0] = a11 * a22 - a12 * a21;
    out[1] = a02 * a21 - a01 * a22;
    out[2] = a01 * a12 - a02 * a11;
    out[3] = a12 * a20 - a10 * a22;
    out[4] = a00 * a22 - a02 * a20;
    out[5] = a02 * a10 - a00 * a12;
    out[6] = a10 * a21 - a11 * a20;
    out[7] = a01 * a20 - a00 * a21;
    out[8] = a00 * a11 - a01 * a10;

    *out
}

/// Calculates the determinant of a mat3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn determinant(a: &Mat3) -> f32 {
    let a00 = a[0];
    let a01 = a[1];
    let a02 = a[2];
    let a10 = a[3];
    let a11 = a[4];
    let a12 = a[5];
    let a20 = a[6];
    let a21 = a[7];
    let a22 = a[8];

    a00 * (a22 * a11 - a12 * a21) 
    + a01 * (-a22 * a10 + a12 * a20) 
    + a02 * (a21 * a10 - a11 * a20)
}


/// Multiplies two mat3's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn multiply(out: &mut Mat3, a: &Mat3, b: &Mat3) -> Mat3 {
    let a00 = a[0];
    let a01 = a[1]; 
    let a02 = a[2];
    let a10 = a[3]; 
    let a11 = a[4];
    let a12 = a[5];
    let a20 = a[6]; 
    let a21 = a[7];
    let a22 = a[8];

    let b00 = b[0]; 
    let b01 = b[1];
    let b02 = b[2];
    let b10 = b[3]; 
    let b11 = b[4];
    let b12 = b[5];
    let b20 = b[6]; 
    let b21 = b[7];
    let b22 = b[8];

    out[0] = b00 * a00 + b01 * a10 + b02 * a20;
    out[1] = b00 * a01 + b01 * a11 + b02 * a21;
    out[2] = b00 * a02 + b01 * a12 + b02 * a22;
    out[3] = b10 * a00 + b11 * a10 + b12 * a20;
    out[4] = b10 * a01 + b11 * a11 + b12 * a21;
    out[5] = b10 * a02 + b11 * a12 + b12 * a22;
    out[6] = b20 * a00 + b21 * a10 + b22 * a20;
    out[7] = b20 * a01 + b21 * a11 + b22 * a21;
    out[8] = b20 * a02 + b21 * a12 + b22 * a22;

    *out
}


/// Translate a mat3 by the given vector.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn translate(out: &mut Mat3, a: &Mat3, v: &Vec2) -> Mat3 {
    let a00 = a[0];
    let a01 = a[1];
    let a02 = a[2];
    let a10 = a[3];
    let a11 = a[4];
    let a12 = a[5];
    let a20 = a[6];
    let a21 = a[7];
    let a22 = a[8];

    let x = v[0];
    let y = v[1];

    out[0] = a00; 
    out[1] = a01;
    out[2] = a02;
    out[3] = a10;
    out[4] = a11;
    out[5] = a12;
    out[6] = x * a00 + y * a10 + a20;
    out[7] = x * a01 + y * a11 + a21;
    out[8] = x * a02 + y * a12 + a22;

    *out
}


/// Rotates a mat3 by the given angle.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn rotate(out: &mut Mat3, a: &Mat3, rad: f32) -> Mat3 {
    let a00 = a[0];
    let a01 = a[1];
    let a02 = a[2];
    let a10 = a[3];
    let a11 = a[4];
    let a12 = a[5];
    let a20 = a[6];
    let a21 = a[7];
    let a22 = a[8];

    let s = f32::sin(rad);
    let c = f32::cos(rad);

    out[0] = c * a00 + s * a10;
    out[1] = c * a01 + s * a11;
    out[2] = c * a02 + s * a12;
    out[3] = c * a10 - s * a00;
    out[4] = c * a11 - s * a01;
    out[5] = c * a12 - s * a02;
    out[6] = a20;
    out[7] = a21;
    out[8] = a22;

    *out
}

/// Scales the mat3 by the dimensions in the given vec2.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn scale(out: &mut Mat3, a: &Mat3, v: &Vec2) -> Mat3 {
    let x = v[0];
    let y = v[1];

    out[0] = x * a[0];
    out[1] = x * a[1];
    out[2] = x * a[2];
    out[3] = y * a[3];
    out[4] = y * a[4];
    out[5] = y * a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];

    *out
}

/// Creates a matrix from a vector translation.
/// 
/// This is equivalent to (but much faster than):
/// ```
/// use gl_matrix::common::*;
/// use gl_matrix::mat3;
///
/// let dest = &mut [0., 0., 0., 
///                  0., 0., 0.,
///                  0., 0., 0.];
/// let vec: Vec2 = [2., 3.];
/// 
/// mat3::identity(dest);
/// mat3::translate(dest, &mat3::clone(dest), &vec);
/// ```
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn from_translation(out: &mut Mat3 , v: &Vec2) -> Mat3 {
    out[0] = 1.;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = 1.;
    out[5] = 0.;
    out[6] = v[0];
    out[7] = v[1];
    out[8] = 1.;

    *out
}
  
/// Creates a matrix from a given angle.
/// 
/// This is equivalent to (but much faster than):
/// ```
/// use gl_matrix::common::*;
/// use gl_matrix::mat3;
///
/// let dest = &mut [0., 0., 0., 
///                  0., 0., 0.,
///                  0., 0., 0.];
/// let rad = PI * 0.5;
/// 
/// mat3::identity(dest);
/// mat3::rotate(dest, &mat3::clone(dest), rad);
/// ```
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn from_rotation(out: &mut Mat3, rad: f32) -> Mat3 {
    let s = f32::sin(rad);
    let c = f32::cos(rad);

    out[0] = c;
    out[1] = s;
    out[2] = 0.;
    out[3] = -s;
    out[4] = c;
    out[5] = 0.;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = 1.;

    *out
}
  
/// Creates a matrix from a vector scaling.
/// 
/// This is equivalent to (but much faster than):
/// ```
/// use gl_matrix::common::*;
/// use gl_matrix::mat3;
///
/// let dest = &mut [0., 0., 0., 
///                  0., 0., 0.,
///                  0., 0., 0.];
/// let vec: Vec2 = [2., 3.];
/// 
/// mat3::identity(dest);
/// mat3::scale(dest, &mat3::clone(dest), &vec);
/// ```
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn from_scaling(out: &mut Mat3, v: &Vec2) -> Mat3 {
    out[0] = v[0];
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = v[1];
    out[5] = 0.;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = 1.;

    *out
}

/// Copies the values from a mat2d into a mat3.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn from_mat2d(out: &mut Mat3, a: &Mat2d) -> Mat3 {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = 0.;
    out[3] = a[2];
    out[4] = a[3];
    out[5] = 0.;
    out[6] = a[4];
    out[7] = a[5];
    out[8] = 1.;

    *out
}

/// Calculates a 3x3 matrix from the given quaternion.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn from_quat(out: &mut Mat3, q: &Quat) -> Mat3 {
    let x = q[0];
    let y = q[1];
    let z = q[2];
    let w = q[3];
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let yx = y * x2;
    let yy = y * y2;
    let zx = z * x2;
    let zy = z * y2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    out[0] = 1. - yy - zz;
    out[3] = yx - wz;
    out[6] = zx + wy;
    out[1] = yx + wz;
    out[4] = 1. - xx - zz;
    out[7] = zy - wx;
    out[2] = zx - wy;
    out[5] = zy + wx;
    out[8] = 1. - xx - yy;

    *out
}


/// Calculates a 3x3 normal matrix (transpose inverse) from the 4x4 matrix.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn normal_from_mat4(out: &mut Mat3, a: &Mat4) -> Option<Mat3> {
    let a00 = a[0];
    let a01 = a[1];
    let a02 = a[2];
    let a03 = a[3];
    let a10 = a[4];
    let a11 = a[5];
    let a12 = a[6];
    let a13 = a[7];
    let a20 = a[8];
    let a21 = a[9];
    let a22 = a[10];
    let a23 = a[11];
    let a30 = a[12];
    let a31 = a[13];
    let a32 = a[14];
    let a33 = a[15];
  
    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;
    // Calculate the determinant
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  
    if det == 0_f32 {  
        return None;
    }
  
    let det = 1_f32 / det;
  
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[2] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[3] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[4] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[5] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[6] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[7] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[8] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    
    Some(*out)
}
  
/// Generates a 2D projection matrix with the given bounds.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn projection(out: &mut Mat3, width: f32, height: f32) -> Mat3 {
    out[0] = 2. / width;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = -2. / height;
    out[5] = 0.;
    out[6] = -1.;
    out[7] = 1.;
    out[8] = 1.;

    *out
}
 
/// Returns a string representation of a mat3.
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn string(a: &Mat3) -> String {
    let a0 = ["mat3(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = a[2].to_string(); 
    let a3 = a[3].to_string(); 
    let a4 = a[4].to_string(); 
    let a5 = a[5].to_string(); 
    let a6 = a[6].to_string(); 
    let a7 = a[7].to_string(); 
    let a8 = [a[8].to_string(), ")".to_string()].join("");

    [a0, a1, a2, a3, a4, a5, a6, a7, a8].join(", ")
}
 
/// Returns Frobenius norm of a mat3.
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn frob(a: &Mat3) -> f32 {
    hypot(a)
}
  
/// Adds two mat3's.
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn add(out: &mut Mat3, a: &Mat3, b: &Mat3) -> Mat3 {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
    out[4] = a[4] + b[4];
    out[5] = a[5] + b[5];
    out[6] = a[6] + b[6];
    out[7] = a[7] + b[7];
    out[8] = a[8] + b[8];

    *out
}
  
/// Subtracts matrix b from matrix a.
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn subtract(out: &mut Mat3, a: &Mat3, b: &Mat3) -> Mat3 {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    out[3] = a[3] - b[3];
    out[4] = a[4] - b[4];
    out[5] = a[5] - b[5];
    out[6] = a[6] - b[6];
    out[7] = a[7] - b[7];
    out[8] = a[8] - b[8];

    *out
}

/// Multiply each element of the matrix by a scalar.
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn multiply_scalar(out: &mut Mat3, a: &Mat3, b: f32) -> Mat3 {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
    out[3] = a[3] * b;
    out[4] = a[4] * b;
    out[5] = a[5] * b;
    out[6] = a[6] * b;
    out[7] = a[7] * b;
    out[8] = a[8] * b;

    *out
}
  
/// Adds two mat3's after multiplying each element of the second operand by a scalar value.
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn multiply_scalar_and_add(out: &mut Mat3, a: &Mat3, b: &Mat3, scale: f32) -> Mat3 {
    out[0] = a[0] + (b[0] * scale);
    out[1] = a[1] + (b[1] * scale);
    out[2] = a[2] + (b[2] * scale);
    out[3] = a[3] + (b[3] * scale);
    out[4] = a[4] + (b[4] * scale);
    out[5] = a[5] + (b[5] * scale);
    out[6] = a[6] + (b[6] * scale);
    out[7] = a[7] + (b[7] * scale);
    out[8] = a[8] + (b[8] * scale);

    *out
}
 
/// Returns whether or not the matrices have exactly the same elements in the same position (when compared with ==).
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn exact_equals(a: &Mat3, b: &Mat3) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] &&
    a[3] == b[3] && a[4] == b[4] && a[5] == b[5] &&
    a[6] == b[6] && a[7] == b[7] && a[8] == b[8]
}
  

/// Returns whether or not the matrices have approximately the same elements in the same position.
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn equals(a: &Mat3, b: &Mat3) -> bool {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    let a4 = a[4];
    let a5 = a[5];
    let a6 = a[6];
    let a7 = a[7];
    let a8 = a[8];
    
    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3];
    let b4 = b[4];
    let b5 = b[5]; 
    let b6 = b[6];
    let b7 = b[7];
    let b8 = b[8];

    f32::abs(a0 - b0) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a0), f32::abs(b0))) && 
    f32::abs(a1 - b1) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a1), f32::abs(b1))) && 
    f32::abs(a2 - b2) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a2), f32::abs(b2))) && 
    f32::abs(a3 - b3) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a3), f32::abs(b3))) &&
    f32::abs(a4 - b4) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a4), f32::abs(b4))) &&
    f32::abs(a5 - b5) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a5), f32::abs(b5))) &&
    f32::abs(a6 - b6) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a6), f32::abs(b6))) && 
    f32::abs(a7 - b7) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a7), f32::abs(b7))) && 
    f32::abs(a8 - b8) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a8), f32::abs(b8)))
}


/// Alias for mat3::multiply.
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn mul(out: &mut Mat3, a: &Mat3, b: &Mat3) -> Mat3 {
    multiply(out, a, b)
}

/// Alias for mat3::subtract.
///
/// [glMatrix Documentation](http://glmatrix.net/docs/module-mat3.html)
pub fn sub(out: &mut Mat3, a: &Mat3, b: &Mat3) -> Mat3 {
    subtract(out, a, b)
}


#[cfg(test)] 
mod tests {
    use super::*; 

    #[test] 
    fn create_a_mat3() {
        let ident: Mat3 = [1., 0., 0., 
                           0., 1., 0., 
                           0., 0., 1.];
 
        let out = create();
        
        assert_eq!(ident, out);
    }

    #[test] 
    fn mat3_from_mat4() {
        let mut out: Mat3 = [0., 0., 0.,
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           0., 0., 0., 1.];
       
        let result = from_mat4(&mut out, &mat_a);

        assert_eq!([1., 0., 0., 
                    0., 1., 0., 
                    0., 0., 1.], out);
        assert_eq!(result, out);
    }
    
    #[test] 
    fn clone_a_mat3() {
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
            
        let out = clone(&mat_a);
       
        assert_eq!(mat_a, out);
    }

    #[test]
    fn copy_values_from_a_mat3_to_another() {
        let mut out: Mat3 = [0., 0., 0.,
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
   
        let result = copy(&mut out, &mat_a);
      
        assert_eq!(mat_a, out);
        assert_eq!(result, out);
    }

    #[test]
    fn create_mat3_from_values() { 
        let out = from_values(1., 2., 3.,
                              4., 5., 6.,
                              7., 8., 9.); 
    
        assert_eq!([1., 2., 3.,
                    4., 5., 6.,
                    7., 8., 9.], out); 
    }
   
    #[test]
    fn set_mat3_with_values() { 
        let mut out: Mat3 = [0., 0., 0.,
                             0., 0., 0.,
                             0., 0., 0.];
     
        let result = set(&mut out, 1., 2., 3.,
                                   4., 5., 6.,
                                   7., 8., 9.);

        assert_eq!([1., 2., 3.,
                    4., 5., 6.,
                    7., 8., 9.], out);
        assert_eq!(result, out);
    }
   
    #[test]
    fn set_a_mat3_to_identity() {  
        let mut out: Mat3 = [0., 0., 0.,
                             0., 0., 0.,
                             0., 0., 0.];
        let ident: Mat3 = [1., 0., 0., 
                           0., 1., 0., 
                           0., 0., 1.];
    
        let result = identity(&mut out);
     
        assert_eq!(ident, out);
        assert_eq!(result, out);
    }
   
    #[test] 
    fn transpose_same_mat3() { 
        let mut mat_a: Mat3 = [1., 0., 0.,
                               0., 1., 0.,
                               1., 2., 1.];
        let mat_a_copy: Mat3 = [1., 0., 0.,
                                0., 1., 0.,
                                1., 2., 1.];
      
        let result = transpose(&mut mat_a, &mat_a_copy);

        assert_eq!([1., 0., 1.,
                    0., 1., 2.,
                    0., 0., 1.], mat_a);
        assert_eq!(result, mat_a);
    }

    #[test] 
    fn transpose_different_mat3() { 
        let mut out: Mat3 = [0., 0., 0.,
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
        
        let result = transpose(&mut out, &mat_a); 
       
        assert_eq!([1., 0., 1.,
                    0., 1., 2.,
                    0., 0., 1.], out);
        assert_eq!(result, out);
    }
   
    #[test]
    fn adjugate_mat3() { 
        let mut out: Mat3 = [0., 0., 0.,
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
        
        let result = adjoint(&mut out, &mat_a); 
        
        assert_eq!([1., 0., 0.,
                    0., 1., 0.,
                   -1., -2., 1.], out);         
        assert_eq!(result, out);
    }
    
    #[test]
    fn get_mat3_determinant() { 
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];

        let det: f32 = determinant(&mat_a);

        assert_eq!(1_f32, det); 
    }

    #[test]
    fn multiply_two_mat3s() {  
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
        let mat_b: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           3., 4., 1.];

        let result = multiply(&mut out, &mat_a, &mat_b);

        assert_eq!([1., 0., 0.,
                    0., 1., 0.,
                    4., 6., 1.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn mul_two_mat3s() {  
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
        let mat_b: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           3., 4., 1.];

        let result = mul(&mut out, &mat_a, &mat_b);

        assert_eq!([1., 0., 0.,
                    0., 1., 0.,
                    4., 6., 1.], out); 
        assert_eq!(result, out);
    }

    #[test]
    fn mul_is_equal_to_multiply() {  
        let mut out_a: Mat3 = [0., 0., 0., 
                               0., 0., 0.,
                               0., 0., 0.];
        let mut out_b: Mat3 = [0., 0., 0., 
                               0., 0., 0.,
                               0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
        let mat_b: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           3., 4., 1.];

        multiply(&mut out_a, &mat_a, &mat_b);
        mul(&mut out_b, &mat_a, &mat_b);

        assert_eq!(out_a, out_b); 
    }

    #[test]
    fn translate_mat3() { 
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
        let vec_a: Vec2 = [2., 3.];

        let result = translate(&mut out, &mat_a, &vec_a);

        assert_eq!([1., 0., 0.,
                    0., 1., 0.,
                    3., 5., 1.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_a_mat3() { 
        use super::super::common::{PI};
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
        
        let result = rotate(&mut out, &mat_a, PI * 0.5);
        
        assert!(equals(&[0., 1., 0.,
                        -1., 0., 0.,
                         1., 2., 1.], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn scale_mat3() { 
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];
        let vec_a: Vec2 = [2., 3.];

        let scale = scale(&mut out, &mat_a, &vec_a);

        assert_eq!([ 2., 0., 0.,
                     0., 3., 0.,
                     1., 2., 1.], out);
        assert_eq!(scale, out);
    }
   
    #[test]
    fn mat3_from_translation() { 
        let mut out = create(); 
        let vec_a: Vec2 = [2., 3.];

        let result = from_translation(&mut out, &vec_a);

        assert_eq!([1., 0., 0.,
                    0., 1., 0.,
                    2., 3., 1.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mat3_from_rotation() { 
        use super::super::common::{PI};
        let mut out = create(); 
        
        let result = from_rotation(&mut out, PI);
        
        assert!(equals(&[-1., 0., 0.,
                          0.,-1., 0.,
                          0., 0., 1.], &out));
        assert_eq!(result, out);
        }

    #[test]
    fn mat3_from_scaling() { 
        let mut out = create(); 
        let vec_a: Vec2 = [2., 3.];

        let result = from_scaling(&mut out, &vec_a);

        assert_eq!([ 2., 0., 0.,
                     0., 3., 0.,
                     0., 0., 1.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mat3_from_mat2d() { 
        let mut out = create(); 
        let mat_a: Mat2d = [1., 2., 
                            3., 4.,
                            5., 6.];

        let result = from_mat2d(&mut out, &mat_a);

        assert_eq!([1., 2., 0.,
                    3., 4., 0.,
                    5., 6., 1.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mat3_from_quat() { 
        let mut out = create();
        let q: Quat = [0., -0.7071067811865475, 0., 0.7071067811865475];

        let result = from_quat(&mut out, &q);

        assert!(equals(&[0., 0., 1., 
                         0., 1., 0., 
                        -1., 0., 0.], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn invert_mat3() { 
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];

        let result = invert(&mut out, &mat_a).unwrap();

        assert_eq!([1., 0., 0.,
                    0., 1., 0.,
                   -1., -2., 1.], out);
        assert_eq!(result, out);
    }

    #[test] 
    fn invert_singular_mat2d() {  
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [-1., 3./2., 0.,
                            2./3., -1., 0.,
                            0., 0., 1.]; 

        let result = invert(&mut out, &mat_a); 

        assert_eq!([0., 0., 0., 
                    0., 0., 0.,
                    0., 0., 0.], out);
        assert_eq!(None, result);
    } 

    #[test]
    fn projection_of_mat3() { 
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        
        let result = projection(&mut out, 100.0, 200.0);

        assert_eq!([0.02, 0., 0.,
                    0., -0.01, 0.,
                    -1., 1., 1.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn get_mat3_string() { 
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];

        let str_a = string(&mat_a);

        assert_eq!("mat3(1, 0, 0, 0, 1, 0, 1, 2, 1)".to_string(), str_a);
    }

    #[test]
    fn calc_frob_norm_of_mat3() { 
        let mat_a: Mat3 = [1., 0., 0.,
                           0., 1., 0.,
                           1., 2., 1.];

        let frob_a = frob(&mat_a);

        assert_eq!((1_f32.powi(2) + 0_f32.powi(2) + 0_f32.powi(2) + 
                    0_f32.powi(2) + 1_f32.powi(2) + 0_f32.powi(2) + 
                    1_f32.powi(2) + 2_f32.powi(2) + 1_f32.powi(2)).sqrt(), frob_a);
    }

    #[test]
    fn add_two_mat3s() { 
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 2., 3.,
                           4., 5., 6.,
                           7., 8., 9.];
        let mat_b: Mat3 = [10., 11., 12.,
                           13., 14., 15.,
                           16., 17., 18.];
        
        let result = add(&mut out, &mat_a, &mat_b);

        assert_eq!([11., 13., 15., 
                    17., 19., 21., 
                    23., 25., 27.], out);
        assert_eq!(result, out);
    } 

    #[test]
    fn subtract_two_mat3s() { 
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 2., 3.,
                           4., 5., 6.,
                           7., 8., 9.];
        let mat_b: Mat3 = [10., 11., 12.,
                           13., 14., 15.,
                           16., 17., 18.];
        
        let result = subtract(&mut out, &mat_a, &mat_b);

        assert_eq!([-9., -9., -9., 
                    -9., -9., -9., 
                    -9., -9., -9.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_two_mat3s() { 
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 2., 3.,
                           4., 5., 6.,
                           7., 8., 9.];
        let mat_b: Mat3 = [10., 11., 12.,
                           13., 14., 15.,
                           16., 17., 18.];
        
        let result = sub(&mut out, &mat_a, &mat_b);

        assert_eq!([-9., -9., -9., 
                    -9., -9., -9., 
                    -9., -9., -9.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn sub_is_equal_to_subtract() { 
        let mut out_a: Mat3 = [0., 0., 0., 
                               0., 0., 0.,
                               0., 0., 0.];
        let mut out_b: Mat3 = [0., 0., 0., 
                               0., 0., 0.,
                               0., 0., 0.];
        let mat_a: Mat3 = [1., 2., 3.,
                           4., 5., 6.,
                           7., 8., 9.];
        let mat_b: Mat3 = [10., 11., 12.,
                           13., 14., 15.,
                           16., 17., 18.];
        
        sub(&mut out_a, &mat_a, &mat_b);
        subtract(&mut out_b, &mat_a, &mat_b);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn multiply_mat3_by_scalar() { 
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 2., 3.,
                           4., 5., 6.,
                           7., 8., 9.];
        
        let result = multiply_scalar(&mut out, &mat_a, 2.);

        assert_eq!([2., 4., 6., 
                    8., 10., 12., 
                    14., 16., 18.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn multiply_mat3_by_scalar_and_add() { 
        let mut out: Mat3 = [0., 0., 0., 
                             0., 0., 0.,
                             0., 0., 0.];
        let mat_a: Mat3 = [1., 2., 3.,
                           4., 5., 6.,
                           7., 8., 9.];
        let mat_b: Mat3 = [10., 11., 12.,
                           13., 14., 15.,
                           16., 17., 18.];

        let result = multiply_scalar_and_add(&mut out, &mat_a, &mat_b, 0.5);

        assert_eq!([6., 7.5, 9., 
                    10.5, 12., 13.5, 
                    15., 16.5, 18.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mat3s_are_exact_equal() { 
        let mat_a: Mat3 = [0., 1., 2., 
                           3., 5., 6., 
                           7., 8., 9.];
        let mat_b: Mat3 = [0., 1., 2., 
                           3., 5., 6., 
                           7., 8., 9.];

        let r0 = exact_equals(&mat_a, &mat_b);

        assert!(r0);  
    }

    #[test]
    fn mat3s_are_not_exact_equal() { 
        let mat_a: Mat3 = [0., 1., 2., 
                           3., 4., 5., 
                           6., 7., 8.];
        let mat_b: Mat3 = [1., 2., 3.,
                           4., 5., 6., 
                           7., 8., 9.];

        let r0 = exact_equals(&mat_a, &mat_b);

        assert!(!r0); 
    }

    #[test]
    fn mat3s_are_equal() { 
        let mat_a: Mat3 = [0., 1., 2., 
                           3., 5., 6., 
                           7., 8., 9.];
        let mat_b: Mat3 = [0., 1., 2., 
                           3., 5., 6., 
                           7., 8., 9.];

        let r0 = equals(&mat_a, &mat_b);

        assert!(r0);  
    }

    #[test]
    fn mat3s_are_equal_enough() { 
        let mat_a: Mat3 = [0., 1., 2., 
                           3., 5., 6., 
                           7., 8., 9.];
        let mat_b: Mat3 = [1_f32*10_f32.powi(-16), 1., 2., 
                           3., 5., 6., 
                           7., 8., 9.];

        let r0 = equals(&mat_a, &mat_b);

        assert!(r0);  
    }

    #[test]
    fn mat3s_are_not_equal() { 
        let mat_a: Mat3 = [0., 1., 2., 
                           3., 4., 5., 
                           6., 7., 8.];
        let mat_b: Mat3 = [1., 2., 3.,
                           4., 5., 6., 
                           7., 8., 9.];

        let r0 = equals(&mat_a, &mat_b);

        assert!(!r0);  
    }
}