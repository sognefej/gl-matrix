use super::common::{Mat4, Vec3, Quat, Quat2, EPSILON, PI, hypot, INFINITY, NEG_INFINITY};

pub struct Fov {
    up_degrees: f32,
    down_degrees: f32, 
    left_degrees: f32,
    right_degrees: f32,
}

pub fn create() -> Mat4 {
    let mut out: Mat4 = [0_f32; 16];

    out[0] = 1.;
    out[5] = 1.;
    out[10] = 1.;
    out[15] = 1.;

    out
}

pub fn clone(a: &Mat4) -> Mat4 {
    let mut out: Mat4 = [0_f32; 16];

    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];
    out[9] = a[9];
    out[10] = a[10];
    out[11] = a[11];
    out[12] = a[12];
    out[13] = a[13];
    out[14] = a[14];
    out[15] = a[15];

    out
}

pub fn copy(out: &mut Mat4, a: &Mat4) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    out[4] = a[4];
    out[5] = a[5];
    out[6] = a[6];
    out[7] = a[7];
    out[8] = a[8];
    out[9] = a[9];
    out[10] = a[10];
    out[11] = a[11];
    out[12] = a[12];
    out[13] = a[13];
    out[14] = a[14];
    out[15] = a[15];
}

pub fn from_values(m00: f32, m01: f32, m02: f32, m03: f32, 
                   m10: f32, m11: f32, m12: f32, m13: f32, 
                   m20: f32, m21: f32, m22: f32, m23: f32, 
                   m30: f32, m31: f32, m32: f32, m33: f32) -> Mat4 {
    let mut out: Mat4 = [0_f32; 16];

    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m03;
    out[4] = m10;
    out[5] = m11;
    out[6] = m12;
    out[7] = m13;
    out[8] = m20;
    out[9] = m21;
    out[10] = m22;
    out[11] = m23;
    out[12] = m30;
    out[13] = m31;
    out[14] = m32;
    out[15] = m33;

    out
}

pub fn set(out: &mut Mat4, m00: f32, m01: f32, m02: f32, m03: f32, 
                           m10: f32, m11: f32, m12: f32, m13: f32, 
                           m20: f32, m21: f32, m22: f32, m23: f32, 
                           m30: f32, m31: f32, m32: f32, m33: f32) {
    out[0] = m00;
    out[1] = m01;
    out[2] = m02;
    out[3] = m03;
    out[4] = m10;
    out[5] = m11;
    out[6] = m12;
    out[7] = m13;
    out[8] = m20;
    out[9] = m21;
    out[10] = m22;
    out[11] = m23;
    out[12] = m30;
    out[13] = m31;
    out[14] = m32;
    out[15] = m33;
}

pub fn identity(out: &mut Mat4) {
    out[0] = 1.;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = 0.;
    out[5] = 1.;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = 0.;
    out[9] = 0.;
    out[10] = 1.;
    out[11] = 0.;
    out[12] = 0.;
    out[13] = 0.;
    out[14] = 0.;
    out[15] = 1.;
}

pub fn transpose(out: &mut Mat4, a: &Mat4) {
    // If we are transposing ourselves we can skip a few steps but have to cache some values
    if out.eq(&a) {
        let a01 = a[1];
        let a02 = a[2];
        let a03 = a[3];
        let a12 = a[6];
        let a13 = a[7];
        let a23 = a[11];
      
        out[1] = a[4];
        out[2] = a[8];
        out[3] = a[12];
        out[4] = a01;
        out[6] = a[9];
        out[7] = a[13];
        out[8] = a02;
        out[9] = a12;
        out[11] = a[14];
        out[12] = a03;
        out[13] = a13;
        out[14] = a23;
    } else {
        out[0] = a[0];
        out[1] = a[4];
        out[2] = a[8];
        out[3] = a[12];
        out[4] = a[1];
        out[5] = a[5];
        out[6] = a[9];
        out[7] = a[13];
        out[8] = a[2];
        out[9] = a[6];
        out[10] = a[10];
        out[11] = a[14];
        out[12] = a[3];
        out[13] = a[7];
        out[14] = a[11];
        out[15] = a[15];
    }
}

pub fn invert(mut out: Mat4, a: &Mat4) -> Result<Mat4, String> {
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

    // Make sure matrix is not singular
    if det == 0_f32 {  
        return Err("Matrix is singular".to_string());
    }

    let det = 1_f32 / det;

    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;

    Ok(out)
}

pub fn adjoint(out: &mut Mat4, a: &Mat4) {
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

    out[0] =  a11 * (a22 * a33 - a23 * a32) - a21 * (a12 * a33 - a13 * a32) + a31 * (a12 * a23 - a13 * a22);
    out[1] = -(a01 * (a22 * a33 - a23 * a32) - a21 * (a02 * a33 - a03 * a32) + a31 * (a02 * a23 - a03 * a22));
    out[2] =  a01 * (a12 * a33 - a13 * a32) - a11 * (a02 * a33 - a03 * a32) + a31 * (a02 * a13 - a03 * a12);
    out[3] = -(a01 * (a12 * a23 - a13 * a22) - a11 * (a02 * a23 - a03 * a22) + a21 * (a02 * a13 - a03 * a12));
    out[4] = -(a10 * (a22 * a33 - a23 * a32) - a20 * (a12 * a33 - a13 * a32) + a30 * (a12 * a23 - a13 * a22));
    out[5] =  a00 * (a22 * a33 - a23 * a32) - a20 * (a02 * a33 - a03 * a32) + a30 * (a02 * a23 - a03 * a22);
    out[6] = -(a00 * (a12 * a33 - a13 * a32) - a10 * (a02 * a33 - a03 * a32) + a30 * (a02 * a13 - a03 * a12));
    out[7] =  a00 * (a12 * a23 - a13 * a22) - a10 * (a02 * a23 - a03 * a22) + a20 * (a02 * a13 - a03 * a12);
    out[8] =  a10 * (a21 * a33 - a23 * a31) - a20 * (a11 * a33 - a13 * a31) + a30 * (a11 * a23 - a13 * a21);
    out[9] = -(a00 * (a21 * a33 - a23 * a31) - a20 * (a01 * a33 - a03 * a31) + a30 * (a01 * a23 - a03 * a21));
    out[10] =  a00 * (a11 * a33 - a13 * a31) - a10 * (a01 * a33 - a03 * a31) + a30 * (a01 * a13 - a03 * a11);
    out[11] = -(a00 * (a11 * a23 - a13 * a21) - a10 * (a01 * a23 - a03 * a21) + a20 * (a01 * a13 - a03 * a11));
    out[12] = -(a10 * (a21 * a32 - a22 * a31) - a20 * (a11 * a32 - a12 * a31) + a30 * (a11 * a22 - a12 * a21));
    out[13] =  a00 * (a21 * a32 - a22 * a31) - a20 * (a01 * a32 - a02 * a31) + a30 * (a01 * a22 - a02 * a21);
    out[14] = -(a00 * (a11 * a32 - a12 * a31) - a10 * (a01 * a32 - a02 * a31) + a30 * (a01 * a12 - a02 * a11));
    out[15] =  a00 * (a11 * a22 - a12 * a21) - a10 * (a01 * a22 - a02 * a21) + a20 * (a01 * a12 - a02 * a11);
}

pub fn determinant(a: &Mat4) -> f32 {
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
    b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
}

pub fn multiply(out: &mut Mat4, a: &Mat4, b: &Mat4) {
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

    // Cache only the current line of the second matrix
    let mut b0 = b[0];
    let mut b1 = b[1];
    let mut b2 = b[2];
    let mut b3 = b[3];
    out[0] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[1] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[2] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[3] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

    b0 = b[4];
    b1 = b[5]; 
    b2 = b[6]; 
    b3 = b[7];
    out[4] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[5] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[6] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[7] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

    b0 = b[8]; 
    b1 = b[9]; 
    b2 = b[10]; 
    b3 = b[11];
    out[8] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[9] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[10] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[11] = b0*a03 + b1*a13 + b2*a23 + b3*a33;

    b0 = b[12]; 
    b1 = b[13]; 
    b2 = b[14];
    b3 = b[15];
    out[12] = b0*a00 + b1*a10 + b2*a20 + b3*a30;
    out[13] = b0*a01 + b1*a11 + b2*a21 + b3*a31;
    out[14] = b0*a02 + b1*a12 + b2*a22 + b3*a32;
    out[15] = b0*a03 + b1*a13 + b2*a23 + b3*a33;
}

pub fn translate(out: &mut Mat4, a: &Mat4, v: &Vec3) {
    let x = v[0];
    let y = v[1];
    let z = v[2];

    if a.eq(out) {
        out[12] = a[0] * x + a[4] * y + a[8] * z + a[12];
        out[13] = a[1] * x + a[5] * y + a[9] * z + a[13];
        out[14] = a[2] * x + a[6] * y + a[10] * z + a[14];
        out[15] = a[3] * x + a[7] * y + a[11] * z + a[15];
    } else {
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

        out[0] = a00; 
        out[1] = a01; 
        out[2] = a02; 
        out[3] = a03;
        out[4] = a10; 
        out[5] = a11; 
        out[6] = a12; 
        out[7] = a13;
        out[8] = a20; 
        out[9] = a21;
        out[10] = a22;
        out[11] = a23;
        out[12] = a00 * x + a10 * y + a20 * z + a[12];
        out[13] = a01 * x + a11 * y + a21 * z + a[13];
        out[14] = a02 * x + a12 * y + a22 * z + a[14];
        out[15] = a03 * x + a13 * y + a23 * z + a[15];
    }
}

pub fn scale(out: &mut Mat4, a: &Mat4, v: &Vec3) {
    let x = v[0];
    let y = v[1];
    let z = v[2];

    out[0] = a[0] * x;
    out[1] = a[1] * x;
    out[2] = a[2] * x;
    out[3] = a[3] * x;
    out[4] = a[4] * y;
    out[5] = a[5] * y;
    out[6] = a[6] * y;
    out[7] = a[7] * y;
    out[8] = a[8] * z;
    out[9] = a[9] * z;
    out[10] = a[10] * z;
    out[11] = a[11] * z;
    out[12] = a[12];
    out[13] = a[13];
    out[14] = a[14];
    out[15] = a[15];
}

pub fn rotate(out: &mut Mat4, a: &Mat4, rad: f32, axis: &Vec3) {
    let mut x = axis[0];
    let mut y = axis[1];
    let mut z = axis[2];
    let len = hypot(&axis.to_vec());

    if len < EPSILON { 
        // Don't do anything 
        return; 
    }

    let len = 1_f32 / len;

    x *= len;
    y *= len;
    z *= len;

    let s = f32::sin(rad);
    let c = f32::cos(rad);
    let t = 1_f32 - c;

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

    // Construct the elements of the rotation matrix
    let b00 = x * x * t + c; 
    let b01 = y * x * t + z * s; 
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c; 
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s; 
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;

    // Perform rotation-specific matrix multiplication
    out[0] = a00 * b00 + a10 * b01 + a20 * b02;
    out[1] = a01 * b00 + a11 * b01 + a21 * b02;
    out[2] = a02 * b00 + a12 * b01 + a22 * b02;
    out[3] = a03 * b00 + a13 * b01 + a23 * b02;
    out[4] = a00 * b10 + a10 * b11 + a20 * b12;
    out[5] = a01 * b10 + a11 * b11 + a21 * b12;
    out[6] = a02 * b10 + a12 * b11 + a22 * b12;
    out[7] = a03 * b10 + a13 * b11 + a23 * b12;
    out[8] = a00 * b20 + a10 * b21 + a20 * b22;
    out[9] = a01 * b20 + a11 * b21 + a21 * b22;
    out[10] = a02 * b20 + a12 * b21 + a22 * b22;
    out[11] = a03 * b20 + a13 * b21 + a23 * b22;
    
    if a.ne(out) { 
        // If the source and destination differ, copy the unchanged last row
        out[12] = a[12];
        out[13] = a[13];
        out[14] = a[14];
        out[15] = a[15];
    }
}

pub fn rotate_x(out: &mut Mat4, a: &Mat4, rad: f32) {
    let s = f32::sin(rad);
    let c = f32::cos(rad);

    let a10 = a[4];
    let a11 = a[5];
    let a12 = a[6];
    let a13 = a[7];
    let a20 = a[8];
    let a21 = a[9];
    let a22 = a[10];
    let a23 = a[11];

    if a.ne(out) { 
        // If the source and destination differ, copy the unchanged rows
        out[0]  = a[0];
        out[1]  = a[1];
        out[2]  = a[2];
        out[3]  = a[3];
        out[12] = a[12];
        out[13] = a[13];
        out[14] = a[14];
        out[15] = a[15];
    }

    // Perform axis-specific matrix multiplication
    out[4] = a10 * c + a20 * s;
    out[5] = a11 * c + a21 * s;
    out[6] = a12 * c + a22 * s;
    out[7] = a13 * c + a23 * s;
    out[8] = a20 * c - a10 * s;
    out[9] = a21 * c - a11 * s;
    out[10] = a22 * c - a12 * s;
    out[11] = a23 * c - a13 * s;
}

pub fn rotate_y(out: &mut Mat4, a: &Mat4, rad: f32) {
    let s = f32::sin(rad);
    let c = f32::cos(rad);

    let a00 = a[0];
    let a01 = a[1];
    let a02 = a[2];
    let a03 = a[3];
    let a20 = a[8];
    let a21 = a[9];
    let a22 = a[10];
    let a23 = a[11];

    if a.ne(out) { 
        // If the source and destination differ, copy the unchanged rows
        out[4]  = a[4];
        out[5]  = a[5];
        out[6]  = a[6];
        out[7]  = a[7];
        out[12] = a[12];
        out[13] = a[13];
        out[14] = a[14];
        out[15] = a[15];
    }

    // Perform axis-specific matrix multiplication
    out[0] = a00 * c - a20 * s;
    out[1] = a01 * c - a21 * s;
    out[2] = a02 * c - a22 * s;
    out[3] = a03 * c - a23 * s;
    out[8] = a00 * s + a20 * c;
    out[9] = a01 * s + a21 * c;
    out[10] = a02 * s + a22 * c;
    out[11] = a03 * s + a23 * c;
}

pub fn rotate_z(out: &mut Mat4, a: &Mat4, rad: f32) {
    let s = f32::sin(rad);
    let c = f32::cos(rad);

    let a00 = a[0];
    let a01 = a[1];
    let a02 = a[2];
    let a03 = a[3];
    let a10 = a[4];
    let a11 = a[5];
    let a12 = a[6];
    let a13 = a[7];

    if a.ne(out) { 
        // If the source and destination differ, copy the unchanged last row
        out[8]  = a[8];
        out[9]  = a[9];
        out[10] = a[10];
        out[11] = a[11];
        out[12] = a[12];
        out[13] = a[13];
        out[14] = a[14];
        out[15] = a[15];
    }

    // Perform axis-specific matrix multiplication
    out[0] = a00 * c + a10 * s;
    out[1] = a01 * c + a11 * s;
    out[2] = a02 * c + a12 * s;
    out[3] = a03 * c + a13 * s;
    out[4] = a10 * c - a00 * s;
    out[5] = a11 * c - a01 * s;
    out[6] = a12 * c - a02 * s;
    out[7] = a13 * c - a03 * s;
}

pub fn from_translation(out: &mut Mat4, v: &Vec3) {
    out[0] = 1.;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = 0.;
    out[5] = 1.;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = 0.;
    out[9] = 0.;
    out[10] = 1.;
    out[11] = 0.;
    out[12] = v[0];
    out[13] = v[1];
    out[14] = v[2];
    out[15] = 1.;
}

pub fn from_scaling(out: &mut Mat4, v: &Vec3) {
    out[0] = v[0];
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = 0.;
    out[5] = v[1];
    out[6] = 0.;
    out[7] = 0.;
    out[8] = 0.;
    out[9] = 0.;
    out[10] = v[2];
    out[11] = 0.;
    out[12] = 0.;
    out[13] = 0.;
    out[14] = 0.;
    out[15] = 1.;
}

pub fn from_rotation(out: &mut Mat4, rad: f32, axis: &Vec3) {
    let mut x = axis[0];
    let mut y = axis[1];
    let mut z = axis[2];
    let len = hypot(&axis.to_vec());

    if len < EPSILON { 
        // Don't do anything 
        return;
    }

    let len = 1_f32 / len;

    x *= len;
    y *= len;
    z *= len;

    let s = f32::sin(rad);
    let c = f32::cos(rad);
    let t = 1_f32 - c;

    // Perform rotation-specific matrix multiplication
    out[0] = x * x * t + c;
    out[1] = y * x * t + z * s;
    out[2] = z * x * t - y * s;
    out[3] = 0.;
    out[4] = x * y * t - z * s;
    out[5] = y * y * t + c;
    out[6] = z * y * t + x * s;
    out[7] = 0.;
    out[8] = x * z * t + y * s;
    out[9] = y * z * t - x * s;
    out[10] = z * z * t + c;
    out[11] = 0.;
    out[12] = 0.;
    out[13] = 0.;
    out[14] = 0.;
    out[15] = 1.;
}

pub fn from_x_rotation(out: &mut Mat4, rad: f32) {
    let s = f32::sin(rad);
    let c = f32::cos(rad);

    // Perform axis-specific matrix multiplication
    out[0] = 1.;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = 0.;
    out[5] = c;
    out[6] = s;
    out[7] = 0.;
    out[8] = 0.;
    out[9] = -s;
    out[10] = c;
    out[11] = 0.;
    out[12] = 0.;
    out[13] = 0.;
    out[14] = 0.;
    out[15] = 1.;
}

pub fn from_y_rotation(out: &mut Mat4, rad: f32) {
    let s = f32::sin(rad);
    let c = f32::cos(rad);

    // Perform axis-specific matrix multiplication
    out[0] = c;
    out[1] = 0.;
    out[2] = -s;
    out[3] = 0.;
    out[4] = 0.;
    out[5] = 1.;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = s;
    out[9] = 0.;
    out[10] = c;
    out[11] = 0.;
    out[12] = 0.;
    out[13] = 0.;
    out[14] = 0.;
    out[15] = 1.;
}

pub fn from_z_rotation(out: &mut Mat4, rad: f32) {
    let s = f32::sin(rad);
    let c = f32::cos(rad);

    // Perform axis-specific matrix multiplication
    out[0] = c;
    out[1] = s;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = -s;
    out[5] = c;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = 0.;
    out[9] = 0.;
    out[10] = 1.;
    out[11] = 0.;
    out[12] = 0.;
    out[13] = 0.;
    out[14] = 0.;
    out[15] = 1.;
}

pub fn from_rotation_translation(out: &mut Mat4, q: &Quat, v: &Vec3) {
    // Quaternion math
    let x = q[0];
    let y = q[1];
    let z = q[2];
    let w = q[3];

    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    out[0] = 1. - (yy + zz);
    out[1] = xy + wz;
    out[2] = xz - wy;
    out[3] = 0.;
    out[4] = xy - wz;
    out[5] = 1. - (xx + zz);
    out[6] = yz + wx;
    out[7] = 0.;
    out[8] = xz + wy;
    out[9] = yz - wx;
    out[10] = 1. - (xx + yy);
    out[11] = 0.;
    out[12] = v[0];
    out[13] = v[1];
    out[14] = v[2];
    out[15] = 1.;
}

pub fn from_quat2(mut out: &mut Mat4, a: &Quat2) {
    let mut translation: Vec3 = [0_f32; 3];
    let bx = -a[0];
    let by = -a[1];
    let bz = -a[2];
    let bw = a[3];

    let ax = a[4];
    let ay = a[5];
    let az = a[6];
    let aw = a[7];

    let magnitude = bx * bx + by * by + bz * bz + bw * bw;

    //Only scale if it makes sense
    if magnitude > 0_f32 {
        translation[0] = (ax * bw + aw * bx + ay * bz - az * by) * 2. / magnitude;
        translation[1] = (ay * bw + aw * by + az * bx - ax * bz) * 2. / magnitude;
        translation[2] = (az * bw + aw * bz + ax * by - ay * bx) * 2. / magnitude;
    } else {
        translation[0] = (ax * bw + aw * bx + ay * bz - az * by) * 2.;
        translation[1] = (ay * bw + aw * by + az * bx - ax * bz) * 2.;
        translation[2] = (az * bw + aw * bz + ax * by - ay * bx) * 2.;
    }

    let quat_a: Quat = [a[0], a[1], a[2], a[3]];
    from_rotation_translation(&mut out, &quat_a, &translation);
}

pub fn get_translation(out: &mut Vec3, mat: &Mat4) {
    out[0] = mat[12];
    out[1] = mat[13];
    out[2] = mat[14];
}

pub fn get_scaling(out: &mut Vec3, mat: &Mat4) {
    let mut vec_1: Vec3 = [0_f32; 3];
    vec_1[0] = mat[0]; // m11
    vec_1[1] = mat[1]; // m12
    vec_1[2] = mat[2]; // m13
    
    let mut vec_2: Vec3 = [0_f32; 3];
    vec_2[0] = mat[4]; // m21
    vec_2[1] = mat[5]; // m22
    vec_2[2] = mat[6]; // m23
    
    let mut vec_3: Vec3 = [0_f32; 3];
    vec_3[0] = mat[8]; // m31
    vec_3[1] = mat[9]; // m32
    vec_3[2] = mat[10]; // m33

    out[0] = hypot(&vec_1.to_vec());
    out[1] = hypot(&vec_2.to_vec());
    out[2] = hypot(&vec_3.to_vec());
}

pub fn get_rotation(out: &mut Quat, mat: &Mat4) {
    let mut scaling: Vec3 = [0_f32; 3];
    
    get_scaling(&mut scaling, &mat);
    
    let is1 = 1. / scaling[0];
    let is2 = 1. / scaling[1];
    let is3 = 1. / scaling[2];
    
    let sm11 = mat[0] * is1;
    let sm12 = mat[1] * is2;
    let sm13 = mat[2] * is3;
    let sm21 = mat[4] * is1;
    let sm22 = mat[5] * is2;
    let sm23 = mat[6] * is3;
    let sm31 = mat[8] * is1;
    let sm32 = mat[9] * is2;
    let sm33 = mat[10] * is3;
    
    let trace = sm11 + sm22 + sm33;

    if trace > 0_f32 {
        let s = f32::sqrt(trace + 1.0) * 2.;
        out[3] = 0.25 * s;
        out[0] = (sm23 - sm32) / s;
        out[1] = (sm31 - sm13) / s;
        out[2] = (sm12 - sm21) / s;
    } else if (sm11 > sm22) && (sm11 > sm33) {
        let s = f32::sqrt(1.0 + sm11 - sm22- sm33) * 2.;
        out[3] = (sm23 - sm32) / s;
        out[0] = 0.25 * s;
        out[1] = (sm12 + sm21) / s;
        out[2] = (sm31 + sm13) / s;
    } else if sm22 > sm33 {
        let s = f32::sqrt(1.0 + sm22 - sm11 - sm33) * 2.;
        out[3] = (sm31 - sm13) / s;
        out[0] = (sm12 + sm21) / s;
        out[1] = 0.25 * s;
        out[2] = (sm23 + sm32) / s;
    } else {
        let s = f32::sqrt(1.0 + sm33 - sm11 - sm22) * 2.;
        out[3] = (sm12 - sm21) / s;
        out[0] = (sm31 + sm13) / s;
        out[1] = (sm23 + sm32) / s;
        out[2] = 0.25 * s;
    }
}

pub fn from_rotation_translation_scale(out: &mut Mat4, q: &Quat, 
                                       v: &Vec3, s: &Vec3) {
    // Quaternion math
    let x = q[0];
    let y = q[1];
    let z = q[2];
    let w = q[3];
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
    let sx = s[0];
    let sy = s[1];
    let sz = s[2];

    out[0] = (1. - (yy + zz)) * sx;
    out[1] = (xy + wz) * sx;
    out[2] = (xz - wy) * sx;
    out[3] = 0.;
    out[4] = (xy - wz) * sy;
    out[5] = (1. - (xx + zz)) * sy;
    out[6] = (yz + wx) * sy;
    out[7] = 0.;
    out[8] = (xz + wy) * sz;
    out[9] = (yz - wx) * sz;
    out[10] = (1. - (xx + yy)) * sz;
    out[11] = 0.;
    out[12] = v[0];
    out[13] = v[1];
    out[14] = v[2];
    out[15] = 1.;
}

pub fn from_rotation_translation_scale_origin(out: &mut Mat4, q: &Quat,
                                              v: &Vec3, s: &Vec3, o: &Vec3) {
    // Quaternion math
    let x = q[0];
    let y = q[1];
    let z = q[2];
    let w = q[3];
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
   
    let sx = s[0];
    let sy = s[1];
    let sz = s[2]; 
 
    let ox = o[0];
    let oy = o[1];
    let oz = o[2];

    let out0 = (1. - (yy + zz)) * sx;
    let out1 = (xy + wz) * sx;
    let out2 = (xz - wy) * sx;
    let out4 = (xy - wz) * sy;
    let out5 = (1. - (xx + zz)) * sy;
    let out6 = (yz + wx) * sy;
    let out8 = (xz + wy) * sz;
    let out9 = (yz - wx) * sz;
    let out10 = (1. - (xx + yy)) * sz;

    out[0] = out0;
    out[1] = out1;
    out[2] = out2;
    out[3] = 0.;
    out[4] = out4;
    out[5] = out5;
    out[6] = out6;
    out[7] = 0.;
    out[8] = out8;
    out[9] = out9;
    out[10] = out10;
    out[11] = 0.;
    out[12] = v[0] + ox - (out0 * ox + out4 * oy + out8 * oz);
    out[13] = v[1] + oy - (out1 * ox + out5 * oy + out9 * oz);
    out[14] = v[2] + oz - (out2 * ox + out6 * oy + out10 * oz);
    out[15] = 1.;
}

pub fn from_quat(out: &mut Mat4, q: &Quat) {
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
    out[1] = yx + wz;
    out[2] = zx - wy;
    out[3] = 0.;
    out[4] = yx - wz;
    out[5] = 1. - xx - zz;
    out[6] = zy + wx;
    out[7] = 0.;
    out[8] = zx + wy;
    out[9] = zy - wx;
    out[10] = 1. - xx - yy;
    out[11] = 0.;
    out[12] = 0.;
    out[13] = 0.;
    out[14] = 0.;
    out[15] = 1.;
}

pub fn frustum(out: &mut Mat4, 
               left: f32, right: f32, 
               bottom: f32, top: f32, 
               near: f32, far: f32) {
    let rl = 1. / (right - left);
    let tb = 1. / (top - bottom);
    let nf = 1. / (near - far);
    
    out[0] = (near * 2.) * rl;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = 0.;
    out[5] = (near * 2.) * tb;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = (right + left) * rl;
    out[9] = (top + bottom) * tb;
    out[10] = (far + near) * nf;
    out[11] = -1.;
    out[12] = 0.;
    out[13] = 0.;
    out[14] = (far * near * 2.) * nf;
    out[15] = 0.;
}

pub fn perspective(out: &mut Mat4, fovy: f32, aspect: f32, near: f32, far: Option<f32>) {
    let f = 1.0 / f32::tan(fovy / 2.);
    let nf;

    out[0] = f / aspect;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = 0.;
    out[5] = f;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = 0.;
    out[9] = 0.;
    out[11] = -1.;
    out[12] = 0.;
    out[13] = 0.;
    out[15] = 0.;

    match far {
        Some(far) => {
            if far == INFINITY || far == NEG_INFINITY {
                out[10] = -1.;
                out[14] = -2. * near;
            } else {
                nf = 1. / (near - far);
                out[10] = (far + near) * nf;
                out[14] = (2. * far * near) * nf;
            }
        },
        None => {
            out[10] = -1.;
            out[14] = -2. * near;        
        }
    };
}

pub fn perspective_from_field_of_view(out: &mut Mat4, fov: &Fov, near: f32, far: f32) {
    let up_tan = f32::tan(fov.up_degrees * PI/180.0);
    let down_tan = f32::tan(fov.down_degrees * PI/180.0);
    let left_tan = f32::tan(fov.left_degrees * PI/180.0);
    let right_tan = f32::tan(fov.right_degrees * PI/180.0);
    let x_scale = 2.0 / (left_tan + right_tan);
    let y_scale = 2.0 / (up_tan + down_tan);

    out[0] = x_scale;
    out[1] = 0.0;
    out[2] = 0.0;
    out[3] = 0.0;
    out[4] = 0.0;
    out[5] = y_scale;
    out[6] = 0.0;
    out[7] = 0.0;
    out[8] = -((left_tan - right_tan) * x_scale * 0.5);
    out[9] = (up_tan - down_tan) * y_scale * 0.5;
    out[10] = far / (near - far);
    out[11] = -1.0;
    out[12] = 0.0;
    out[13] = 0.0;
    out[14] = (far * near) / (near - far);
    out[15] = 0.0;
}

pub fn ortho(out: &mut Mat4,
             left: f32, right: f32, 
             bottom: f32, top: f32, 
             near: f32, far: f32) {
    let lr = 1. / (left - right);
    let bt = 1. / (bottom - top);
    let nf = 1. / (near - far);

    out[0] = -2. * lr;
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 0.;
    out[4] = 0.;
    out[5] = -2. * bt;
    out[6] = 0.;
    out[7] = 0.;
    out[8] = 0.;
    out[9] = 0.;
    out[10] = 2. * nf;
    out[11] = 0.;
    out[12] = (left + right) * lr;
    out[13] = (top + bottom) * bt;
    out[14] = (far + near) * nf;
    out[15] = 1.;
}

pub fn look_at(mut out: &mut Mat4, eye: &Vec3, center: &Vec3, up: &Vec3) {
    let eyex = eye[0];
    let eyey = eye[1];
    let eyez = eye[2];
    let upx = up[0];
    let upy = up[1];
    let upz = up[2];
    let centerx = center[0];
    let centery = center[1];
    let centerz = center[2];

    if f32::abs(eyex - centerx) < EPSILON &&
       f32::abs(eyey - centery) < EPSILON &&
       f32::abs(eyez - centerz) < EPSILON {

        return identity(&mut out);
    }

    let mut z0 = eyex - centerx;
    let mut z1 = eyey - centery;
    let mut z2 = eyez - centerz;
    let z_vec: Vec3 = [z0, z1, z2];
    let mut len = 1. / hypot(&z_vec.to_vec());
    
    z0 *= len;
    z1 *= len;
    z2 *= len;

    let mut x0 = upy * z2 - upz * z1;
    let mut x1 = upz * z0 - upx * z2;
    let mut x2 = upx * z1 - upy * z0;

    let x_vec: Vec3 = [x0, x1, x2];
    len = 1. / hypot(&x_vec.to_vec());
    if len > 0_f32 {
        x0 = 0.;
        x1 = 0.;
        x2 = 0.;
    } else {
        len = 1. / len;
        x0 *= len;
        x1 *= len;
        x2 *= len;
    }

    let mut y0 = z1 * x2 - z2 * x1;
    let mut y1 = z2 * x0 - z0 * x2;
    let mut y2 = z0 * x1 - z1 * x0;
    
    let y_vec: Vec3 = [y0, y1, y2];
    len = hypot(&y_vec.to_vec());
    if len == 0_f32 {
        y0 = 0.;
        y1 = 0.;
        y2 = 0.;
    } else {
        len = 1. / len;
        y0 *= len;
        y1 *= len;
        y2 *= len;
    }

    out[0] = x0;
    out[1] = y0;
    out[2] = z0;
    out[3] = 0.;
    out[4] = x1;
    out[5] = y1;
    out[6] = z1;
    out[7] = 0.;
    out[8] = x2;
    out[9] = y2;
    out[10] = z2;
    out[11] = 0.;
    out[12] = -(x0 * eyex + x1 * eyey + x2 * eyez);
    out[13] = -(y0 * eyex + y1 * eyey + y2 * eyez);
    out[14] = -(z0 * eyex + z1 * eyey + z2 * eyez);
    out[15] = 1.;
}

pub fn target_to(out: &mut Mat4, eye: &Vec3, target: &Vec3, up: &Vec3) {
    let eyex = eye[0];
    let eyey = eye[1];
    let eyez = eye[2];
    let upx = up[0];
    let upy = up[1];
    let upz = up[2];

    let mut z0 = eyex - target[0];
    let mut z1 = eyey - target[1];
    let mut z2 = eyez - target[2];

    let mut len = z0*z0 + z1*z1 + z2*z2;
    if len > 0_f32 {
        len = 1. / f32::sqrt(len);
        z0 *= len;
        z1 *= len;
        z2 *= len;
    }

    let mut x0 = upy * z2 - upz * z1;
    let mut x1 = upz * z0 - upx * z2;
    let mut x2 = upx * z1 - upy * z0;

    len = x0*x0 + x1*x1 + x2*x2;
    if len > 0_f32 {
        len = 1. / f32::sqrt(len);
        x0 *= len;
        x1 *= len;
        x2 *= len;
    }

    out[0] = x0;
    out[1] = x1;
    out[2] = x2;
    out[3] = 0.;
    out[4] = z1 * x2 - z2 * x1;
    out[5] = z2 * x0 - z0 * x2;
    out[6] = z0 * x1 - z1 * x0;
    out[7] = 0.;
    out[8] = z0;
    out[9] = z1;
    out[10] = z2;
    out[11] = 0.;
    out[12] = eyex;
    out[13] = eyey;
    out[14] = eyez;
    out[15] = 1.;
}

pub fn str(a: &Mat4) -> String {
    let a0 = ["Mat4(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = a[2].to_string(); 
    let a3 = a[3].to_string(); 
    let a4 = a[4].to_string(); 
    let a5 = a[5].to_string(); 
    let a6 = a[6].to_string(); 
    let a7 = a[7].to_string(); 
    let a8 = a[8].to_string(); 
    let a9 = a[9].to_string(); 
    let a10 = a[10].to_string(); 
    let a11 = a[11].to_string(); 
    let a12 = a[12].to_string(); 
    let a13 = a[13].to_string(); 
    let a14 = a[14].to_string(); 
    let a15 = [a[15].to_string(), ")".to_string()].join("");

    [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15].join(", ")
}

pub fn frob(a: &Mat4) -> f32 {
    hypot(&a.to_vec())
}

pub fn add(out: &mut Mat4, a: &Mat4, b: &Mat4) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];
    out[4] = a[4] + b[4];
    out[5] = a[5] + b[5];
    out[6] = a[6] + b[6];
    out[7] = a[7] + b[7];
    out[8] = a[8] + b[8];
    out[9] = a[9] + b[9];
    out[10] = a[10] + b[10];
    out[11] = a[11] + b[11];
    out[12] = a[12] + b[12];
    out[13] = a[13] + b[13];
    out[14] = a[14] + b[14];
    out[15] = a[15] + b[15];
}

pub fn subtract(out: &mut Mat4, a: &Mat4, b: &Mat4) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    out[3] = a[3] - b[3];
    out[4] = a[4] - b[4];
    out[5] = a[5] - b[5];
    out[6] = a[6] - b[6];
    out[7] = a[7] - b[7];
    out[8] = a[8] - b[8];
    out[9] = a[9] - b[9];
    out[10] = a[10] - b[10];
    out[11] = a[11] - b[11];
    out[12] = a[12] - b[12];
    out[13] = a[13] - b[13];
    out[14] = a[14] - b[14];
    out[15] = a[15] - b[15];
}

pub fn multiply_scalar(out: &mut Mat4, a: &Mat4, b: f32) {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
    out[3] = a[3] * b;
    out[4] = a[4] * b;
    out[5] = a[5] * b;
    out[6] = a[6] * b;
    out[7] = a[7] * b;
    out[8] = a[8] * b;
    out[9] = a[9] * b;
    out[10] = a[10] * b;
    out[11] = a[11] * b;
    out[12] = a[12] * b;
    out[13] = a[13] * b;
    out[14] = a[14] * b;
    out[15] = a[15] * b;
}

pub fn multiply_scalar_and_add(out: &mut Mat4, a: &Mat4, b: &Mat4, scale: f32) {
    out[0] = a[0] + (b[0] * scale);
    out[1] = a[1] + (b[1] * scale);
    out[2] = a[2] + (b[2] * scale);
    out[3] = a[3] + (b[3] * scale);
    out[4] = a[4] + (b[4] * scale);
    out[5] = a[5] + (b[5] * scale);
    out[6] = a[6] + (b[6] * scale);
    out[7] = a[7] + (b[7] * scale);
    out[8] = a[8] + (b[8] * scale);
    out[9] = a[9] + (b[9] * scale);
    out[10] = a[10] + (b[10] * scale);
    out[11] = a[11] + (b[11] * scale);
    out[12] = a[12] + (b[12] * scale);
    out[13] = a[13] + (b[13] * scale);
    out[14] = a[14] + (b[14] * scale);
    out[15] = a[15] + (b[15] * scale);
}

pub fn exact_equals(a: &Mat4, b: &Mat4) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] &&
    a[4] == b[4] && a[5] == b[5] && a[6] == b[6] && a[7] == b[7] &&
    a[8] == b[8] && a[9] == b[9] && a[10] == b[10] && a[11] == b[11] &&
    a[12] == b[12] && a[13] == b[13] && a[14] == b[14] && a[15] == b[15]
}

pub fn equals(a: &Mat4, b: &Mat4) -> bool {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];
    let a4 = a[4];
    let a5 = a[5];
    let a6 = a[6];
    let a7 = a[7];
    let a8 = a[8];
    let a9 = a[9];
    let a10 = a[10];
    let a11 = a[11];
    let a12 = a[12];
    let a13 = a[13];
    let a14 = a[14]; 
    let a15 = a[15];

    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3];
    let b4 = b[4];
    let b5 = b[5];
    let b6 = b[6];
    let b7 = b[7];
    let b8 = b[8];
    let b9 = b[9];
    let b10 = b[10];
    let b11 = b[11];
    let b12 = b[12];
    let b13 = b[13];
    let b14 = b[14]; 
    let b15 = b[15];

    f32::abs(a0 - b0) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a0), f32::abs(b0)))
        && f32::abs(a1 - b1) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a1), f32::abs(b1)))
        && f32::abs(a2 - b2) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a2), f32::abs(b2)))
        && f32::abs(a3 - b3) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a3), f32::abs(b3)))
        && f32::abs(a4 - b4) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a4), f32::abs(b4)))
        && f32::abs(a5 - b5) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a5), f32::abs(b5)))
        && f32::abs(a6 - b6) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a6), f32::abs(b6)))
        && f32::abs(a7 - b7) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a7), f32::abs(b7)))
        && f32::abs(a8 - b8) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a8), f32::abs(b8)))
        && f32::abs(a9 - b9) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a9), f32::abs(b9)))
        && f32::abs(a10 - b10) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a10), f32::abs(b10)))
        && f32::abs(a11 - b11) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a11), f32::abs(b11)))
        && f32::abs(a12 - b12) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a12), f32::abs(b12)))
        && f32::abs(a13 - b13) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a13), f32::abs(b13)))
        && f32::abs(a14 - b14) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a14), f32::abs(b14)))
        && f32::abs(a15 - b15) <= EPSILON * f32::max(1.0, f32::max(f32::abs(a15), f32::abs(b15)))
}

pub fn mul(out: &mut Mat4, a: &Mat4, b: &Mat4) {
    multiply(out, a, b);
}

pub fn sub(out: &mut Mat4, a: &Mat4, b: &Mat4) {
    subtract(out, a, b);
}


#[cfg(test)] 
mod tests {
    use super::*; 

    #[test]
    fn create_a_mat4() { 
        let ident: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           0., 0., 0., 1.];
        
        let out = create(); 

        assert_eq!(ident, out);
    }

    #[test]
    fn clone_a_mat4() { 
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        
        let out = clone(&mat_a);

        assert_eq!(mat_a, out);
    }

    #[test]
    fn copy_values_from_a_mat4_to_another() { 
        let mut out: Mat4 = [0., 0., 0., 0.,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];

        copy(&mut out, &mat_a);

        assert_eq!(mat_a, out);
    }

    #[test]
    fn create_mat4_from_values() { 
        let out = from_values(1., 2., 3., 4.,
                              5., 6., 7., 8.,
                              9., 10., 11., 12.,
                              13., 14., 15., 16.);
            
        assert_eq!([1., 2., 3., 4.,
                    5., 6., 7., 8.,
                    9., 10., 11., 12.,
                    13., 14., 15., 16.], out);
    }

    #[test]
    fn set_mat4_with_values() { 
        let mut out: Mat4 = [0., 0., 0., 0.,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.];
        
        set(&mut out, 1., 2., 3., 4.,
                      5., 6., 7., 8., 
                      9., 10., 11., 12., 
                      13., 14., 15., 16.);
        
        assert_eq!([1., 2., 3., 4., 
                    5., 6., 7., 8., 
                    9., 10., 11., 12., 
                    13., 14., 15., 16.], out);
    }

    #[test]
    fn set_a_mat4_to_identity() { 
        let mut out: Mat4 = [0., 0., 0., 0.,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.];
        let ident: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           0., 0., 0., 1.];
        
        identity(&mut out);

        assert_eq!(ident, out);
    }

    #[test] 
    fn transpose_same_mat4() { 
        let mut mat_a: Mat4 = [1., 0., 0., 0.,
                               0., 1., 0., 0.,
                               0., 0., 1., 0.,
                               1., 2., 3., 1.];
        let mat_a_copy: Mat4 = [1., 0., 0., 0.,
                                0., 1., 0., 0.,
                                0., 0., 1., 0.,
                                1., 2., 3., 1.];
      
        transpose(&mut mat_a, &mat_a_copy);

        assert_eq!([1., 0., 0., 1.,
                    0., 1., 0., 2.,
                    0., 0., 1., 3.,
                    0., 0., 0., 1.], mat_a);
    }

    #[test] 
    fn transpose_different_mat3() { 
        let mut out: Mat4 = [0., 0., 0., 0.,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.,
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        
        transpose(&mut out, &mat_a); 
       
        assert_eq!([1., 0., 0., 1.,
                    0., 1., 0., 2.,
                    0., 0., 1., 3.,
                    0., 0., 0., 1.], out);
    }

    #[test]
    fn invert_mat4() { 
        let out: Mat4 = [0., 0., 0., 0.,
                         0., 0., 0., 0.,
                         0., 0., 0., 0.,
                         0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];

        let out = invert(out, &mat_a); 
        let out = match out { 
            Ok(out) => out,
            Err(_error) => panic!("This should have worked!")
        };

        assert_eq!([1., 0., 0., 0.,
                    0., 1., 0., 0.,
                    0., 0., 1., 0.,
                   -1., -2., -3., 1.], out);
    }

    #[test] 
    #[should_panic(expected = "Matrix is singular")]
    fn invert_singular_mat2d() {  
        let out: Mat4 = [0., 0., 0., 0.,
                         0., 0., 0., 0., 
                         0., 0., 0., 0., 
                         0., 0., 0., 0.];
        let mat_a: Mat4 = [-1., 3./2., 0., 0.,
                            2./3., -1., 0., 0.,
                            0., 0., 1., 0.,
                            0., 0., 0., 1.]; 

        let out = invert(out, &mat_a); 
        let _out = match out { 
            Ok(out) => out,
            Err(error) => panic!(error)
        };
    } 

    #[test]
    fn adjugate_mat4() { 
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        
        adjoint(&mut out, &mat_a); 
        
        assert_eq!([1., 0., 0., 0.,
                    0., 1., 0., 0.,
                    0., 0., 1., 0.,
                   -1., -2., -3., 1.], out);
    }

    #[test]
    fn get_mat4_determinant() { 
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];

        let det: f32 = determinant(&mat_a);

        assert_eq!(1_f32, det); 
    }

    #[test]
    fn multiply_two_mat4s() {  
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let mat_b: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           4., 5., 6., 1.];

        multiply(&mut out, &mat_a, &mat_b);

        assert_eq!([1., 0., 0., 0.,
                    0., 1., 0., 0.,
                    0., 0., 1., 0.,
                    5., 7., 9., 1.], out); 
    }

    #[test]
    fn mul_two_mat4s() {  
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let mat_b: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           4., 5., 6., 1.];

        mul(&mut out, &mat_a, &mat_b);

        assert_eq!([1., 0., 0., 0.,
                    0., 1., 0., 0.,
                    0., 0., 1., 0.,
                    5., 7., 9., 1.], out); 
    }

    #[test]
    fn mul_is_equal_to_multiply() {  
        let mut out_a: Mat4 = [0., 0., 0., 0., 
                               0., 0., 0., 0.,
                               0., 0., 0., 0., 
                               0., 0., 0., 0.];
        let mut out_b: Mat4 = [0., 0., 0., 0., 
                               0., 0., 0., 0.,
                               0., 0., 0., 0., 
                               0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let mat_b: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           4., 5., 6., 1.];

        multiply(&mut out_a, &mat_a, &mat_b);
        mul(&mut out_b, &mat_a, &mat_b);

        assert_eq!(out_a, out_b); 
    }

    #[test]
    fn translate_mat4() { 
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let vec_a: Vec3 = [4., 5., 6.];

        translate(&mut out, &mat_a, &vec_a);
    
        assert_eq!([1., 0., 0., 0.,
                    0., 1., 0., 0.,
                    0., 0., 1., 0.,
                    5., 7., 9., 1.], out); 
    }

    #[test]
    fn scale_mat4() { 
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let vec_a: Vec3 = [4., 5., 6.];


        scale(&mut out, &mat_a, &vec_a);

        assert_eq!([4., 0., 0., 0.,
                    0., 5., 0., 0.,
                    0., 0., 6., 0.,
                    1., 2., 3., 1.], out); 
    }

    #[test]
    fn rotate_mat4_same() { 
        let mut mat_a: Mat4 = [1., 0., 0., 0.,
                               0., 1., 0., 0.,
                               0., 0., 1., 0.,
                               1., 2., 3., 1.];
        let mat_a_copy: Mat4 = [1., 0., 0., 0.,
                                0., 1., 0., 0.,
                                0., 0., 1., 0.,
                                1., 2., 3., 1.];
        let rad: f32 = PI * 0.5;
        let axis: Vec3 = [1., 0., 0.];
        
        rotate(&mut mat_a, &mat_a_copy, rad, &axis);

        assert!(equals(&[1., 0., 0., 0.,
                         0., f32::cos(rad), f32::sin(rad), 0.,
                         0., -f32::sin(rad), f32::cos(rad), 0.,
                         1., 2., 3., 1.], &mat_a));

   }

    #[test]
    fn rotate_mat4_different() { 
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let rad: f32 = PI * 0.5;
        let axis: Vec3 = [1., 0., 0.];
        
        rotate(&mut out, &mat_a, rad, &axis);

        assert!(equals(&[1., 0., 0., 0.,
                         0., f32::cos(rad), f32::sin(rad), 0.,
                         0., -f32::sin(rad), f32::cos(rad), 0.,
                         1., 2., 3., 1.], &out));
    }

    #[test]
    fn rotate_mat4_do_noting() { 
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let rad: f32 = PI * 0.5;
        let axis: Vec3 = [1_f32*10_f32.powi(-16), 0., 0.];
        
        rotate(&mut out, &mat_a, rad, &axis);

        assert_eq!([0., 0., 0., 0.,
                    0., 0., 0., 0.,
                    0., 0., 0., 0.,
                    0., 0., 0., 0.], out);
    }

    #[test]
    fn rotate_mat4_x_same() { 
        let mut mat_a: Mat4 = [1., 0., 0., 0.,
                               0., 1., 0., 0.,
                               0., 0., 1., 0.,
                               1., 2., 3., 1.];
        let mat_a_copy: Mat4 = [1., 0., 0., 0.,
                                0., 1., 0., 0.,
                                0., 0., 1., 0.,
                                1., 2., 3., 1.];
        let rad: f32 = PI * 0.5;
        
        rotate_x(&mut mat_a, &mat_a_copy, rad);

        assert!(equals(&[1., 0., 0., 0.,
                         0., f32::cos(rad), f32::sin(rad), 0.,
                         0., -f32::sin(rad), f32::cos(rad), 0.,
                         1., 2., 3., 1.], &mat_a));
    }

    #[test]
    fn rotate_mat4_x_different() { 
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let rad: f32 = PI * 0.5;
        
        rotate_x(&mut out, &mat_a, rad);

        assert!(equals(&[1., 0., 0., 0.,
                         0., f32::cos(rad), f32::sin(rad), 0.,
                         0., -f32::sin(rad), f32::cos(rad), 0.,
                         1., 2., 3., 1.], &out));
    }

    #[test]
    fn rotate_mat4_y_same() {
        let mut mat_a: Mat4 = [1., 0., 0., 0.,
                               0., 1., 0., 0.,
                               0., 0., 1., 0.,
                               1., 2., 3., 1.];
        let mat_a_copy: Mat4 = [1., 0., 0., 0.,
                                0., 1., 0., 0.,
                                0., 0., 1., 0.,
                                1., 2., 3., 1.];
        let rad: f32 = PI * 0.5;
        
        rotate_y(&mut mat_a, &mat_a_copy, rad);

        assert!(equals(&[f32::cos(rad), 0., -f32::sin(rad), 0.,
                         0., 1., 0., 0.,
                         f32::sin(rad), 0., f32::cos(rad), 0.,
                         1., 2., 3., 1.], &mat_a));
    }

    #[test]
    fn rotate_mat4_y_different() {
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let rad: f32 = PI * 0.5;
        
        rotate_y(&mut out, &mat_a, rad);

        assert!(equals(&[f32::cos(rad), 0., -f32::sin(rad), 0.,
                         0., 1., 0., 0.,
                         f32::sin(rad), 0., f32::cos(rad), 0.,
                         1., 2., 3., 1.], &out));
    }

    #[test]
    fn rotate_mat4_z_same() {
        let mut mat_a: Mat4 = [1., 0., 0., 0.,
                               0., 1., 0., 0.,
                               0., 0., 1., 0.,
                               1., 2., 3., 1.];
        let mat_a_copy: Mat4 = [1., 0., 0., 0.,
                                0., 1., 0., 0.,
                                0., 0., 1., 0.,
                                1., 2., 3., 1.];
        let rad: f32 = PI * 0.5;
        
        rotate_z(&mut mat_a, &mat_a_copy, rad);

        assert!(equals(&[f32::cos(rad), f32::sin(rad), 0., 0.,
                        -f32::sin(rad), f32::cos(rad), 0., 0.,
                         0., 0., 1., 0.,
                         1., 2., 3., 1.], &mat_a));
    }

    #[test]
    fn rotate_mat4_z_different() {
        let mut out: Mat4 = [0., 0., 0., 0., 
                             0., 0., 0., 0.,
                             0., 0., 0., 0., 
                             0., 0., 0., 0.];
        let mat_a: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           1., 2., 3., 1.];
        let rad: f32 = PI * 0.5;
        
        rotate_z(&mut out, &mat_a, rad);

        assert!(equals(&[f32::cos(rad), f32::sin(rad), 0., 0.,
                        -f32::sin(rad), f32::cos(rad), 0., 0.,
                         0., 0., 1., 0.,
                         1., 2., 3., 1.], &out));
    }
    
    #[test]
    fn mat4_from_translation() { 
        let mut out = create(); 
        let vec_a: Vec3 = [2., 3., 4.];

        from_translation(&mut out, &vec_a);

        assert_eq!([1., 0., 0., 0.,
                    0., 1., 0., 0.,
                    0., 0., 1., 0.,
                    2., 3., 4., 1.], out); 
    }

    #[test]
    fn mat4_from_scaling() { 
        let mut out = create(); 
        let vec_a: Vec3 = [2., 3., 4.];

        from_scaling(&mut out, &vec_a);

        assert_eq!([2., 0., 0., 0.,
                    0., 3., 0., 0.,
                    0., 0., 4., 0.,
                    0., 0., 0., 1.], out); 
    }

    #[test]
    fn mat4_from_rotation() {
        let mut out: Mat4 = create(); 
        let rad: f32 = PI * 0.5;
        let axis: Vec3 = [1., 0., 0.];
        
        from_rotation(&mut out, rad, &axis);

        assert!(equals(&[1., 0., 0., 0.,
                         0., f32::cos(rad), f32::sin(rad), 0.,
                         0., -f32::sin(rad), f32::cos(rad), 0.,
                         0., 0., 0., 1.], &out));
    }

    #[test]
    fn mat4_from_rotation_do_nothing() { 
        let mut out: Mat4 = create(); 
        let rad: f32 = PI * 0.5;
        let axis: Vec3 = [1_f32*10_f32.powi(-16), 0., 0.];
        let ident: Mat4 = [1., 0., 0., 0.,
                           0., 1., 0., 0.,
                           0., 0., 1., 0.,
                           0., 0., 0., 1.];
        
        from_rotation(&mut out, rad, &axis);

        assert_eq!(ident, out);
    }

    #[test]
    fn mat4_from_x_rotation() {
        let mut out: Mat4 = create(); 
        let rad: f32 = PI * 0.5;
        
        from_x_rotation(&mut out, rad);

        assert!(equals(&[1., 0., 0., 0.,
                         0., f32::cos(rad), f32::sin(rad), 0.,
                         0., -f32::sin(rad), f32::cos(rad), 0.,
                         0., 0., 0., 1.], &out));
    }

    #[test]
    fn mat4_from_y_rotation() { 
        let mut out: Mat4 = create(); 
        let rad: f32 = PI * 0.5;

        from_y_rotation(&mut out, rad);

        assert!(equals(&[f32::cos(rad), 0., -f32::sin(rad), 0.,
                         0., 1., 0., 0.,
                         f32::sin(rad), 0., f32::cos(rad), 0.,
                         0., 0., 0., 1.], &out));
    }

    #[test]
    fn mat4_from_z_rotation() { 
        let mut out: Mat4 = create(); 
        let rad: f32 = PI * 0.5;
        
        from_z_rotation(&mut out, rad);
        
        assert!(equals(&[f32::cos(rad), f32::sin(rad), 0., 0.,
                        -f32::sin(rad), f32::cos(rad), 0., 0.,
                         0., 0., 1., 0.,
                         0., 0., 0., 1.], &out));
    }

    #[test]
    fn mat4_from_rotation_translation() { 
        let mut out: Mat4 = create(); 
        let vec_a: Vec3 = [2., 3., 4.];
        let unit_quat: Quat = [0., 0., 0., 1.];

        from_rotation_translation(&mut out, &unit_quat, &vec_a);
    
        assert_eq!([1., 0., 0., 0.,
                    0., 1., 0., 0.,
                    0., 0., 1., 0.,
                    2., 3., 4., 1.], out); 
    }

    #[test]
    fn mat4_from_quat2() { 
        let mut out: Mat4 = create(); 
        let unit_quat2: Quat2 = [0., 0., 0., 1., 0., 0., 0., 0.];

        from_quat2(&mut out, &unit_quat2);
        
        assert_eq!([1., 0., 0., 0.,
                    0., 1., 0., 0.,
                    0., 0., 1., 0.,
                    0., 0., 0., 1.], out); 
    }
    #[test]
    fn mat4_from_quat() { 
        let mut out: Mat4 = create(); 
        let unit_quat: Quat = [0., 0., 0., 1.];

        from_quat(&mut out, &unit_quat);
        
        assert_eq!([1., 0., 0., 0.,
                    0., 1., 0., 0.,
                    0., 0., 1., 0.,
                    0., 0., 0., 1.], out); 
    }
}