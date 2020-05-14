//! Quaternion
 
use super::common::{Quat, Vec4, Vec3, Mat3, random_f32, EPSILON, PI};
use super::{vec3, vec4, mat3};

/// Creates a new identity quat.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn create() -> Quat {
    let mut out: Quat = [0_f32; 4];

    out[3] = 1.;

    out
}

/// Creates a new identity quat.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn identity(out: &mut Quat) -> Quat {
    out[0] = 0.; 
    out[1] = 0.;
    out[2] = 0.;
    out[3] = 1.;

    *out
}

/// Sets a quat from the given angle and rotation axis,
/// then returns it.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn set_axis_angle(out: &mut Quat, axis: &Vec3, rad: f32) -> Quat {
    let rad = rad * 0.5;
    let s = f32::sin(rad);
   
    out[0] = s * axis[0];
    out[1] = s * axis[1];
    out[2] = s * axis[2];
    out[3] = f32::cos(rad);

    *out
}

/// Gets the rotation axis and angle for a given
/// quaternion. If a quaternion is created with
/// setAxisAngle, this method will return the same
/// values as providied in the original parameter list
/// OR functionally equivalent values.
/// 
/// Example: The quaternion formed by axis [0, 0, 1] and
/// angle -90 is the same as the quaternion formed by
/// [0, 0, 1] and 270. This method favors the latter.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
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

/// Gets the angular distance between two unit quaternions.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn get_angle(a: &Quat, b: &Quat) -> f32 {
    let dot_product = dot(a, b);

    f32::acos(2_f32 * dot_product * dot_product - 1_f32)
}

/// Multiplies two quat's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn multiply(out: &mut Quat, a: &Quat, b: &Quat) -> Quat {
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

    *out
}

/// Rotates a quaternion by the given angle about the X axis.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn rotate_x(out: &mut Quat, a: &Quat, rad: f32) -> Quat {
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

    *out
}

/// Rotates a quaternion by the given angle about the Y axis.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn rotate_y(out: &mut Quat, a: &Quat, rad: f32) -> Quat {
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

    *out
}

/// Rotates a quaternion by the given angle about the Z axis.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn rotate_z(out: &mut Quat, a: &Quat, rad: f32) -> Quat {
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

    *out
}

/// Calculates the W component of a quat from the X, Y, and Z components.
/// Assumes that quaternion is 1 unit in length.
/// Any existing W component will be ignored.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn calculate_w(out: &mut Quat, a: &Quat) -> Quat {
    let x = a[0];
    let y = a[1];
    let z = a[2];

    out[0] = x;
    out[1] = y;
    out[2] = z;
    out[3] = f32::sqrt(f32::abs(1_f32 - x * x - y * y - z * z));

    *out
}

/// Calculate the exponential of a unit quaternion.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn exp(out: &mut Quat, a: &Quat) -> Quat {
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

    *out
}

/// Calculate the natural logarithm of a unit quaternion.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn ln(out: &mut Quat, a: &Quat) -> Quat {
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

    *out
}

/// Calculate the scalar power of a unit quaternion.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn pow(out: &mut Quat, a: &Quat, b: f32) -> Quat {
    ln(out, a);
    scale(out, &clone(out), b);
    exp(out, &clone(out))
}

/// Performs a spherical linear interpolation between two quats.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn slerp(out: &mut Quat, a: &Quat, b: &Quat, t: f32) -> Quat {
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

    let omega: f32;
    let mut cosom: f32;
    let sinom: f32;
    let scale0: f32;
    let scale1: f32;

    // calc cosine
    cosom = ax * bx + ay * by + az * bz + aw * bw;

    // adjust signs (if necessary)
    if cosom < 0.{
        cosom = -cosom;
        bx = -bx;
        by = -by;
        bz = -bz;
        bw = -bw;
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

    *out
}

/// Generates a random unit quaternion
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn random(out: &mut Quat) -> Quat {
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

    *out
}

/// Calculates the inverse of a quat.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn invert(out: &mut Quat, a: &Quat) -> Quat {
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

    *out
}

/// Calculates the conjugate of a quat
/// If the quaternion is normalized, this function is faster than quat.inverse and produces the same result.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
pub fn conjugate(out: &mut Quat, a: &Quat) -> Quat {
    out[0] = -a[0];
    out[1] = -a[1];
    out[2] = -a[2];
    out[3] = a[3];

    *out
}

/// Creates a quaternion from the given 3x3 rotation matrix.
///
/// NOTE: The resultant quaternion is not normalized, so you should be sure
/// to renormalize the quaternion yourself where necessary.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html)
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

/// Creates a quaternion from the given euler angle x, y, z.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn from_euler(out: &mut Quat, x: f32, y: f32, z: f32) -> Quat {
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

    *out
}

/// Returns a string representation of a quatenion.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn string(a: &Quat) -> String {
    let a0 = ["quat(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = a[2].to_string(); 
    let a3 = [a[3].to_string(), ")".to_string()].join("");

    [a0, a1, a2, a3].join(", ")
}

/// Creates a new quat initialized with values from an existing quaternion.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn clone(a: &Quat) -> Quat { 
     vec4::clone(a)
}

/// Creates a new quat initialized with the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn from_values(x: f32, y: f32, z: f32, w: f32) -> Quat {
    vec4::from_values(x, y, z, w)
}

/// Copy the values from one quat to another.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn copy(out: &mut Quat, a: &Quat) -> Quat {
    vec4::copy(out, a)
}

/// Set the components of a quat to the given values.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn set(out: &mut Vec4, x: f32, y: f32, z: f32, w: f32) -> Quat {
    vec4::set(out, x, y, z, w)
}

/// Adds two quat's.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn add(out: &mut Quat, a: &Quat, b: &Quat) -> Quat {
    vec4::add(out, a, b)
}

/// Alias for link quat::multiply}
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn mul(out: &mut Quat, a: &Quat, b: &Quat) -> Quat {
    multiply(out, a, b)
}

/// Scales a quat by a scalar number.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn scale(out: &mut Quat, a: &Quat, b: f32) -> Quat {
    vec4::scale(out, a, b)
}

/// Calculates the dot product of two quats.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn dot(a: &Quat, b: &Quat) -> f32 { 
    vec4::dot(a, b)
}

/// Performs a linear interpolation between two quats.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn lerp(out: &mut Quat, a: &Quat, b: &Quat, t: f32) -> Quat {
    vec4::lerp(out, a, b, t)
}

/// Calculates the length of a quat.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn length(a: &Quat) -> f32 {
    vec4::length(a)
}

/// Alias for quat::length.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn len(a: &Quat) -> f32 {
    length(a)
}

/// Calculates the squared length of a quat.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn squared_length(a: &Quat) -> f32 {
    vec4::squared_length(a)
}

///Alias for quat::squaredLength
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn sqr_len(a: &Quat) -> f32 {
    squared_length(a)
}

/// Normalize a quat.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn normalize(out: &mut Quat, a: &Quat) -> Quat {
    vec4::normalize(out, a)
}

/// Returns whether or not the quaternions have exactly the same elements in the same position (when compared with ==).
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn exact_equals(a: &Quat, b: &Quat) -> bool {
    vec4::exact_equals(a, b)
}

/// Returns whether or not the quaternions have approximately the same elements in the same position.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn equals(a: &Quat, b: &Quat) -> bool {
    vec4::equals(a, b)
}

/// Sets a quaternion to represent the shortest rotation from one
/// vector to another.
//
/// Both vectors are assumed to be unit length.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn rotation_to(out: &mut Quat, a: &Vec3, b: &Vec3) -> Quat { 
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

    *out
}

/// Performs a spherical linear interpolation with two control points.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn sqlerp(out: &mut Quat, a: &Quat, b: &Quat, 
                              c: &Quat, d: &Quat, 
                              t: f32) -> Quat { 
    let temp1 = &mut create();
    let temp2 = &mut create();

    slerp(temp1, a, d, t);
    slerp(temp2, b, c, t);
    slerp(out, temp1, temp2, 2. * t * (1. - t))
}

/// Sets the specified quaternion with values corresponding to the given
/// axes. Each axis is a vec3 and is expected to be unit length and
/// perpendicular to all other specified axes.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-quat.html) 
pub fn set_axis(out: &mut Quat, view: &Vec3, 
                right: &Vec3, up: &Vec3) -> Quat {
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
   
    normalize(out, &from_mat3(&mut clone(out), matr))
}


#[cfg(test)]
mod tests { 
    use super::*;

    #[test]
    fn create_a_quat() { 
        let ident: Quat = [0., 0., 0., 1.];

        let out = create(); 

        assert_eq!(ident, out);
    }

    #[test]
    fn set_a_quat_to_identity() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let ident: Quat = [0., 0., 0., 1.];

        let result = identity(&mut out);

        assert_eq!(ident, out); 
        assert_eq!(result, out);
    }

    #[test]
    fn set_quat_axis_angle() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let axis: Vec3 = [1., 0., 0.];
        let rad = PI * 0.5; 

        let result = set_axis_angle(&mut out, &axis, rad);

        assert!(equals(&[0.707106, 0., 0., 0.707106], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn get_axis_angle_of_quat_no_rotation() {
        use super::super::common;

        let mut out: Quat = [0., 0., 0., 0.]; 
        let mut vec: Vec3 = [0., 0., 0.];

        set_axis_angle(&mut out, &[0., 1., 0.], 0_f32);
        let deg = get_axis_angle(&mut vec, &out); 
        
        assert_eq!([1., 0., 0.], vec);
        assert!(common::equals(deg % (PI * 2.0), 0_f32));
    }

    #[test]
    fn get_axis_angle_of_quat_about_x() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let mut vec: Vec3 = [0., 0., 0.];

        set_axis_angle(&mut out, &[1., 0., 0.],  0.7778);
        let deg = get_axis_angle(&mut vec, &out); 
       
        assert_eq!([1., 0., 0.], vec);
        assert_eq!(deg,  0.7778);
    }

    #[test]
    fn get_axis_angle_of_quat_about_y() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let mut vec: Vec3 = [0., 0., 0.];

        set_axis_angle(&mut out, &[0., 1., 0.],  0.879546);
        let deg = get_axis_angle(&mut vec, &out); 
       
        assert_eq!([0., 1., 0.], vec);
        assert_eq!(deg, 0.879546);
    }
    
    #[test]
    fn get_axis_angle_of_quat_about_z() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let mut vec: Vec3 = [0., 0., 0.];

        set_axis_angle(&mut out, &[0., 0., 1.], 0.123456);
        let deg = get_axis_angle(&mut vec, &out); 
       
        assert_eq!([0.0, 0.0, 1.0000055], vec);
        assert_eq!(deg, 0.12345532);
    }

    #[test]
    fn get_axis_angle_of_quat_slightly_weird() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let mut vec: Vec3 = [0., 0., 0.];

        set_axis_angle(&mut out, &[0.707106, 0., 0.707106], PI * 0.5);
        let deg = get_axis_angle(&mut vec, &out); 
       
        assert_eq!([0.707106, 0., 0.707106], vec);
        assert_eq!(deg, PI * 0.5);
    }

    #[test]
    fn get_angle_of_quat() { 
        use super::super::common;

        let quat_a = &mut [1., 2., 3., 4.];
        let mut quat_b = create();

        normalize(quat_a, &clone(quat_a));
        rotate_x(&mut quat_b, &quat_a, PI / 4_f32);
        let rad = get_angle(quat_a, &quat_b);

        assert!(common::equals(rad, PI / 4_f32 ));
    }

    #[test]
    fn multiply_two_quats() {     
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a = [1., 2., 3., 4.];
        let quat_b = [5., 6., 7., 8.];

        let result = multiply(&mut out, &quat_a, &quat_b);

        assert_eq!([24., 48., 48., -6.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn mul_two_quats() {     
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a = [1., 2., 3., 4.];
        let quat_b = [5., 6., 7., 8.];

        let result = mul(&mut out, &quat_a, &quat_b);

        assert_eq!([24., 48., 48., -6.], out);
        assert_eq!(result, out);
    }
    #[test]
    fn mul_is_equal_to_multiply() {
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a = [1., 2., 3., 4.];
        let quat_b = [5., 6., 7., 8.];

        let result_a = multiply(&mut out, &quat_a, &quat_b);
        let result_b = mul(&mut out, &quat_a, &quat_b);

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn rotate_quat_x() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let mut vec = vec3::create();
        let id: Quat = [0., 0., 0., 1.];
        let deg_90 = PI / 2_f32;

        let result = rotate_x(&mut out, &id, deg_90);
        vec3::transform_quat(&mut vec, &[0.,0.,-1.], &out);

        assert_eq!([0.70710677, 0.0, 0.0, 0.70710677], out);
        assert!(vec3::equals(&[0., 1., 0.], &vec));
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_quat_y() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let mut vec = vec3::create();
        let id: Quat = [0., 0., 0., 1.];
        let deg_90 = PI / 2_f32;

        let result = rotate_y(&mut out, &id, deg_90);
        vec3::transform_quat(&mut vec, &[0.,0.,-1.], &out);

        assert_eq!([0.0, 0.70710677, 0., 0.70710677], out);
        assert!(vec3::equals(&[-1., 0., 0.], &vec));
        assert_eq!(result, out);
    }

    #[test]
    fn rotate_quat_z() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let mut vec = vec3::create();
        let id: Quat = [0., 0., 0., 1.];
        let deg_90 = PI / 2_f32;

        let result = rotate_z(&mut out, &id, deg_90);
        vec3::transform_quat(&mut vec, &[0.,1.,0.], &out);

        assert_eq!([0., 0., 0.70710677, 0.70710677], out);
        assert!(vec3::equals(&[-1., 0., 0.], &vec));
        assert_eq!(result, out);
    }

    #[test]
    fn calculate_quat_w() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a: Quat = [0.70710677, 0., 0., 0.70710677];

        let result = calculate_w(&mut out, &quat_a);

        assert_eq!([0.70710677, 0., 0., 0.70710677], quat_a);
        assert_eq!(result, out);
    }

    #[test]
    fn exp_quat() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a: Quat = [0.70710677, 0., 0., 0.70710677];

        let result = exp(&mut out, &quat_a);

        assert_eq!([0.70710677, 0., 0., 0.70710677], quat_a);
        assert_eq!(result, out);
    }

    #[test]
    fn exp_quat_do_nothing() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a: Quat = [0., 0., 0., 1.];

        let result = exp(&mut out, &quat_a);

        assert_eq!([0., 0., 0., 1.], quat_a);
        assert_eq!(result, out);
    }
   
    #[test]
    fn ln_quat() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a: Quat = [0.70710677, 0., 0., 0.70710677];

        let result = ln(&mut out, &quat_a);

        assert_eq!([0.70710677, 0., 0., 0.70710677], quat_a);
        assert_eq!(result, out);
    }

    #[test]
    fn ln_quat_do_nothing() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a: Quat = [0., 0., 0., 1.];

        let result = ln(&mut out, &quat_a);

        assert_eq!([0., 0., 0., 1.], quat_a);
        assert_eq!(result, out);
    }
    
    #[test]
    fn pow_quat() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a: Quat = [0.70710677, 0., 0., 0.70710677];

        let result = pow(&mut out, &quat_a, 2_f32);

        assert_eq!([0.70710677, 0., 0., 0.70710677], quat_a);
        assert_eq!(result, out);
    }

    #[test]
    fn slerp_quat_normal() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a = [0., 0., 0., 1.];
        let quat_b = [0., 1., 0., 0.];

        let result = slerp(&mut out, &quat_a, &quat_b, 0.5);

        assert!(equals(&[0., 0.707106, 0., 0.707106], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn slerp_quat_a_equals_b() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a = [0., 0., 0., 1.];
        let quat_b = [0., 0., 0., 1.];

        let result = slerp(&mut out, &quat_a, &quat_b, 0.5);

        assert!(equals(&[0., 0., 0., 1.], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn slerp_quat_theta_is_180_degs() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let mut quat_a = [1., 2., 3., 4.];
        rotate_x(&mut quat_a, &[1., 0., 0., 0.], PI); // 180 deg

        let result = slerp(&mut out, &[1., 0., 0., 0.], &quat_a, 1.);
        

        // umm results may differ
        //assert!(equals(&[0., 0., 0., -1.], &out));
        assert!(equals(&[0., 0., 0., 1.], &out));
        assert_eq!(result, out);
    }


    #[test]
    fn slerp_quat_a_is_equal_to_neg_b() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let quat_a = [1., 0., 0., 0.];
        let quat_b = [-1., 0., 0., 0.];

        let result = slerp(&mut out, &quat_a, &quat_b, 0.5);

        assert!(equals(&[1., 0., 0., 0.], &out));
        assert_eq!(result, out);
    }
    
    #[test]
    fn random_quat() { 
        let mut out: Quat = [0., 0., 0., 0.]; 
        let result = random(&mut out);
        
        assert!(out[0] >= -1_f32 && out[0] <= 1_f32);
        assert!(out[1] >= -1_f32 && out[1] <= 1_f32);
        assert!(out[2] >= -1_f32 && out[2] <= 1_f32);
        assert!(out[3] >= -1_f32 && out[3] <= 1_f32);
        assert_eq!(result, out);
    }

    #[test]
    fn invert_quat() {
        let mut out: Quat = [0., 0., 0., 0.];
        let quat_a: Quat = [1., 2., 3., 4.];

        let result = invert(&mut out, &quat_a);

        assert!(equals(&[-0.033333, -0.066666, -0.1, 0.133333], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn invert_singular_quat() { 
        let mut out: Quat = [0., 0., 0., 0.];
        let quat_a: Quat = [0., 0., 0., 0.];

        let result = invert(&mut out, &quat_a);

        assert_eq!([0., 0., 0., 0.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn conjugate_quat() { 
        let mut out: Quat = [0., 0., 0., 0.];
        let quat_a: Quat = [1., 2., 3., 4.];

        let result = conjugate(&mut out, &quat_a);

        assert_eq!([-1., -2., -3., 4.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn quat_from_mat3_legacy() { 
        let mut out: Quat = [0., 0., 0., 0.];
        let mat_r: Mat3 =[ 1., 0.,  0.,
                           0., 0., -1.,
                           0., 1.,  0.];

        let result = from_mat3(&mut out, &mat_r);

        assert!(equals(&[-0.707106, 0., 0., 0.707106], &out));
        assert_eq!(result, out);
    }

    #[test]
    fn quat_from_mat3_looking_backwards() { 
        use super::super::mat4;

        let mut out: Quat = [0., 0., 0., 0.];
        let mat_r = &mut mat3::create();
        mat3::transpose(mat_r, &mat3::invert(&mut mat3::clone(mat_r), 
                                                 &mat3::from_mat4(&mut mat3::clone(mat_r), 
                                                                      &mat4::look_at(&mut mat4::create(), 
                                                                                     &[0., 0., 0.], 
                                                                                     &[0., 0., 1.], 
                                                                                     &[0., 1., 0.]))).unwrap()
                                                );

        let result = from_mat3(&mut out, &mat_r);
        
        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[3., 2., -1.], &normalize(&mut create(), &out)),
                &vec3::transform_mat3(&mut vec3::create(), &[3., 2., -1.], mat_r)
            )
        );
        assert_eq!(result, out);
    }

    #[test]
    fn quat_from_mat3_looking_left_and_upside_down() { 
        use super::super::mat4;

        let mut out: Quat = [0., 0., 0., 0.];
        let mat_r = &mut mat3::create();
        mat3::transpose(mat_r, &mat3::invert(&mut mat3::clone(mat_r), 
                                                 &mat3::from_mat4(&mut mat3::clone(mat_r), 
                                                                      &mat4::look_at(&mut mat4::create(), 
                                                                                     &[0., 0., 0.],
                                                                                     &[-1., 0., 0.],
                                                                                     &[0., -1., 0.]))).unwrap()
                                                );

        let result = from_mat3(&mut out, &mat_r);
        
        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[3., 2., -1.], &normalize(&mut create(), &out)),
                &vec3::transform_mat3(&mut vec3::create(), &[3., 2., -1.], mat_r)
            )
        );
        assert_eq!(result, out);
    }
    
    #[test]
    fn quat_from_mat3_looking_upside_down() { 
        use super::super::mat4;

        let mut out: Quat = [0., 0., 0., 0.];
        let mat_r = &mut mat3::create();
        mat3::transpose(mat_r, &mat3::invert(&mut mat3::clone(mat_r), 
                                                 &mat3::from_mat4(&mut mat3::clone(mat_r), 
                                                                      &mat4::look_at(&mut mat4::create(), 
                                                                                     &[0., 0., 0.],
                                                                                     &[0., 0., -1.],
                                                                                     &[0., -1., 0.]))).unwrap()
                                                );

        let result = from_mat3(&mut out, &mat_r);
        
        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[3., 2., -1.], &normalize(&mut create(), &out)),
                &vec3::transform_mat3(&mut vec3::create(), &[3., 2., -1.], mat_r)
            )
        );
        assert_eq!(result, out);
    }
    
    #[test]
    fn quat_from_euler_legacy() {
        let mut out: Quat = [0., 0., 0., 0.];

        let result = from_euler(&mut out, -90., 0., 0.);

        assert!(equals(&out, &[-0.707106, 0., 0., 0.707106]));
        assert_eq!(result, out);
    }

    #[test]
    fn quat_from_euler_trace_greater_than_zero() {
        let mut out: Quat = [0., 0., 0., 0.];

        let result = from_euler(&mut out, -90., 0., 0.);

        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[0.,1.,0.], &out),
                &[0.,0.,-1.]
            )
        );
        assert_eq!(result, out);
    }
    
    #[test]
    fn get_quat_string() { 
        let quat_a: Quat = [1., 2., 3., 4.];

        let quat_string = string(&quat_a);

        assert_eq!("quat(1, 2, 3, 4)".to_string(), quat_string);
    }

    #[test]
    fn clone_a_quat() { 
        let quat_a: Quat = [1., 2., 3., 4.];

        let out = clone(&quat_a);

        assert_eq!([1., 2., 3., 4.], out);
    }

    #[test] 
    fn create_quat_from_values() { 
        let out = from_values(1., 2., 3., 4.);

        assert_eq!([1., 2., 3., 4.], out);
    }

    #[test]
    fn copy_values_from_a_quat_it_another() { 
        let mut out: Quat = [0., 0., 0., 0.];
        let quat_a: Quat = [1., 2., 3., 4.];

        let result = copy(&mut out, &quat_a);

        assert_eq!([1., 2., 3., 4.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn set_quat_with_values() { 
        let mut out: Quat = [0., 0., 0., 0.];

        let result = set(&mut out, 1., 2., 3., 4.);

        assert_eq!([1., 2., 3., 4.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn add_two_quats() { 
        let mut out: Quat = [0., 0., 0., 0.];
        let quat_a: Quat = [1., 2., 3., 4.];
        let quat_b: Quat = [5., 6., 7., 8.];

        let result = add(&mut out, &quat_a, &quat_b);

        assert_eq!([6., 8., 10., 12.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn scale_quat() { 
        let mut out: Quat = [0., 0., 0., 0.];
        let quat_a: Quat = [1., 2., 3., 4.];

        let result = scale(&mut out, &quat_a, 2_f32);

        assert_eq!([2., 4., 6., 8.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn dot_quat() { 
        let mut out: Quat = [0., 0., 0., 0.];
        let quat_a: Quat = [1., 2., 3., 4.];

        let result = scale(&mut out, &quat_a, 2_f32);

        assert_eq!([2., 4., 6., 8.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn lerp_quat() { 
        let mut out: Quat = [0., 0., 0., 0.];
        let quat_a: Quat = [1., 2., 3., 4.];
        let quat_b: Quat = [5., 6., 7., 8.];

        let result = lerp(&mut out, &quat_a, &quat_b, 0.5);

        assert_eq!([3., 4., 5., 6.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn length_of_quat() {
        let quat_a: Quat = [1., 2., 3., 4.];

        let out = length(&quat_a);

        // they get 5.477225
        assert_eq!(5.477226, out);
    }

    #[test]
    fn len_of_quat() { 
        let quat_a: Quat = [1., 2., 3., 4.];

        let out = len(&quat_a);

        // they get 5.477225
        assert_eq!(5.477226, out);
    }

    #[test]
    fn length_is_equal_to_len() {
        let quat_a: Quat = [1., 2., 3., 4.];

        let out_a = length(&quat_a);
        let out_b = len(&quat_a);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn squared_length_of_quat() { 
        let quat_a: Quat = [1., 2., 3., 4.];

        let out = squared_length(&quat_a);

        // they get 5.477225
        assert_eq!(30_f32, out);
    }

    #[test]
    fn sqr_len_of_quat() { 
        let quat_a: Quat = [1., 2., 3., 4.];

        let out = sqr_len(&quat_a);

        // they get 5.477225
        assert_eq!(30_f32, out);
    }

    #[test]
    fn squared_length_is_equal_to_sqr_len() { 
        let quat_a: Quat = [1., 2., 3., 4.];

        let out_a = squared_length(&quat_a);
        let out_b = sqr_len(&quat_a);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn normalize_quat() { 
        let mut out: Quat = [0., 0., 0., 0.];
        let quat_a: Quat = [5., 0., 0., 0.];

        let result = normalize(&mut out, &quat_a);

        assert_eq!([1., 0., 0., 0.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn quat_are_exact_equal() { 
        let quat_a: Quat = [0., 1., 2., 3.];
        let quat_b: Quat = [0., 1., 2., 3.];

        let r0 = exact_equals(&quat_a, &quat_b);

        assert!(r0);  
    }

    #[test]
    fn quat_are_not_exact_equal() { 
        let quat_a: Quat = [0., 1., 2., 3.];
        let quat_b: Quat = [1., 2., 3., 4.];

        let r0 = exact_equals(&quat_a, &quat_b);

        assert!(!r0); 
    }

    #[test]
    fn quat_are_equal() { 
        let quat_a: Quat = [0., 1., 2., 3.];
        let quat_b: Quat = [0., 1., 2., 3.];

        let r0 = equals(&quat_a, &quat_b);

        assert!(r0);  
    }

    #[test]
    fn quat_are_equal_enough() { 
        let quat_a: Quat = [0., 1., 2., 3.];
        let quat_b: Quat = [1_f32*10_f32.powi(-16), 1., 2., 3.];

        let r0 = equals(&quat_a, &quat_b);

        assert!(r0);  
    }

    #[test]
    fn quat_are_not_equal() { 
        let quat_a: Quat = [0., 1., 2., 3.];
        let quat_b: Quat = [1., 2., 3., 4.];

        let r0 = equals(&quat_a, &quat_b);

        assert!(!r0);  
    }

    #[test]
    fn rotation_to_right_angle() { 
        let mut out: Quat = [0., 0., 0., 0.];

        let result = rotation_to(&mut out, &[0., 1., 0.], &[1., 0., 0.]);

        assert!(equals(&[0., 0., -0.707106, 0.707106], &out));  
        assert_eq!(result, out);
    }

    #[test]
    fn rotation_to_when_vectors_are_parallel() { 
        let mut out: Quat = [0., 0., 0., 0.];

        let result = rotation_to(&mut out, &[0., 1., 0.], &[0., 1., 0.]);

        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[0., 1., 0.], &out),
                &[0., 1., 0.]
            )
        );
        assert_eq!(result, out);
    }

    #[test]
    fn rotation_to_when_vectors_are_opposed_x() { 
        let mut out: Quat = [0., 0., 0., 0.];

        let result = rotation_to(&mut out, &[1., 0., 0.], &[-1., 0., 0.]);

        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[1., 0., 0.], &out),
                &[-1., 0., 0.]
            )
        );
        assert_eq!(result, out);
    }

    #[test]
    fn rotation_to_when_vectors_are_opposed_y() { 
        let mut out: Quat = [0., 0., 0., 0.];

        let result = rotation_to(&mut out, &[0., 1., 0.], &[0., -1., 0.]);

        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[0., 1., 0.], &out),
                &[0., -1., 0.]
            )
        );
        assert_eq!(result, out);
    }

    #[test]
    fn rotation_to_when_vectors_are_opposed_z() { 
        let mut out: Quat = [0., 0., 0., 0.];

        let result = rotation_to(&mut out, &[0., 0., 1.], &[0., 0., -1.]);

        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[0., 0., 1.], &out),
                &[0., 0., -1.]
            )
        );
        assert_eq!(result, out);
    }
   
    #[test]
    fn set_axis_looking_left() {
        let mut out: Quat = [0., 0., 0., 0.];
        let view  = [-1., 0., 0.];
        let up    = [ 0., 1., 0.];
        let right = [ 0., 0.,-1.];
        
        let result = set_axis(&mut out, &view, &right, &up);

        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[0., 0., -1.], &result),
                &[1., 0., 0.]
            )
        );
        assert!(
            vec3::equals(
                &vec3::transform_quat(&mut vec3::create(), &[1., 0., 0.], &result),
                &[0., 0., 1.]
            )
        );
    }

    #[test]
    fn set_axis_given_opengl_defaults() {
        let mut out: Quat = [0., 0., 0., 0.];
        let view  = [0., 0., -1.];
        let up    = [0., 1.,  0.];
        let right = [1., 0.,  0.];
        
        let result = set_axis(&mut out, &view, &right, &up);

        assert_eq!([0., 0., 0., 1.], out);
        assert_eq!(result, out);
    }

    #[test]
    fn set_axis_legacy_example() {
        let mut out: Quat = [0., 0., 0., 0.];
        let view  = [0., 0., -1.];
        let up    = [0., 1.,  0.];
        let right = [1., 0.,  0.];
        
        let result = set_axis(&mut out, &view, &right, &up);

        assert_eq!([0., 0., 0., 1.], out);
        assert_eq!(result, out);
    }
}