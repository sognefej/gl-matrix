use super::common::{Mat2, Vec2, hypot, EPSILON};

pub fn create() -> Mat2 {
    let mut out: Mat2 = [0_f32; 4];    
    
    out[0] = 1_f32; 
    out[3] = 1_f32; 

    out 
}

pub fn clone(a: &Mat2) -> Mat2 { 
    let mut out: Mat2 = [0_f32; 4]; 
    
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];

    out
} 

pub fn copy(out: &mut Mat2, a: &Mat2) -> Mat2 {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];

    *out
}

pub fn identity(out: &mut Mat2) -> Mat2 {
    out[0] = 1_f32;
    out[1] = 0_f32;
    out[2] = 0_f32;
    out[3] = 1_f32;

    *out
}

pub fn from_values(m00: f32, m01: f32, m10: f32, m11: f32) -> Mat2 {
    let mut out: Mat2 = [0_f32; 4]; 
    out[0] = m00;
    out[1] = m01;
    out[2] = m10;
    out[3] = m11;

    out
}

pub fn set(out: &mut Mat2, m00: f32, m01: f32, m10: f32, m11: f32) -> Mat2 {
    out[0] = m00;
    out[1] = m01;
    out[2] = m10;
    out[3] = m11;

    *out
}

pub fn transpose(out: &mut Mat2, a: &Mat2) {
    // If we are transposing ourselves we can skip a few steps but have to cache
    // some values
    if out.eq(&a) {
        let a1 = a[1];
        out[1] = a[2];
        out[2] = a1;
    } else {
        out[0] = a[0];
        out[1] = a[2];
        out[2] = a[1];
        out[3] = a[3];
    }
}

pub fn invert(mut out: Mat2, a: &Mat2) -> Result<Mat2, String> {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2]; 
    let a3 = a[3];
    // Calculate the determinant
    let det = a0 * a3 - a2 * a1;
    
    // Make sure matrix is not singular
    if det == 0_f32 { 
        return Err("Matrix is singular".to_string());
    }
     
    let det = 1_f32 / det;
    
    out[0] =  a3 * det;
    out[1] = -a1 * det;
    out[2] = -a2 * det;
    out[3] =  a0 * det;
    
    Ok(out)
}

pub fn adjoint(out: &mut Mat2, a: &Mat2) {
    // Caching this value is nessecary if out == a
    let a0 = a[0];

    out[0] =  a[3];
    out[1] = -a[1];
    out[2] = -a[2];
    out[3] =  a0;
}

pub fn determinant(a: &Mat2) -> f32 {
    a[0] * a[3] - a[2] * a[1]
}

pub fn multiply(out: &mut Mat2, a: &Mat2, b: &Mat2) {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];
    let a3 = a[3];

    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3];

    out[0] = a0 * b0 + a2 * b1;
    out[1] = a1 * b0 + a3 * b1;
    out[2] = a0 * b2 + a2 * b3;
    out[3] = a1 * b2 + a3 * b3;
}

pub fn rotate(out: &mut Mat2, a: &Mat2, rad: f32) {
    let a0 = a[0];
    let a1 = a[1]; 
    let a2 = a[2];
    let a3 = a[3];

    let s = f32::sin(rad);
    let c = f32::cos(rad);

    out[0] = a0 *  c + a2 * s;
    out[1] = a1 *  c + a3 * s;
    out[2] = a0 * -s + a2 * c;
    out[3] = a1 * -s + a3 * c;
}

pub fn scale(out: &mut Mat2, a: &Mat2, v: &Vec2) {
    let a0 = a[0]; 
    let a1 = a[1];
    let a2 = a[2]; 
    let a3 = a[3];
    
    let v0 = v[0]; 
    let v1 = v[1];

    out[0] = a0 * v0;
    out[1] = a1 * v0;
    out[2] = a2 * v1;
    out[3] = a3 * v1;
}

pub fn from_rotation(out: &mut Mat2, rad: f32) {
    let s = f32::sin(rad);
    let c = f32::cos(rad);

    out[0] = c;
    out[1] = s;
    out[2] = -s;
    out[3] = c;
}

pub fn from_scaling(out: &mut Mat2, v: &Vec2) {
    out[0] = v[0];
    out[1] = 0_f32;
    out[2] = 0_f32;
    out[3] = v[1];
}

pub fn string(a: &Mat2) -> String {
    let a0 = ["mat2(".to_string(), a[0].to_string()].join("");
    let a1 = a[1].to_string(); 
    let a2 = a[2].to_string(); 
    let a3 = [a[3].to_string(), ")".to_string()].join("");

    [a0, a1, a2, a3].join(", ")
}

pub fn frob(a: &Mat2) -> f32 {
    hypot(a)
}

pub fn ldu(l: &mut Mat2, _d: &mut Mat2, u: &mut Mat2, a: &Mat2) {
    l[2] = a[2] / a[0];
    u[0] = a[0];
    u[1] = a[1];
    u[3] = a[3] - l[2] * u[1];
}

pub fn add(out: &mut Mat2, a: &Mat2, b: &Mat2) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
    out[3] = a[3] + b[3];  
}

pub fn subtract(out: &mut Mat2, a: &Mat2, b: &Mat2) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
    out[3] = a[3] - b[3];
}

pub fn exact_equals(a: &Mat2, b: &Mat2) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
}

pub fn equals(a: &Mat2, b: &Mat2) -> bool {
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

pub fn multiply_scalar(out: &mut Mat2, a: &Mat2, b: f32) {
    out[0] = a[0] * b;
    out[1] = a[1] * b;
    out[2] = a[2] * b;
    out[3] = a[3] * b;
}

pub fn multiply_scalar_and_add(out: &mut Mat2, a: &Mat2, b: &Mat2, scale: f32) {
    out[0] = a[0] + (b[0] * scale);
    out[1] = a[1] + (b[1] * scale);
    out[2] = a[2] + (b[2] * scale);
    out[3] = a[3] + (b[3] * scale);
}

pub fn mul(out: &mut Mat2, a: &Mat2, b: &Mat2) {
    multiply(out, a, b);
}

pub fn sub(out: &mut Mat2, a: &Mat2, b: &Mat2) {
    subtract(out, a, b);
}


#[cfg(test)] 
mod tests {
    use super::*; 

    #[test] 
    fn create_a_mat2() {
        let ident: Mat2 = [1., 0., 0., 1.];
 
        let out = create();
        
        assert_eq!(ident, out);
    }

    #[test] 
    fn clone_a_mat2() {
        let mat_a: Mat2 = [1., 2., 3., 4.];
  
        let out = clone(&mat_a);
       
        assert_eq!(mat_a, out);
    }

    #[test] 
    fn copy_values_from_a_mat2_to_another() {
        let mut out =  [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
   
        copy(&mut out, &mat_a);
      
        assert_eq!(mat_a, out);
    }

    #[test]
    fn copy_result_equal_to_out() { 
        let mut out =  [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
   
        let result = copy(&mut out, &mat_a);
      
        assert_eq!(result, out);
    }

    #[test] 
    fn set_a_mat2_to_identity() {  
        let mut out: Mat2 = [0., 0., 0., 0.];  
        let ident = [1., 0., 0., 1.];
    
        identity(&mut out);
     
        assert_eq!(out, ident);
    }

    #[test] 
    fn identity_reslut_equal_to_out() {  
        let mut out: Mat2 = [0., 0., 0., 0.];  
    
        let result = identity(&mut out);
     
        assert_eq!(result, out);
    }
    
    #[test]
    fn create_mat2_from_values() { 
        let out = from_values(1., 2., 3. ,4.); 
    
        assert_eq!([1., 2., 3., 4.], out); 
    }

    #[test]
    fn set_mat2_with_values() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
     
        set(&mut out, 1., 2., 3., 4.);

        assert_eq!([1., 2., 3., 4.], out); 
    }
    
    #[test]
    fn set_reslut_equal_to_out() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
     
        let result = set(&mut out, 1., 2., 3., 4.);

        assert_eq!(result, out);
    }
    
    #[test] 
    fn transpose_same_mat2() { 
        let mut mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_a_copy: Mat2 = [1., 2., 3., 4.];
      
        transpose(&mut mat_a, &mat_a_copy);

        assert_eq!([1., 3., 2., 4.], mat_a);
    }

    #[test] 
    fn transpose_different_mat2() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        
        transpose(&mut out, &mat_a); 
       
        assert_eq!([1., 3., 2., 4.], out);
    }
    
    #[test] 
    fn invert_mat2() {  
        let out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];

        let out = invert(out, &mat_a); 
        let out = match out { 
            Ok(out) => out,
            Err(_error) => panic!("This should have worked!")
        };

        assert_eq!([-2., 1., 1.5, -0.5], out);
    } 

    #[test] 
    #[should_panic(expected = "Matrix is singular")]
    fn invert_singular_mat2() {  
        let out: Mat2 =  [0., 0., 0., 0.];
        let mat_a: Mat2 = [-1., 3./2., 2./3., -1.]; 

        let out = invert(out, &mat_a); 
        let _out = match out { 
            Ok(out) => out,
            Err(error) => panic!(error)
        };
    } 
    
    #[test] 
    fn adjugate_mat2() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
       
        adjoint(&mut out, &mat_a); 
        
        assert_eq!([4., -2., -3., 1.], out);         
    }

    #[test] 
    fn adjugate_mat2_same() { 
        let mut mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_a_copy: Mat2 = [1., 2., 3., 4.];

        adjoint(&mut mat_a, &mat_a_copy); 
        
        assert_eq!([4., -2., -3., 1.], mat_a);         
    }

    #[test]
    fn get_mat2_determinant() { 
        let mat_a: Mat2 = [1., 2., 3., 4.];

        let det: f32 = determinant(&mat_a);

        assert_eq!(-2_f32, det); 
    }

    #[test]
    fn multiply_two_mat2s() {  
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_b: Mat2 = [5., 6., 7., 8.];

        multiply(&mut out, &mat_a, &mat_b);

        assert_eq!([23., 34., 31., 46.], out); 
    }

    #[test]
    fn mul_two_mat2s() {  
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_b: Mat2 = [5., 6., 7., 8.];

        mul(&mut out, &mat_a, &mat_b);

        assert_eq!([23., 34., 31., 46.], out); 
    }

    #[test]
    fn mul_is_equal_to_multiply() {  
        let mut out_a: Mat2 = [0., 0., 0., 0.];
        let mut out_b: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_b: Mat2 = [5., 6., 7., 8.];

        multiply(&mut out_a, &mat_a, &mat_b);
        mul(&mut out_b, &mat_a, &mat_b);

        assert_eq!(out_a, out_b);
    }
    
    #[test]
    fn rotate_a_mat2() { 
        use super::super::common::{PI};
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        
        rotate(&mut out, &mat_a, PI * 0.5);
        
        assert!(equals(&[3., 4., -1., -2.], &out));
    }
    
    #[test]
    fn scale_mat2() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let vec_a: Vec2 = [2., 3.];

        scale(&mut out, &mat_a, &vec_a);

        assert_eq!([2., 4., 9., 12.], out)
    }

    #[test]
    fn get_mat2_string() { 
        let mat_a: Mat2 = [1., 2., 3., 4.];
        
        let str_a = string(&mat_a);

        assert_eq!("mat2(1, 2, 3, 4)".to_string(), str_a);
    }

    #[test]
    fn calc_frob_norm_of_mat2() {
        let mat_a: Mat2 = [1., 2., 3., 4.];

        let frob_a = frob(&mat_a);

        assert_eq!((1_f32.powi(2) + 2_f32.powi(2) + 3_f32.powi(2) + 4_f32.powi(2)).sqrt(), frob_a);
    }

    #[test]
    fn get_ldu_mat2() {
        let mut l: Mat2 = create();
        let mut d: Mat2 = create();
        let mut u: Mat2 = create();
        
        let mut l_result: Mat2 = create(); 
        l_result[2] = 1.5;
        let d_result: Mat2 = create();
        let mut u_result: Mat2 = create();
        u_result[0] = 4.; 
        u_result[1] = 3.; 
        u_result[3] = -1.5;

        ldu(&mut l, &mut d, &mut u, &[4.,3.,6.,3.]);
        assert_eq!(l_result, l);
        assert_eq!(d_result, d);
        assert_eq!(u_result, u);
    }

    #[test]
    fn add_two_mat2s() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_b: Mat2 = [5., 6., 7., 8.];

        add(&mut out, &mat_a, &mat_b);

        assert_eq!([6., 8., 10., 12.], out);
    }

    #[test]
    fn subtract_two_mat2s() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_b: Mat2 = [5., 6., 7., 8.];

        subtract(&mut out, &mat_a, &mat_b);

        assert_eq!([-4., -4., -4., -4.], out);
    }

    #[test]
    fn sub_two_mat2s() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_b: Mat2 = [5., 6., 7., 8.];

        sub(&mut out, &mat_a, &mat_b);

        assert_eq!([-4., -4., -4., -4.], out);
    }

    #[test]
    fn sub_is_equal_to_subtract() { 
        let mut out_a: Mat2 = [0., 0., 0., 0.];
        let mut out_b: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_b: Mat2 = [5., 6., 7., 8.];

        sub(&mut out_a, &mat_a, &mat_b);
        subtract(&mut out_b, &mat_a, &mat_b);

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn mat2s_are_exact_equal() { 
        let mat_a: Mat2 = [0., 1., 2., 3.];
        let mat_b: Mat2 = [0., 1., 2., 3.];

        let r0 = exact_equals(&mat_a, &mat_b);

        assert!(r0);  
    }

    #[test]
    fn mat2s_are_not_exact_equal() { 
        let mat_a: Mat2 = [0., 1., 2., 3.];
        let mat_b: Mat2 = [1., 2., 3., 4.];

        let r0 = exact_equals(&mat_a, &mat_b);

        assert!(!r0); 
    }

    #[test]
    fn mat2s_are_equal() { 
        let mat_a: Mat2 = [0., 1., 2., 3.];
        let mat_b: Mat2 = [0., 1., 2., 3.];

        let r0 = equals(&mat_a, &mat_b);

        assert!(r0);  
    }

    #[test]
    fn mat2s_are_equal_enough() { 
        let mat_a: Mat2 = [0., 1., 2., 3.];
        let mat_b: Mat2 = [1_f32*10_f32.powi(-16), 1., 2., 3.];

        let r0 = equals(&mat_a, &mat_b);

        assert!(r0);  
    }

    #[test]
    fn mat2s_are_not_equal() { 
        let mat_a: Mat2 = [0., 1., 2., 3.];
        let mat_b: Mat2 = [1., 2., 3., 4.];

        let r0 = equals(&mat_a, &mat_b);

        assert!(!r0);  
    }

    #[test]
    fn multiply_mat2_by_scalar() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        
        multiply_scalar(&mut out, &mat_a, 2.);

        assert_eq!([2., 4., 6., 8.], out);
    }

    #[test]
    fn multiply_mat2_by_scalar_and_add() { 
        let mut out: Mat2 = [0., 0., 0., 0.];
        let mat_a: Mat2 = [1., 2., 3., 4.];
        let mat_b: Mat2 = [5., 6., 7., 8.];

        multiply_scalar_and_add(&mut out, &mat_a, &mat_b, 0.5);

        assert_eq!([3.5, 5., 6.5, 8.], out);
    } 
}
