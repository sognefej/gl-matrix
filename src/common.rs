//! Common utilities

use rand::Rng;

// Public types 
/// 2x2 Matrix
pub type Mat2 = [f32; 4];
/// 2x3 Matrix
pub type Mat2d = [f32; 6];
/// 3x3 Matrix
pub type Mat3 = [f32; 9];
/// 4x4 Matrix
pub type Mat4 = [f32; 16];
/// Quaternion
pub type Quat = [f32; 4];
/// Dual Quaternion
pub type Quat2 = [f32; 8];
/// 2 Dimensional Vector
pub type Vec2 = [f32; 2];
/// 3 Dimensional Vector
pub type Vec3 = [f32; 3];
/// 4 Dimensional Vector
pub type Vec4 = [f32; 4];

// Configuration Constants 
/// Archimedes' constant (π).
/// 
/// [core::f32::consts::PI](https://doc.rust-lang.org/core/f32/consts/constant.PI.html)
pub static PI: f32 = core::f32::consts::PI; 
/// This is the difference between 1.0 and the next largest representable number.
/// 
/// our value is ```0.000001```.
pub static EPSILON: f32 = 0.000001;
/// Infinity (∞).
/// 
/// [core::f32::INFINITY](https://doc.rust-lang.org/core/f32/constant.INFINITY.html)
pub static INFINITY: f32 = 1.0_f32 / 0.0_f32;
/// Negative infinity (-∞).
/// 
/// [core::f32::NEG_INFINITY](https://doc.rust-lang.org/core/f32/constant.NEG_INFINITY.html)
pub static NEG_INFINITY: f32 = -1.0_f32 / 0.0_f32;

// We don't have a set_matrix_array_type
// we only support f32

static DEGREE: f32 = PI / 180.0_f32;

/// Convert Degree to Radian.
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-glMatrix.html)
pub fn to_radian(a: f32) -> f32{ 
    return a * DEGREE
}

/// Tests whether or not the arguments have approximately the same value, within an absolute
/// or relative tolerance of common::EPSILON (an absolute tolerance is used for values less
/// than or equal to 1.0, and a relative tolerance is used for larger values).
/// 
/// [glMatrix Documentation](http://glmatrix.net/docs/module-glMatrix.html)
pub fn equals(a: f32, b: f32) -> bool {
  return (a - b).abs() <= EPSILON * 1.0_f32.max(a.abs().max(b.abs()));
}

/// The hypot function returns the square root of the sum of squares of the array elements.
/// 
/// [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/hypot) 
pub fn hypot(arguments: &[f32]) -> f32 { 
    let mut y: f32 = 0_f32;
    let len = arguments.len();
    
    for i in 0..len { 
        y += arguments[i].powi(2); 
    }

    y.sqrt()
}

/// A random f32 between 0-1.
pub fn random_f32() -> f32 { 
    let mut rng = rand::thread_rng();
    // f64 gives a uniform distriution over 0-1
    // f32 gives random numbers over the entire f32 space
    // however we want a f32 between 0-1
    let r_f32: f64 = rng.gen();
    // convert the f64 to f32 so we can use it 
    let r_f32 = r_f32 as f32;

    r_f32
}


#[cfg(test)] 
mod tests {
    use super::*; 

    #[test] 
    fn degrees_to_radian() { 
        let deg = 80_f32; 
        let rad = to_radian(deg); 
       
        assert_eq!(1.3962634, rad);
    }

    #[test] 
    fn epsilon_equals_false() { 
        let a = 1.00001_f32; 
        let b = 1_f32; 
        
        assert!(!equals(a, b));
    } 

    #[test] 
    fn epsilon_equals_true() { 
        let a = 1.000001_f32; 
        let b = 1_f32; 
     
        assert!(equals(a, b)); 
    } 
    
    #[test] 
    fn get_hypot() { 
        let x = 2.0_f32;
        let y = 3.0_f32;
        let vec2: [f32; 2] = [x, y]; 

        assert_eq!(x.hypot(y), hypot(&vec2));
    }

    #[test]
    fn random_f32_between_zero_and_one() { 
        let r = random_f32(); 

        assert!(r >= 0_f32 && r <= 1_f32);
    }
}
