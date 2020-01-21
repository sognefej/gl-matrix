// Public types 
pub type Mat2 = [f32; 4];
pub type Mat2d = [f32; 6];
pub type Mat3 = [f32; 9];
pub type Mat4 = [f32; 16];
pub type Quat = [f32; 4];
pub type Quat2 = [f32; 8];
pub type Vec2 = [f32; 2];
pub type Vec3 = [f32; 3];
pub type Vec4 = [f32; 4];

// Configuration Constants 
pub static PI: f32 = core::f32::consts::PI; 
pub static EPSILON: f32 = 0.000001;
pub static INFINITY: f32 = 1.0_f32 / 0.0_f32;
pub static NEG_INFINITY: f32 = -1.0_f32 / 0.0_f32;

// We don't have a set_matrix_array_type
// we only support f32

static DEGREE: f32 = PI / 180.0_f32;

pub fn to_radian(a: f32) -> f32{ 
    return a * DEGREE
}

pub fn equals(a: f32, b: f32) -> bool {
  return (a - b).abs() <= EPSILON * 1.0_f32.max(a.abs().max(b.abs()));
}

pub fn hypot(arguments: &[f32]) -> f32 { 
    let mut y: f32 = 0_f32;
    let len = arguments.len();
    
    for i in 0..len { 
        y += arguments[i].powi(2); 
    }

    y.sqrt()
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

        assert_eq!(x.hypot(y), hypot(&vec2.to_vec()));
    }
}
