//! A Rust implementation of [glMatrix](http://glmatrix.net/)
//!
//! gl_matrix provides utilities or all [glMatrix](http://glmatrix.net/) functions in Rust. 
//! 
//! # Quick Start
//!
//! Getting started should be easy if you are already familiar with [glMatrix](http://glmatrix.net/) 
//! and Rust. All functions have been re-named to be idiomatic Rust. 
//! 
//! ```
//! use gl_matrix::common::*;
//! use gl_matrix::{vec3, mat4};
//! 
//! let canvas_w = 800_f32; 
//! let canvas_h = 600_f32;
//!  
//! let mut world_matrix: Mat4 = [0.; 16];
//! let mut view_matrix: Mat4 = [0.; 16];
//! let mut proj_matrix: Mat4 = [0.; 16];
//! 
//! let eye = vec3::from_values(0., 0., -8.);
//! let center = vec3::from_values(0., 0., 0.); 
//! let up = vec3::from_values(0., 1., 0.);
//! 
//! mat4::identity(&mut world_matrix);
//! mat4::look_at(&mut view_matrix, &eye, &center, &up);
//! mat4::perspective(&mut proj_matrix, to_radian(45.), canvas_w / canvas_h, 0.1, Some(100.0));
//! ```

//#![cfg_attr(not(feature = "std"), no_std)]
#[deny(missing_docs)]
pub mod common; 
#[deny(missing_docs)]
pub mod mat2;
#[deny(missing_docs)]
pub mod mat2d;
#[deny(missing_docs)]
pub mod mat3;
#[deny(missing_docs)]
pub mod mat4;
pub mod quat; 
pub mod quat2;
pub mod vec2;
pub mod vec3; 
pub mod vec4; 