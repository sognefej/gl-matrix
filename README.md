# gl-matrix

[![Build Status](https://travis-ci.org/sognefej/gl-matrix.svg?branch=master)](https://travis-ci.org/sognefej/gl-matrix)
[![Coverage Status](https://coveralls.io/repos/github/sognefej/gl-matrix/badge.svg)](https://coveralls.io/github/sognefej/gl-matrix)
[![Crate](https://img.shields.io/crates/v/rand.svg)](https://crates.io/crates/gl_matrix)

A Rust implementation of [glMatrix](http://glmatrix.net/)

gl_matrix provides utilities or all [glMatrix](http://glmatrix.net/) functions in Rust. 
 
 # Quick Start

 Getting started should be easy if you are already familiar with [glMatrix](http://glmatrix.net/) 
 and Rust. All functions have been re-named to be idiomatic Rust. 
 
 ```rust
 use gl_matrix::common::*;
 use gl_matrix::{vec3, mat4};
 
 let canvas_w = 800_f32; 
 let canvas_h = 600_f32;
  
 let mut world_matrix: Mat4 = [0.; 16];
 let mut view_matrix: Mat4 = [0.; 16];
 let mut proj_matrix: Mat4 = [0.; 16];
 
 let eye = vec3::from_values(0., 0., -8.);
 let center = vec3::from_values(0., 0., 0.); 
 let up = vec3::from_values(0., 1., 0.);
 
 mat4::identity(&mut world_matrix);
 mat4::look_at(&mut view_matrix, &eye, &center, &up);
 mat4::perspective(&mut proj_matrix, to_radian(45.), canvas_w / canvas_h, 0.1, Some(100.0));
```
## Features 
- No std 
- Compiles to wasm 
- Familar syntax

## Future work 
- Add support for f64 (Currently only f32) 
- Add support for forEach 

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
gl-matrix = 0.0
```

## Contributing 

See [CONTRIBUTING.md](./CONTRIBUTING.md)

## License
gl-matrix is under the terms of the MIT license.

[LICENSE](LICENSE)
