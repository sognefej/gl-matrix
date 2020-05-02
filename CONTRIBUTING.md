# Contributing Guidelines
----------------------

Contributions are welcome and an important part of any open source project!

---
gl-matrix is downstream of [glMatrix](http://glmatrix.net/). New functionality in this project that is 
not first accepted into [glMatrix](http://glmatrix.net/) will not be accepted here. If there is functionality 
you feel is missing please attempt to add it to [glMatrix](http://glmatrix.net/) before submitting it here. 

## Project objectives

gl-matrix is intended to be a complete Rust port of [glMatrix](http://glmatrix.net/) for those who want 
to compile Rust to [Web­Assembly](https://www.rust-lang.org/what/wasm) for [WebGL](https://rustwasm.github.io/wasm-bindgen/examples/webgl.html) 
programs. 

## What to contribute

- Missing functionality (Features available in [glMatrix](http://glmatrix.net/) not available in gl-matrix).
- Fixes to missing coverage.
- Fixes to bad tests.
- Optimizations (Improvements to compilation into wasm).
- Bug fixes.
- Fixes to missing or bad documentation. 

## Bugs 

Make sure you have the latest version of the package installed and see if that does not fix your issue. 
If the issue still exists consider opening an issue or submit a pull request with the bug fix. 

## Submitting pull requests

For optimization please provide a reasoning and a way to show that the optimization is an improvement to 
the code base. gl-matrix uses ```#[deny(missing_docs)]``` on each module...so please add documentation to new/missing
functionality by following the conventions of the existing documentation.

### Tests
Read this before submitting your pull request. 

See [TESTING.md](./TESTING.md)

### Code review
Maintainers tend to be picky. Make sure to: 

- Follow the conventions of the existing code 
- Follow Rust's naming conventions 
- Have no warnings when building 
- read [TESTING.md](./TESTING.md) 