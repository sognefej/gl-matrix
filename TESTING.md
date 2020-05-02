# Testing 

gl-matrix strives to have 100% code coverage. If a pull request changes 
the coverage statistics please update the test or write a new test to fix the
coverage. When writing new tests or updating existing ones please: 

- Follow the existing naming conventions 
- Hit all lines 

Writing tests the cover all possible cases would be crazy just make 
sure you hit all lines, and hit the most pertinent cases. 

gl-matrix uses [tarpaulin](https://github.com/xd009642/tarpaulin) to 
generate coverage stats. [tarpaulin](https://github.com/xd009642/tarpaulin) however
is still under development. If coverage is missing due to [tarpaulin](https://github.com/xd009642/tarpaulin) 
and not your test don't worry about it. Just make a note in the pull request if that happens.

Before you submit your request: 

```bash
cargo build
cargo test 
cargo check --target wasm32-unknown-unknown
```

If you can run [tarpaulin](https://github.com/xd009642/tarpaulin) on your 
code before submitting the request, however at the moment [tarpaulin](https://github.com/xd009642/tarpaulin) 
only supports x86_64 Linux. If you don't have this available to you just make sure to check the build stats
on travis and the coverage stats for your pull request on coveralls. 