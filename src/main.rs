// SPDX-License-Identifier: CC-BY-4.0

//! Native desktop entry point. The browser build is driven by the `web_start`
//! entry in the library crate (see `src/lib.rs`); this binary is not used there.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    pollster::block_on(gpu_path_tracing::run())
}

// The library exposes `web_start` for wasm; the binary is a no-op there so a
// `cargo build --target wasm32-unknown-unknown` of the whole crate still works.
#[cfg(target_arch = "wasm32")]
fn main() {}
