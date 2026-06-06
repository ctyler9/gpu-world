// SPDX-License-Identifier: CC-BY-4.0

//! Small platform shims so the rest of the code can stay platform-agnostic.
//!
//! `std::time::Instant`/`SystemTime` panic on `wasm32-unknown-unknown`; the
//! `web-time` crate provides drop-in replacements backed by the browser's
//! `Performance`/`Date` APIs.

#[cfg(target_arch = "wasm32")]
pub use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;
