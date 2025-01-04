#![no_std]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

#[doc(hidden)]
pub mod classic;

#[doc(hidden)]
pub mod generic;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
    not(miri)
))]
mod sse2;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2",
    not(miri)
))]
use sse2 as simd;

#[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
mod neon;

#[cfg(all(target_arch = "aarch64", target_feature = "neon", not(miri)))]
use neon as simd;

#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2",
        not(miri)
    ),
    all(target_arch = "aarch64", target_feature = "neon", not(miri))
)))]
use generic as simd;

/// Compares two equal-sized byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq;
///
/// assert!(constant_time_eq(b"foo", b"foo"));
/// assert!(!constant_time_eq(b"foo", b"bar"));
/// assert!(!constant_time_eq(b"bar", b"baz"));
/// # assert!(constant_time_eq(b"", b""));
///
/// // Not equal-sized, so won't take constant time.
/// assert!(!constant_time_eq(b"foo", b""));
/// assert!(!constant_time_eq(b"foo", b"quux"));
/// ```
#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    simd::constant_time_eq(a, b)
}

// Fixed-size array variant.

/// Compares two fixed-size byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq_n;
///
/// assert!(constant_time_eq_n(&[3; 20], &[3; 20]));
/// assert!(!constant_time_eq_n(&[3; 20], &[7; 20]));
/// ```
#[must_use]
pub fn constant_time_eq_n<const N: usize>(a: &[u8; N], b: &[u8; N]) -> bool {
    simd::constant_time_eq_n(a, b)
}

// Fixed-size variants for the most common sizes.

/// Compares two 128-bit byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq_16;
///
/// assert!(constant_time_eq_16(&[3; 16], &[3; 16]));
/// assert!(!constant_time_eq_16(&[3; 16], &[7; 16]));
/// ```
#[inline]
#[must_use]
pub fn constant_time_eq_16(a: &[u8; 16], b: &[u8; 16]) -> bool {
    constant_time_eq_n(a, b)
}

/// Compares two 256-bit byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq_32;
///
/// assert!(constant_time_eq_32(&[3; 32], &[3; 32]));
/// assert!(!constant_time_eq_32(&[3; 32], &[7; 32]));
/// ```
#[inline]
#[must_use]
pub fn constant_time_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    constant_time_eq_n(a, b)
}

/// Compares two 512-bit byte strings in constant time.
///
/// # Examples
///
/// ```
/// use constant_time_eq::constant_time_eq_64;
///
/// assert!(constant_time_eq_64(&[3; 64], &[3; 64]));
/// assert!(!constant_time_eq_64(&[3; 64], &[7; 64]));
/// ```
#[inline]
#[must_use]
pub fn constant_time_eq_64(a: &[u8; 64], b: &[u8; 64]) -> bool {
    constant_time_eq_n(a, b)
}
