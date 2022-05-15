use crate::{hide::optimizer_hide, ConstTimeEq};

impl<T: AsRef<[u8]>> ConstTimeEq<T> for &[u8] {
    #[inline]
    fn constant_time_ne(&self, other: &T) -> u8 {
        let other = other.as_ref();
        assert!(self.len() == other.len());

        // These useless slices make the optimizer elide the bounds checks.
        // See the comment in clone_from_slice() added on Rust commit 6a7bc47.
        let len = self.len();
        let a = &self[..len];
        let b = &other[..len];

        let mut tmp = 0;
        for i in 0..len {
            tmp |= a[i] ^ b[i];
        }

        // The compare with 0 must happen outside this function.
        optimizer_hide(tmp)
    }
}

impl<const SIZE: usize> ConstTimeEq<[u8; SIZE]> for [u8; SIZE] {
    #[inline]
    fn constant_time_ne(&self, other: &[u8; SIZE]) -> u8 {
        let mut tmp = 0;
        for i in 0..SIZE {
            tmp |= self[i] ^ other[i];
        }

        // The compare with 0 must happen outside this function.
        optimizer_hide(tmp)
    }
}

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
pub fn constant_time_eq_16(a: &[u8; 16], b: &[u8; 16]) -> bool {
    a.constant_time_eq(b)
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
pub fn constant_time_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.constant_time_eq(b)
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
pub fn constant_time_eq_64(a: &[u8; 64], b: &[u8; 64]) -> bool {
    a.constant_time_eq(b)
}

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
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.constant_time_eq(&b)
}
