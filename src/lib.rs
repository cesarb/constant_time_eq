#![no_std]

pub(crate) mod hide;
#[cfg(not(feature = "const-generics"))]
mod imp;
#[cfg(feature = "const-generics")]
mod imp_cg;

#[cfg(not(feature = "const-generics"))]
pub use imp::*;
#[cfg(feature = "const-generics")]
pub use imp_cg::*;

/// This trait is used to implement a constant time equality checking function.
/// It is expected that both sides are equal in size.
///
/// # Examples
/// ```rust
/// use constant_time_eq::ConstantTimeEq;
///
/// let x = b"foo";
/// let y = b"foo";
/// let z = b"bar";
/// // These two will take the exact same amount of clock cycles!
/// assert!(x.constant_time_eq(&y));
/// assert!(!x.constant_time_eq(&z));
/// ```
pub trait ConstTimeEq<Other> {
    fn constant_time_ne(&self, other: &Other) -> u8;
    #[inline]
    fn constant_time_eq(&self, other: &Other) -> bool {
        self.constant_time_ne(other) == 0
    }
}
