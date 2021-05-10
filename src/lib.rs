#![no_std]

pub trait ConstantTimeEq<B> {
    #[doc(hidden)]
    fn constant_time_ne(&self, other: &B) -> u8;

    /// Compares two equal-sized pieces of data in constant time.
    /// # Examples
    ///
    /// ```rust
    /// use constant_time_eq::ConstantTimeEq;
    ///
    /// let array = &[3; 64];
    /// assert!(array.constant_time_eq(&[3; 64]));
    /// assert!(!array.constant_time_eq(&[7; 64]));
    /// assert!(b"foo".constant_time_eq(b"foo"));
    /// assert!(!b"foo".constant_time_eq(b"bar"));
    /// ```
    fn constant_time_eq(&self, other: &B) -> bool;
}

#[cfg(feature = "const_generics")]
impl<const LEN: usize> ConstantTimeEq<[u8; LEN]> for [u8; LEN] {
    // This function is non-inline to prevent the optimizer from looking inside it.
    #[inline(never)]
    fn constant_time_ne(&self, other: &Self) -> u8 {
        let mut tmp = 0;
        for i in 0..LEN {
            tmp |= self[i] ^ other[i];
        }
        tmp // The compare with 0 must happen outside this function.
    }

    fn constant_time_eq(&self, other: &Self) -> bool {
        self.constant_time_ne(other) == 0
    }
}

// This is solely to avoid an extra call to as_ref in ConstantTimeEq<AsRef<[u8]>> for &[u8]
#[inline(never)]
fn constant_time_ne_array(a: &[u8], b: &[u8]) -> u8 {
    assert!(a.len() == b.len());
    // These useless slices make the optimizer elide the bounds checks.
    // See the comment in clone_from_slice() added on Rust commit 6a7bc47.
    let len = a.len();
    let a = &a[..len];
    let b = &b[..len];

    let mut tmp = 0;
    for i in 0..len {
        tmp |= a[i] ^ b[i];
    }
    tmp // The compare with 0 must happen outside this function.
}

impl<B: AsRef<[u8]>> ConstantTimeEq<B> for &[u8] {
    fn constant_time_ne(&self, other: &B) -> u8 {
        constant_time_ne_array(self, other.as_ref())
    }

    fn constant_time_eq(&self, other: &B) -> bool {
        let b = other.as_ref();
        self.len() == b.len() && constant_time_ne_array(self, b) == 0
    }
}

#[cfg(not(feature = "const_generics"))]
macro_rules! constant_time_impl {
    ($n:expr) => {
        impl ConstantTimeEq<[u8; $n]> for [u8; $n] {
            #[inline(never)]
            fn constant_time_ne(&self, other: &[u8; $n]) -> u8 {
                let mut tmp = 0;
                for i in 0..$n {
                    tmp |= self[i] ^ other[i];
                }
                tmp // The compare with 0 must happen outside this function.
            }

            fn constant_time_eq(&self, other: &[u8; $n]) -> bool {
                self.constant_time_ne(other) == 0
            }
        }
    };
}

#[cfg(not(feature = "const_generics"))]
constant_time_impl!(16);
#[cfg(not(feature = "const_generics"))]
constant_time_impl!(32);
#[cfg(not(feature = "const_generics"))]
constant_time_impl!(64);
