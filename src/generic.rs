use core::mem::size_of;
use core::ops::BitXor;
use core::ptr::read_unaligned;

/// The natural word type for this architecture. All bit patterns must be valid for this type.
type Word = usize;

/// Hides a value from the optimizer.
#[cfg(all(
    not(miri),
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "loongarch64"
    )
))]
#[inline(always)]
#[must_use]
fn optimizer_hide(mut value: Word) -> Word {
    // SAFETY: the input value is passed unchanged to the output, the inline assembly does nothing.
    unsafe {
        core::arch::asm!("/* {0} */", inlateout(reg) value, options(pure, nomem, preserves_flags, nostack));
    }
    value
}

/// Attempts to hide a value from the optimizer.
#[cfg(any(
    miri,
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "loongarch64"
    ))
))]
#[inline(never)]
#[must_use]
fn optimizer_hide(value: Word) -> Word {
    // The current implementation of black_box in the main codegen backends is similar to
    // {
    //     let result = value;
    //     asm!("", in(reg) &result);
    //     result
    // }
    // which round-trips the value through the stack, instead of leaving it in a register.
    // Experimental codegen backends might implement black_box as a pure identity function,
    // without the expected optimization barrier, so it's less guaranteed than inline asm.
    // For that reason, we also use the #[inline(never)] hint, which makes it harder for an
    // optimizer to look inside this function.
    core::hint::black_box(value)
}

/// Generic implementation of constant_time_eq and constant_time_eq_n.
///
/// # Safety
///
/// At least n bytes must be in bounds for both pointers.
#[must_use]
#[inline(always)]
unsafe fn constant_time_eq_impl(mut a: *const u8, mut b: *const u8, mut n: usize) -> bool {
    /// Reads and compares a single word from the input, adjusting the pointers and counter.
    /// Returns zero if both words are equal, non-zero if any byte is different.
    ///
    /// # Safety
    ///
    /// Enough bytes must be in bounds for both pointers; all bit patterns must be valid for T.
    #[must_use]
    #[inline(always)]
    unsafe fn read_step<T: BitXor<Output = T>>(
        a: &mut *const u8,
        b: &mut *const u8,
        n: &mut usize,
    ) -> T {
        // SAFETY: enough bytes are within bounds for both pointers; all bit patterns are valid
        let tmp = unsafe { read_unaligned(*a as *const T) ^ read_unaligned(*b as *const T) };

        // SAFETY: enough bytes are within bounds for both pointers
        unsafe {
            *a = a.add(size_of::<T>());
            *b = b.add(size_of::<T>());
        }
        *n -= size_of::<T>();

        tmp
    }

    // The optimizer is not allowed to assume anything about the value of tmp after each iteration,
    // which prevents it from terminating the loop early if the value becomes non-zero or all-ones.
    let mut tmp = 0;

    while n >= size_of::<Word>() {
        // SAFETY: enough bytes for Word are within bounds; all bit patterns are valid for Word
        tmp = optimizer_hide(tmp | unsafe { read_step::<Word>(&mut a, &mut b, &mut n) });
    }

    while n >= size_of::<u128>() {
        // SAFETY: enough bytes for u128 are within bounds; all bit patterns are valid for u128
        tmp = optimizer_hide(tmp | unsafe { read_step::<u128>(&mut a, &mut b, &mut n) } as Word);
    }
    if n >= size_of::<u64>() {
        // SAFETY: enough bytes for u64 are within bounds; all bit patterns are valid for u64
        tmp = optimizer_hide(tmp | unsafe { read_step::<u64>(&mut a, &mut b, &mut n) } as Word);
    }
    if n >= size_of::<u32>() {
        // SAFETY: enough bytes for u32 are within bounds; all bit patterns are valid for u32
        tmp = optimizer_hide(tmp | unsafe { read_step::<u32>(&mut a, &mut b, &mut n) } as Word);
    }
    if n >= size_of::<u16>() {
        // SAFETY: enough bytes for u16 are within bounds; all bit patterns are valid for u16
        tmp = optimizer_hide(tmp | unsafe { read_step::<u16>(&mut a, &mut b, &mut n) } as Word);
    }
    if n >= size_of::<u8>() {
        // SAFETY: enough bytes for u8 are within bounds; all bit patterns are valid for u8
        tmp = optimizer_hide(tmp | unsafe { read_step::<u8>(&mut a, &mut b, &mut n) } as Word);
    }

    tmp == 0
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
#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    // SAFETY: both pointers point to the same number of bytes
    a.len() == b.len() && unsafe { constant_time_eq_impl(a.as_ptr(), b.as_ptr(), a.len()) }
}

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
    // SAFETY: both pointers point to N bytes
    unsafe { constant_time_eq_impl(a.as_ptr(), b.as_ptr(), N) }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "count_instructions_test")]
    extern crate std;

    #[cfg(feature = "count_instructions_test")]
    #[test]
    fn count_optimizer_hide_instructions() -> std::io::Result<()> {
        use super::{optimizer_hide, Word};
        use count_instructions::count_instructions;

        fn count() -> std::io::Result<usize> {
            // If optimizer_hide does not work, constant propagation and folding
            // will make this identical to count_optimized() below.
            let mut count = 0;
            assert_eq!(
                10 as Word,
                count_instructions(
                    || optimizer_hide(1)
                        + optimizer_hide(2)
                        + optimizer_hide(3)
                        + optimizer_hide(4),
                    |_| count += 1
                )?
            );
            Ok(count)
        }

        fn count_optimized() -> std::io::Result<usize> {
            #[inline(always)]
            fn inline_identity(value: Word) -> Word {
                value
            }

            let mut count = 0;
            assert_eq!(
                10 as Word,
                count_instructions(
                    || inline_identity(1)
                        + inline_identity(2)
                        + inline_identity(3)
                        + inline_identity(4),
                    |_| count += 1
                )?
            );
            Ok(count)
        }

        assert!(count()? > count_optimized()?);
        Ok(())
    }
}
