use core::arch::asm;
use core::mem::size_of;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Equivalent to _mm_cmpeq_epi8, but hidden from the compiler.
///
/// The use of inline assembly instead of an intrinsic prevents a sufficiently
/// smart compiler from computing the mask in other ways which might not be
/// constant time (for instance, looping through the input and using branching
/// to set the vector elements).
#[must_use]
#[inline(always)]
fn cmpeq_epi8(a: __m128i, b: __m128i) -> __m128i {
    let mut c;
    // When AVX is available, the compiler will use the VEX prefix for all
    // SIMD instructions; do the same for this inline assembly.
    if cfg!(target_feature = "avx") {
        // SAFETY: used only when AVX is available
        // SAFETY: assembly instruction touches only these registers
        unsafe {
            asm!("vpcmpeqb {c}, {a}, {b}",
                c = lateout(xmm_reg) c,
                a = in(xmm_reg) a,
                b = in(xmm_reg) b,
                options(pure, nomem, preserves_flags, nostack));
        }
    } else {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: assembly instruction touches only these registers
        unsafe {
            asm!("pcmpeqb {a}, {b}",
                a = inlateout(xmm_reg) a => c,
                b = in(xmm_reg) b,
                options(pure, nomem, preserves_flags, nostack));
        }
    }
    c
}

/// Equivalent to _mm_and_si128, but hidden from the compiler.
///
/// The use of inline assembly instead of an intrinsic prevents a sufficiently
/// smart compiler from short circuiting the computation once the mask becomes
/// all zeros.
#[must_use]
#[inline(always)]
fn and_si128(a: __m128i, b: __m128i) -> __m128i {
    let mut c;
    // When AVX is available, the compiler will use the VEX prefix for all
    // SIMD instructions; do the same for this inline assembly.
    if cfg!(target_feature = "avx") {
        // SAFETY: used only when AVX is available
        // SAFETY: assembly instruction touches only these registers
        unsafe {
            asm!("vpand {c}, {a}, {b}",
                c = lateout(xmm_reg) c,
                a = in(xmm_reg) a,
                b = in(xmm_reg) b,
                options(pure, nomem, preserves_flags, nostack));
        }
    } else {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: assembly instruction touches only these registers
        unsafe {
            asm!("pand {a}, {b}",
                a = inlateout(xmm_reg) a => c,
                b = in(xmm_reg) b,
                options(pure, nomem, preserves_flags, nostack));
        }
    }
    c
}

/// Equivalent to _mm_movemask_epi8, but hidden from the compiler.
///
/// The use of inline assembly instead of an intrinsic prevents a sufficiently
/// smart compiler from extracting the mask in other ways which might not be
/// constant time (for instance, looping through the elements of the vector).
#[must_use]
#[inline(always)]
fn movemask_epi8(a: __m128i) -> i32 {
    let mut mask;
    // When AVX is available, the compiler will use the VEX prefix for all
    // SIMD instructions; do the same for this inline assembly.
    if cfg!(target_feature = "avx") {
        // SAFETY: used only when AVX is available
        // SAFETY: assembly instruction touches only these registers
        unsafe {
            asm!("vpmovmskb {mask:e}, {a}",
                mask = lateout(reg) mask,
                a = in(xmm_reg) a,
                options(pure, nomem, preserves_flags, nostack));
        }
    } else {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: assembly instruction touches only these registers
        unsafe {
            asm!("pmovmskb {mask:e}, {a}",
                mask = lateout(reg) mask,
                a = in(xmm_reg) a,
                options(pure, nomem, preserves_flags, nostack));
        }
    }
    mask
}

/// Loads a partial vector, possibly duplicating some lanes.
///
/// # Safety
///
/// At least n bytes must be in bounds for the pointer.
#[must_use]
#[inline(always)]
unsafe fn loadu_partial(addr: *const u8, n: usize) -> __m128i {
    debug_assert!(n <= size_of::<__m128i>());

    if n > 64 / 8 {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least n bytes (n > 8) are in bounds for the pointer
        unsafe { _mm_unpacklo_epi64(_mm_loadu_si64(addr), _mm_loadu_si64(addr.add(n - (64 / 8)))) }
    } else if n == 64 / 8 {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least 8 bytes are in bounds for the pointer
        unsafe { _mm_loadu_si64(addr) }
    } else if n > 32 / 8 {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least n bytes (n > 4) are in bounds for the pointer
        unsafe { _mm_unpacklo_epi32(_mm_loadu_si32(addr), _mm_loadu_si32(addr.add(n - (32 / 8)))) }
    } else if n == 32 / 8 {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least 4 bytes are in bounds for the pointer
        unsafe { _mm_loadu_si32(addr) }
    } else if n > 16 / 8 {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least n bytes (n > 2) are in bounds for the pointer
        unsafe { _mm_unpacklo_epi16(_mm_loadu_si16(addr), _mm_loadu_si16(addr.add(n - (16 / 8)))) }
    } else if n == 16 / 8 {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least 2 bytes are in bounds for the pointer
        unsafe { _mm_loadu_si16(addr) }
    } else if n > 0 {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least 1 byte is in bounds for the pointer
        unsafe { _mm_set_epi64x(0, (*addr).into()) }
    } else {
        // SAFETY: this file is compiled only when SSE2 is available
        unsafe { _mm_setzero_si128() }
    }
}

/// SSE2/AVX implementation of constant_time_eq and constant_time_eq_n.
///
/// # Safety
///
/// At least n bytes must be in bounds for both pointers.
#[must_use]
#[inline(always)]
unsafe fn constant_time_eq_sse2(mut a: *const u8, mut b: *const u8, mut n: usize) -> bool {
    const LANES: usize = size_of::<__m128i>();

    let mut mask;

    if n >= LANES * 2 {
        let mut mask0;
        let mut mask1;

        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least 256 bits are in bounds for both pointers
        unsafe {
            let tmpa0 = _mm_loadu_si128(a as *const __m128i);
            let tmpb0 = _mm_loadu_si128(b as *const __m128i);
            let tmpa1 = _mm_loadu_si128(a.add(LANES) as *const __m128i);
            let tmpb1 = _mm_loadu_si128(b.add(LANES) as *const __m128i);

            a = a.add(LANES * 2);
            b = b.add(LANES * 2);
            n -= LANES * 2;

            mask0 = cmpeq_epi8(tmpa0, tmpb0);
            mask1 = cmpeq_epi8(tmpa1, tmpb1);
        }

        while n >= LANES * 2 {
            // SAFETY: this file is compiled only when SSE2 is available
            // SAFETY: at least 256 bits are in bounds for both pointers
            unsafe {
                let tmpa0 = _mm_loadu_si128(a as *const __m128i);
                let tmpb0 = _mm_loadu_si128(b as *const __m128i);
                let tmpa1 = _mm_loadu_si128(a.add(LANES) as *const __m128i);
                let tmpb1 = _mm_loadu_si128(b.add(LANES) as *const __m128i);

                a = a.add(LANES * 2);
                b = b.add(LANES * 2);
                n -= LANES * 2;

                let tmp0 = cmpeq_epi8(tmpa0, tmpb0);
                let tmp1 = cmpeq_epi8(tmpa1, tmpb1);

                mask0 = and_si128(mask0, tmp0);
                mask1 = and_si128(mask1, tmp1);
            }
        }

        if n >= LANES {
            // SAFETY: this file is compiled only when SSE2 is available
            // SAFETY: at least 128 bits are in bounds for both pointers
            unsafe {
                let tmpa = _mm_loadu_si128(a as *const __m128i);
                let tmpb = _mm_loadu_si128(b as *const __m128i);

                a = a.add(LANES);
                b = b.add(LANES);
                n -= LANES;

                let tmp = cmpeq_epi8(tmpa, tmpb);

                mask0 = and_si128(mask0, tmp);
            }
        }

        if n > 0 {
            // SAFETY: at least n bytes are in bounds for both pointers
            unsafe {
                let tmpa = loadu_partial(a, n);
                let tmpb = loadu_partial(b, n);

                let tmp = cmpeq_epi8(tmpa, tmpb);

                mask1 = and_si128(mask1, tmp);
            }
        }

        mask = and_si128(mask0, mask1);
    } else if n >= LANES {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least 128 bits are in bounds for both pointers
        unsafe {
            let tmpa = _mm_loadu_si128(a as *const __m128i);
            let tmpb = _mm_loadu_si128(b as *const __m128i);

            a = a.add(LANES);
            b = b.add(LANES);
            n -= LANES;

            mask = cmpeq_epi8(tmpa, tmpb);
        }

        if n > 0 {
            // SAFETY: at least n bytes are in bounds for both pointers
            unsafe {
                let tmpa = loadu_partial(a, n);
                let tmpb = loadu_partial(b, n);

                let tmp = cmpeq_epi8(tmpa, tmpb);

                mask = and_si128(mask, tmp);
            }
        }
    } else if n > 0 {
        // SAFETY: at least n bytes are in bounds for both pointers
        unsafe {
            let tmpa = loadu_partial(a, n);
            let tmpb = loadu_partial(b, n);

            mask = cmpeq_epi8(tmpa, tmpb);
        }
    } else {
        return true;
    }

    movemask_epi8(mask) == 0xFFFF
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
    a.len() == b.len() && unsafe { constant_time_eq_sse2(a.as_ptr(), b.as_ptr(), a.len()) }
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
    unsafe { constant_time_eq_sse2(a.as_ptr(), b.as_ptr(), N) }
}
