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
fn movemask_epi8(a: __m128i) -> u32 {
    let mut mask;
    // When AVX is available, the compiler will use the VEX prefix for all
    // SIMD instructions; do the same for this inline assembly.
    if cfg!(target_feature = "avx") {
        // SAFETY: used only when AVX is available
        // SAFETY: assembly instruction touches only these registers
        // SAFETY: 32-bit operations zero-extend the 64-bit register
        unsafe {
            asm!("vpmovmskb {mask:e}, {a}",
                mask = lateout(reg) mask,
                a = in(xmm_reg) a,
                options(pure, nomem, preserves_flags, nostack));
        }
    } else {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: assembly instruction touches only these registers
        // SAFETY: 32-bit operations zero-extend the 64-bit register
        unsafe {
            asm!("pmovmskb {mask:e}, {a}",
                mask = lateout(reg) mask,
                a = in(xmm_reg) a,
                options(pure, nomem, preserves_flags, nostack));
        }
    }
    // The return type is u32 instead of i32 to avoid a sign extension.
    mask
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

    let tmp = if n >= LANES * 2 {
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

        let mask = and_si128(mask0, mask1);
        movemask_epi8(mask) ^ 0xFFFF
    } else if n >= LANES {
        // SAFETY: this file is compiled only when SSE2 is available
        // SAFETY: at least 128 bits are in bounds for both pointers
        let mask = unsafe {
            let tmpa = _mm_loadu_si128(a as *const __m128i);
            let tmpb = _mm_loadu_si128(b as *const __m128i);

            a = a.add(LANES);
            b = b.add(LANES);
            n -= LANES;

            cmpeq_epi8(tmpa, tmpb)
        };

        movemask_epi8(mask) ^ 0xFFFF
    } else {
        0
    };

    // SAFETY: at least n bytes are in bounds for both pointers
    unsafe { crate::generic::constant_time_eq_impl(a, b, n, tmp.into()) }
}

#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    // SAFETY: both pointers point to the same number of bytes
    a.len() == b.len() && unsafe { constant_time_eq_sse2(a.as_ptr(), b.as_ptr(), a.len()) }
}

#[must_use]
pub fn constant_time_eq_n<const N: usize>(a: &[u8; N], b: &[u8; N]) -> bool {
    // SAFETY: both pointers point to N bytes
    unsafe { constant_time_eq_sse2(a.as_ptr(), b.as_ptr(), N) }
}
