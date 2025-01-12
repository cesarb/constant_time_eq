use core::arch::asm;

#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Equivalent to vceqq_u8, but hidden from the compiler.
///
/// The use of inline assembly instead of an intrinsic prevents a sufficiently
/// smart compiler from computing the mask in other ways which might not be
/// constant time (for instance, looping through the input and using branching
/// to set the vector elements).
#[must_use]
#[inline(always)]
fn vceqq_u8_hide(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut c;
    // SAFETY: this file is compiled only when NEON is available
    // SAFETY: assembly instruction touches only these registers
    #[cfg(target_arch = "aarch64")]
    unsafe {
        asm!("cmeq {c:v}.16b, {a:v}.16b, {b:v}.16b",
            c = lateout(vreg) c,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, preserves_flags, nostack));
    }
    c
}

/// Equivalent to vandq_u8, but hidden from the compiler.
///
/// The use of inline assembly instead of an intrinsic prevents a sufficiently
/// smart compiler from short circuiting the computation once the mask becomes
/// all zeros.
#[must_use]
#[inline(always)]
fn vandq_u8_hide(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut c;
    // SAFETY: this file is compiled only when NEON is available
    // SAFETY: assembly instruction touches only these registers
    #[cfg(target_arch = "aarch64")]
    unsafe {
        asm!("and {c:v}.16b, {a:v}.16b, {b:v}.16b",
            c = lateout(vreg) c,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, preserves_flags, nostack));
    }
    c
}

/// Equivalent to vshrn_n_u16(..., 4), but hidden from the compiler.
///
/// The use of inline assembly instead of an intrinsic prevents a sufficiently
/// smart compiler from extracting the mask in other ways which might not be
/// constant time (for instance, looping through the elements of the vector).
#[must_use]
#[inline(always)]
fn vshrn_n_u16_4_hide(a: uint16x8_t) -> uint8x8_t {
    let mut mask;
    // SAFETY: this file is compiled only when NEON is available
    // SAFETY: assembly instruction touches only these registers
    #[cfg(target_arch = "aarch64")]
    unsafe {
        asm!("shrn {mask:v}.8b, {a:v}.8h, #{n}",
            mask = lateout(vreg) mask,
            a = in(vreg) a,
            n = const 4,
            options(pure, nomem, preserves_flags, nostack));
    }
    mask
}

/// Moves a mask created by vceqq_u8 to a u64 register, with each all-zero or
/// all-ones mask byte represented as an all-zero or all-ones half-byte.
#[must_use]
#[inline(always)]
fn get_mask_u64(mask: uint8x16_t) -> u64 {
    // SAFETY: this file is compiled only when NEON is available
    unsafe {
        let mask = vshrn_n_u16_4_hide(vreinterpretq_u16_u8(mask));
        vget_lane_u64(vreinterpret_u64_u8(mask), 0)
    }
}

/// NEON implementation of constant_time_eq and constant_time_eq_n.
///
/// # Safety
///
/// At least n bytes must be in bounds for both pointers.
#[must_use]
#[inline(always)]
unsafe fn constant_time_eq_neon(mut a: *const u8, mut b: *const u8, mut n: usize) -> bool {
    const LANES: usize = 16;

    let tmp = if n >= LANES * 2 {
        let mut mask0;
        let mut mask1;

        // SAFETY: this file is compiled only when NEON is available
        // SAFETY: at least 256 bits are in bounds for both pointers
        unsafe {
            let tmpa = vld1q_u8_x2(a);
            let tmpb = vld1q_u8_x2(b);

            a = a.add(LANES * 2);
            b = b.add(LANES * 2);
            n -= LANES * 2;

            mask0 = vceqq_u8_hide(tmpa.0, tmpb.0);
            mask1 = vceqq_u8_hide(tmpa.1, tmpb.1);
        }

        while n >= LANES * 2 {
            // SAFETY: this file is compiled only when NEON is available
            // SAFETY: at least 256 bits are in bounds for both pointers
            unsafe {
                let tmpa = vld1q_u8_x2(a);
                let tmpb = vld1q_u8_x2(b);

                a = a.add(LANES * 2);
                b = b.add(LANES * 2);
                n -= LANES * 2;

                let tmp0 = vceqq_u8_hide(tmpa.0, tmpb.0);
                let tmp1 = vceqq_u8_hide(tmpa.1, tmpb.1);

                mask0 = vandq_u8_hide(mask0, tmp0);
                mask1 = vandq_u8_hide(mask1, tmp1);
            }
        }

        if n >= LANES {
            // SAFETY: this file is compiled only when NEON is available
            // SAFETY: at least 128 bits are in bounds for both pointers
            unsafe {
                let tmpa = vld1q_u8(a);
                let tmpb = vld1q_u8(b);

                a = a.add(LANES);
                b = b.add(LANES);
                n -= LANES;

                let tmp = vceqq_u8_hide(tmpa, tmpb);

                mask0 = vandq_u8_hide(mask0, tmp);
            }
        }

        let mask = vandq_u8_hide(mask0, mask1);
        get_mask_u64(mask) ^ !0
    } else if n >= LANES {
        // SAFETY: this file is compiled only when NEON is available
        // SAFETY: at least 128 bits are in bounds for both pointers
        let mask = unsafe {
            let tmpa = vld1q_u8(a);
            let tmpb = vld1q_u8(b);

            a = a.add(LANES);
            b = b.add(LANES);
            n -= LANES;

            vceqq_u8_hide(tmpa, tmpb)
        };

        get_mask_u64(mask) ^ !0
    } else {
        0
    };

    // SAFETY: at least n bytes are in bounds for both pointers
    unsafe { crate::generic::constant_time_eq_impl(a, b, n, tmp) }
}

#[must_use]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    // SAFETY: both pointers point to the same number of bytes
    a.len() == b.len() && unsafe { constant_time_eq_neon(a.as_ptr(), b.as_ptr(), a.len()) }
}

#[must_use]
pub fn constant_time_eq_n<const N: usize>(a: &[u8; N], b: &[u8; N]) -> bool {
    // SAFETY: both pointers point to N bytes
    unsafe { constant_time_eq_neon(a.as_ptr(), b.as_ptr(), N) }
}
