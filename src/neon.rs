use core::arch::asm;
use core::mem::size_of;
use core::ptr::read_unaligned;

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

/// Loads a partial vector, possibly duplicating some lanes.
///
/// # Safety
///
/// At least n bytes must be in bounds for the pointer.
#[must_use]
#[inline(always)]
unsafe fn vld1q_u8_partial(ptr: *const u8, n: usize) -> uint8x16_t {
    debug_assert!(n <= 16);

    #[must_use]
    #[inline(always)]
    unsafe fn load_unaligned<T>(ptr: *const u8) -> uint8x16_t
    where
        T: Into<u64>,
    {
        // TODO safety comment
        unsafe {
            vcombine_u8(
                vcreate_u8(read_unaligned(ptr as *const T).into()),
                vcreate_u8(0),
            )
        }
    }

    #[must_use]
    #[inline(always)]
    unsafe fn load_unaligned_pair<A, B>(ptr: *const u8) -> uint8x16_t
    where
        A: Into<u64>,
        B: Into<u64>,
    {
        // TODO safety comment
        unsafe {
            let a = read_unaligned(ptr as *const A).into();
            let b = read_unaligned(ptr.add(size_of::<A>()) as *const B).into();
            let c = a | (b << (size_of::<A>() * 8));
            vcombine_u8(vcreate_u8(c), vcreate_u8(0))
        }
    }

    match n {
        9.. => {
            // TODO safety comment
            unsafe { vcombine_u8(vld1_u8(ptr), vld1_u8(ptr.add(n - 8))) }
        }
        8 => {
            // TODO safety comment
            unsafe { vcombine_u8(vld1_u8(ptr), vcreate_u8(0)) }
        }
        7 => {
            // TODO safety comment
            unsafe {
                vcombine_u8(
                    vcreate_u8(read_unaligned(ptr as *const u32).into()),
                    vcreate_u8(read_unaligned(ptr.add(n - 4) as *const u32).into()),
                )
            }
        }
        6 => {
            // TODO safety comment
            unsafe { load_unaligned_pair::<u32, u16>(ptr) }
        }
        5 => {
            // TODO safety comment
            unsafe { load_unaligned_pair::<u32, u8>(ptr) }
        }
        4 => {
            // TODO safety comment
            unsafe { load_unaligned::<u32>(ptr) }
        }
        3 => {
            // TODO safety comment
            unsafe { load_unaligned_pair::<u16, u8>(ptr) }
        }
        2 => {
            // TODO safety comment
            unsafe { load_unaligned::<u16>(ptr) }
        }
        1 => {
            // TODO safety comment
            unsafe { load_unaligned::<u8>(ptr) }
        }
        0 => {
            // TODO safety comment
            unsafe { vcombine_u8(vcreate_u8(0), vcreate_u8(0)) }
        }
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

    let mut mask;

    if n >= LANES * 2 {
        let mut mask0;
        let mut mask1;

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

        if n > 0 {
            unsafe {
                let tmpa = vld1q_u8_partial(a, n);
                let tmpb = vld1q_u8_partial(b, n);

                let tmp = vceqq_u8_hide(tmpa, tmpb);

                mask1 = vandq_u8_hide(mask1, tmp);
            }
        }

        mask = vandq_u8_hide(mask0, mask1);
    } else if n >= LANES {
        unsafe {
            let tmpa = vld1q_u8(a);
            let tmpb = vld1q_u8(b);

            a = a.add(LANES);
            b = b.add(LANES);
            n -= LANES;

            mask = vceqq_u8_hide(tmpa, tmpb);
        }

        if n > 0 {
            unsafe {
                let tmpa = vld1q_u8_partial(a, n);
                let tmpb = vld1q_u8_partial(b, n);

                let tmp = vceqq_u8_hide(tmpa, tmpb);

                mask = vandq_u8_hide(mask, tmp);
            }
        }
    } else if n > 0 {
        unsafe {
            let tmpa = vld1q_u8_partial(a, n);
            let tmpb = vld1q_u8_partial(b, n);

            mask = vceqq_u8_hide(tmpa, tmpb);
        }
    } else {
        return true;
    }

    unsafe {
        let mask = vshrn_n_u16_4_hide(vreinterpretq_u16_u8(mask));
        vget_lane_u64(vreinterpret_u64_u8(mask), 0) == !0
    }
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
