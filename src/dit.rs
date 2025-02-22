//! Runs code with the hardware DIT feature enabled when possible.
//!
//! With the "std" feature, detects at compilation time and runtime whether FEAT_DIT is available,
//! and uses it (together with FEAT_SB when available) to enable the data independent timing mode
//! of the processor.
//!
//! Without the "std" feature, this detection is done at compilation time only, which is enough for
//! some targets like aarch64-apple-darwin which is known to always have these features.

use core::arch::asm;

/// Checks whether FEAT_DIT is available.
#[cfg(feature = "std")]
#[inline(always)]
fn is_feat_dit_implemented() -> bool {
    std::arch::is_aarch64_feature_detected!("dit")
}

/// Checks whether FEAT_SB is available.
#[cfg(feature = "std")]
#[inline(always)]
fn is_feat_sb_implemented() -> bool {
    std::arch::is_aarch64_feature_detected!("sb")
}

/// Checks whether FEAT_DIT is available.
#[cfg(not(feature = "std"))]
#[inline(always)]
fn is_feat_dit_implemented() -> bool {
    cfg!(target_feature = "dit")
}

/// Checks whether FEAT_SB is available.
#[cfg(not(feature = "std"))]
#[inline(always)]
fn is_feat_sb_implemented() -> bool {
    cfg!(target_feature = "sb")
}

/// Equivalent to __arm_rsr64("dit").
///
/// # Safety
///
/// Must be called only when FEAT_DIT is implemented.
#[inline]
#[target_feature(enable = "dit")]
unsafe fn rsr64_dit() -> u64 {
    let mut value;
    // SAFETY: called only when FEAT_DIT is implemented
    unsafe {
        asm!("mrs {}, dit", lateout(reg) value, options(nomem, preserves_flags, nostack));
    }
    value
}

/// Equivalent to __arm_wsr64("dit", value).
///
/// # Safety
///
/// Must be called only when FEAT_DIT is implemented.
#[inline]
#[target_feature(enable = "dit")]
unsafe fn wsr64_dit(value: u64) {
    // SAFETY: called only when FEAT_DIT is implemented
    unsafe {
        // The compiler must not cache values or flags across this instruction.
        asm!("msr dit, {}", in(reg) value, options(nostack));
    }
}

/// Equivalent to __arm_wsr64("dit", 1 << 24).
///
/// # Safety
///
/// Must be called only when FEAT_DIT is implemented.
#[inline]
#[target_feature(enable = "dit")]
unsafe fn enable_dit() {
    // SAFETY: called only when FEAT_DIT is implemented
    unsafe {
        // The compiler must not cache values or flags across this instruction.
        asm!("msr dit, #{}", const 1, options(nostack));
    }
}

/// Equivalent to __asm__ __volatile__("sb" ::: "memory").
///
/// # Safety
///
/// Must be called only when FEAT_SB is implemented.
#[inline]
#[target_feature(enable = "sb")]
unsafe fn speculation_barrier() {
    // SAFETY: called only when FEAT_SB is implemented
    unsafe {
        // The compiler must not cache values or flags across this instruction.
        asm!("sb", options(nostack));
    }
}

/// Synchronization barrier for when FEAT_SB is not available.
///
/// This heavier alternative is recommended by Apple when SB is not available, see:
/// https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
#[inline]
fn synchronization_barrier() {
    // SAFETY: these instructions are always available, and have no effects
    // other than being a data barrier and flushing the pipeline
    unsafe {
        // The compiler must not cache values or flags across this instruction.
        asm!("dsb nsh", "isb sy", options(nostack));
    }
}

/// Wraps code with DIT when FEAT_DIT and FEAT_SB were detected.
///
/// # Safety
///
/// FEAT_DIT and FEAT_SB must have been detected.
#[inline]
#[target_feature(enable = "dit,sb")]
unsafe fn with_feat_dit_sb<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    struct Guard {
        dit: u64,
    }

    impl Drop for Guard {
        #[inline]
        fn drop(&mut self) {
            // SAFETY: called only when FEAT_DIT is implemented
            unsafe { wsr64_dit(self.dit) };
        }
    }

    let _guard = Guard {
        // SAFETY: called only when FEAT_DIT is implemented
        dit: unsafe { rsr64_dit() },
    };

    // SAFETY: called only when FEAT_DIT is implemented
    unsafe { enable_dit() };

    // SAFETY: called only when FEAT_SB is implemented
    unsafe { speculation_barrier() };

    f()
}

/// Wraps code with DIT when FEAT_DIT was detected but not FEAT_SB.
///
/// # Safety
///
/// FEAT_DIT must have been detected.
#[inline]
#[target_feature(enable = "dit")]
unsafe fn with_feat_dit<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    struct Guard {
        dit: u64,
    }

    impl Drop for Guard {
        #[inline]
        fn drop(&mut self) {
            // SAFETY: called only when FEAT_DIT is implemented
            unsafe { wsr64_dit(self.dit) };
        }
    }

    let _guard = Guard {
        // SAFETY: called only when FEAT_DIT is implemented
        dit: unsafe { rsr64_dit() },
    };

    // SAFETY: called only when FEAT_DIT is implemented
    unsafe { enable_dit() };

    synchronization_barrier();

    f()
}

/// Runs code with the hardware DIT feature enabled when possible.
#[inline]
pub(crate) fn with_dit<T, F>(f: F) -> T
where
    F: FnOnce() -> T,
{
    // The use of #[target_feature] disables inlining in some cases.
    // Repeating the code three times with different #[target_feature]
    // generates better code.
    if is_feat_dit_implemented() {
        if is_feat_sb_implemented() {
            // SAFETY: both FEAT_DIT and FEAT_SB were detected
            unsafe { with_feat_dit_sb(f) }
        } else {
            // SAFETY: FEAT_DIT was detected
            unsafe { with_feat_dit(f) }
        }
    } else {
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::{is_feat_dit_implemented, rsr64_dit, with_dit};

    #[test]
    fn dit_is_restored_after_with_dit() {
        if is_feat_dit_implemented() {
            // SAFETY: FEAT_DIT was detected
            unsafe {
                let saved = rsr64_dit();
                with_dit(|| assert_ne!(rsr64_dit(), 0));
                assert_eq!(rsr64_dit(), saved);
            }
        }
    }
}
