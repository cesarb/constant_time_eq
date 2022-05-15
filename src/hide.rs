#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn optimizer_hide(mut value: u8) -> u8 {
    // SAFETY: the input value is passed unchanged to the output, the inline assembly does nothing.
    unsafe {
        core::arch::asm!("/* {0} */", inout(reg_byte) value, options(pure, nomem, nostack, preserves_flags));
        value
    }
}

#[cfg(any(
    target_arch = "arm",
    target_arch = "aarch64",
    target_arch = "riscv32",
    target_arch = "riscv64"
))]
#[allow(asm_sub_register)]
#[inline]
pub(crate) fn optimizer_hide(mut value: u8) -> u8 {
    // SAFETY: the input value is passed unchanged to the output, the inline assembly does nothing.
    unsafe {
        core::arch::asm!("/* {0} */", inout(reg) value, options(pure, nomem, nostack, preserves_flags));
        value
    }
}

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "arm",
    target_arch = "aarch64",
    target_arch = "riscv32",
    target_arch = "riscv64"
)))]
#[inline(never)] // This function is non-inline to prevent the optimizer from looking inside it.
pub(crate) fn optimizer_hide(value: u8) -> u8 {
    // SAFETY: the result of casting a reference to a pointer is valid; the type is Copy.
    unsafe { core::ptr::read_volatile(&value) }
}
