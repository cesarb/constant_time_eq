// This function is non-inline to prevent the optimizer from looking inside it.
#[inline(never)]
fn constant_time_ne(a: &[u8], b: &[u8]) -> u8 {
	// Compares the sizes here so the optimizer knows a.len() == b.len()
	if a.len() != b.len() { return !0; }

	let mut tmp = 0;
	for i in 0..a.len() {
		tmp |= a[i] ^ b[i];
		// Ideally, should use an inline asm here:
		// asm!("" : "=r" (tmp) : "0" (tmp));
		// But asm! is not stable yet.
		// With the inline asm, this function could be inlined
		// since the asm would prevent the optimizer from doing
		// an early return.
	}
	tmp	// The compare with 0 must happen outside this function.
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
///
/// // Not equal-sized, so won't take constant time.
/// assert!(!constant_time_eq(b"foo", b""));
/// assert!(!constant_time_eq(b"foo", b"quux"));
/// ```
#[inline]
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
	constant_time_ne(a, b) == 0
}
