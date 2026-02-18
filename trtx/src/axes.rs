//! Axis mask type for operations that reduce or normalize over selected axes.
//!
//! TensorRT represents axes as a `u32` bitmask: bit `i` set means axis `i` is included.

use std::fmt;

/// Bitmask of axes: each bit set indicates one axis (bit 0 = axis 0, bit 1 = axis 1, etc.).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Axes(pub u32);

impl fmt::Debug for Axes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bits = self.0;
        if bits == 0 {
            return f.write_str("Axes()");
        }
        let indices: Vec<u32> = (0u32..32).filter(|&i| (bits & (1u32 << i)) != 0).collect();
        f.write_str("Axes(")?;
        for (i, &axis) in indices.iter().enumerate() {
            if i > 0 {
                f.write_str("|")?;
            }
            write!(f, "{}", axis)?;
        }
        f.write_str(")")
    }
}

impl Axes {
    /// No axes selected.
    pub const fn empty() -> Self {
        Axes(0)
    }

    /// Wrap a raw axes bitmask.
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Axes(bits)
    }

    /// Build from a list of axis indices. Each index sets the corresponding bit.
    ///
    /// # Example
    /// ```
    /// # use trtx::Axes;
    /// // Axes 1 and 2 (e.g. for channel normalization over NCHW)
    /// let axes = Axes::new([1, 2]);
    /// assert_eq!(axes.to_bits(), 0b110);
    /// ```
    #[inline]
    pub const fn new<const N: usize>(indices: [u32; N]) -> Self {
        let mut bits = 0u32;
        let mut i = 0;
        while i < N {
            bits |= 1u32.wrapping_shl(indices[i]);
            i += 1;
        }
        Axes(bits)
    }

    /// Build from a list of axis indices. Each index sets the corresponding bit.
    ///
    /// # Example
    /// ```
    /// # use trtx::Axes;
    /// // Axes 1 and 2 (e.g. for channel normalization over NCHW)
    /// let axes = Axes::from_slice(&[1, 2]);
    /// assert_eq!(axes.to_bits(), 0b110);
    /// ```
    pub fn from_slice(indices: &[u32]) -> Self {
        let mut bits = 0u32;
        for &index in indices {
            bits |= 1u32.wrapping_shl(index);
        }
        Axes(bits)
    }

    /// Add one axis by index. Chainable for const construction.
    #[inline]
    pub const fn with_axis(self, axis: u32) -> Self {
        Axes(self.0 | 1u32.wrapping_shl(axis))
    }

    /// Return the raw bitmask.
    #[inline]
    pub const fn to_bits(self) -> u32 {
        self.0
    }
}

impl From<u32> for Axes {
    #[inline]
    fn from(bits: u32) -> Self {
        Axes(bits)
    }
}

impl From<Axes> for u32 {
    #[inline]
    fn from(axes: Axes) -> u32 {
        axes.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_zero() {
        assert_eq!(Axes::empty().to_bits(), 0);
        assert_eq!(Axes::default().to_bits(), 0);
    }

    #[test]
    fn from_bits_roundtrip() {
        assert_eq!(Axes::from_bits(0).to_bits(), 0);
        assert_eq!(Axes::from_bits(0b111).to_bits(), 0b111);
        assert_eq!(Axes::from_bits(0xFFFF_FFFF).to_bits(), 0xFFFF_FFFF);
    }

    #[test]
    fn from_indices_empty() {
        let axes = Axes::new([]);
        assert_eq!(axes.to_bits(), 0);
    }

    #[test]
    fn from_indices_single() {
        assert_eq!(Axes::new([0]).to_bits(), 1);
        assert_eq!(Axes::new([3]).to_bits(), 1 << 3);
        assert_eq!(Axes::new([31]).to_bits(), 1 << 31);
    }

    #[test]
    fn from_indices_multiple() {
        let axes = Axes::new([1, 2]);
        assert_eq!(axes.to_bits(), 0b110);
        let axes = Axes::new([0, 2, 4]);
        assert_eq!(axes.to_bits(), 0b10101);
    }

    #[test]
    fn from_indices_duplicate_same_as_unique() {
        let a = Axes::new([1, 1, 2]);
        let b = Axes::new([1, 2]);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn with_axis_builder() {
        let axes = Axes::empty().with_axis(0).with_axis(2);
        assert_eq!(axes.to_bits(), 0b101);
        let axes = Axes::from_bits(1).with_axis(1);
        assert_eq!(axes.to_bits(), 0b11);
    }

    #[test]
    fn from_u32_into_u32() {
        let raw: u32 = 0b10110;
        let axes: Axes = raw.into();
        assert_eq!(axes.to_bits(), raw);
        let back: u32 = axes.into();
        assert_eq!(back, raw);
    }

    #[test]
    fn eq_and_clone() {
        let a = Axes::new([1, 2]);
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(b.to_bits(), 0b110);
    }

    #[test]
    fn debug_shows_indices_separated_by_pipe() {
        assert_eq!(format!("{:?}", Axes::empty()), "Axes()");
        assert_eq!(format!("{:?}", Axes::new([0])), "Axes(0)");
        assert_eq!(format!("{:?}", Axes::new([1, 2])), "Axes(1|2)");
        assert_eq!(format!("{:?}", Axes::new([0, 2, 4])), "Axes(0|2|4)");
    }
}
