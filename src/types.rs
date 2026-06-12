use num::Zero;
use num_traits::AsPrimitive;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{AddAssign, Div, Mul, Sub, SubAssign};

pub trait Float:
  Zero
  + Copy
  + AddAssign
  + FromIndex
  + Div<Output = Self>
  + Sub<Output = Self>
  + SubAssign
  + Mul<Output = Self>
  + PartialOrd
  + num_traits::Float
  + Sum<Self>
  + Display
{
}

impl<T> Float for T where
  T: Zero
    + Copy
    + AddAssign
    + FromIndex
    + Div<Output = Self>
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + PartialOrd
    + num_traits::Float
    + Sum<Self>
    + Display
{
}

pub trait FromIndex {
  fn from_index(i: usize) -> Self;
}
impl<T: Copy + 'static> FromIndex for T
where
  usize: AsPrimitive<T>,
{
  fn from_index(i: usize) -> T {
    i.as_()
  }
}
