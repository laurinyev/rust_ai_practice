use std::vec;

use crate::math::tensor::*;

#[test]
fn tensor_zeroes() {
    let t = Tensor::zeroes(2, &[2,2]);
    assert_eq!(t.as_row_major(),vec![0.,0.,0.,0.])
}

#[test]
fn tensor_values() {
    let t = Tensor::values(2, &[2,2],&[1.,2.,3.,4.]);
    assert_eq!(t.as_row_major(),vec![1.,2.,3.,4.])
}

#[test]
fn tensor_transpose() {
    let mut t = Tensor::values(2, &[2,2],&[1.,2.,3.,4.]);
    t.transpose_inplace(&[2,1]);
    assert_eq!(t.as_row_major(),vec![1.,3.,2.,4.])
}