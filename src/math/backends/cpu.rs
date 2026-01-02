fn compute_strides(shape: &[usize], trans: &[usize]) -> Vec<usize>{
    assert!(shape.len() == trans.len(),"shape and transposition have different ranks");
    let mut strides: Vec<usize> = vec![0; shape.len()];
    let mut sorted_axes: Vec<usize> = (0..shape.len()).collect();
    sorted_axes.sort_by_key(|k| trans[*k]);

    let mut stride = 1;
    for i in sorted_axes {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides
}

fn flatten_coords(coords: &[usize], strides: &[usize]) -> usize {
    assert!(coords.len() == strides.len(), "coords and strides have different ranks");
    coords.iter().zip(strides).map(|(&c,&s)| c*s).sum()
}

fn build_coords(mut idx: usize, strides: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(strides.len());

    for &s in strides{
        coords.push(idx / s);
        idx %= s;
    }

    coords
}

pub fn transpose(old: &[f32], cur: &[usize], new: &[usize], shape: &[usize]) -> Vec<f32>{
    assert!(cur.len() == new.len() && cur.len() == shape.len(), "current and new shape is not the same");

    let mut newdata: Vec<f32> = vec![0.; old.len()];

    let src_strides = compute_strides(shape, cur);
    let dst_strides = compute_strides(shape, new);

    for (i,d) in newdata.iter_mut().enumerate() {
        *d = old[flatten_coords(&build_coords(i, &dst_strides), &src_strides)];
    }

    return newdata;
}


pub fn add_inplace(this_data: &mut [f32], other_data: &[f32], this_trans: &[usize], other_trans: &[usize], this_shape: &[usize], other_shape: &[usize]){
    assert!(this_shape == other_shape, "current and new shape is not the same");

    let src_strides = compute_strides(this_shape, this_trans);
    let dst_strides = compute_strides(other_shape, other_trans);

    for (i,d) in this_data.iter_mut().enumerate() {
        *d += other_data[flatten_coords(&build_coords(i, &dst_strides), &src_strides)];
    }
}