
#[derive(Clone)]
pub struct Tensor {
    rank: usize,
    dim: Vec<usize>,
    data: Vec<f32>,
    transposition: Vec<usize>
}

fn get_vec_size(dim: &[usize]) -> usize {
    let mut amt = 0;
    for v in dim.iter() {
        amt += v;
    }
    return amt;
}

fn get_default_transposition(rank: usize) -> Vec<usize> {
    (1..=rank).collect::<Vec<usize>>()
}

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


impl Tensor {
    pub fn zeroes(rank: usize, dim: &[usize]) -> Self {
        Tensor { 
            rank, 
            dim: Vec::from(dim), 
            data: vec![0.; get_vec_size(dim.as_ref()) as usize],
            transposition: get_default_transposition(rank)
        }
    }

    pub fn values(rank: usize, dim: &[usize], val: &[f32]) -> Self {
        Tensor { 
            rank, 
            dim: Vec::from(dim), 
            data: Vec::from(val),
            transposition: get_default_transposition(rank)
        }
    }

    pub fn transpose(&self,transposition: &[usize])  -> Self {
        let mut toret = self.clone();
        toret.transpose_inplace(transposition);
        return toret;
    }

    pub fn transpose_inplace(&mut self,transposition: &[usize]){
        let old_data = self.data.clone();

        let src_strides = compute_strides(&self.dim, &self.transposition);
        let dst_strides = compute_strides(&self.dim, transposition);

        for (i,d) in self.data.iter_mut().enumerate() {
            *d = old_data[flatten_coords(&build_coords(i, &dst_strides), &src_strides)];
        }
    }

    pub fn as_row_major(&self) -> Vec<f32> {
        if self.transposition[0] == 1 {
            return self.data.clone();
        } else {
            todo!("as_row_major for non-row-major tensors")
        }
    }

    pub fn get_data_clone(&self) -> (Vec<f32>,Vec<usize>) {
        (self.data.clone(), self.transposition.clone())
    }

    pub fn get_data_ref(&self) -> (&Vec<f32>,&Vec<usize>) {
        (&self.data, &self.transposition)
    }
}