use super::backends::*;

#[derive(Clone)]
pub struct Tensor {
    rank: usize,
    dim: Vec<usize>,
    data: Vec<f32>,
    trans: Vec<usize>
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

impl Tensor {
    pub fn zeroes(rank: usize, dim: &[usize]) -> Self {
        Tensor { 
            rank, 
            dim: Vec::from(dim), 
            data: vec![0.; get_vec_size(dim.as_ref()) as usize],
            trans: get_default_transposition(rank)
        }
    }

    pub fn values(rank: usize, dim: &[usize], val: &[f32]) -> Self {
        Tensor { 
            rank, 
            dim: Vec::from(dim), 
            data: Vec::from(val),
            trans: get_default_transposition(rank)
        }
    }

    pub fn transpose(&self,trans: &[usize])  -> Self {
        let mut toret = self.clone();
        toret.transpose_inplace(trans);
        return toret;
    }

    pub fn transpose_inplace(&mut self,trans: &[usize]){
        self.data = cpu::transpose(&self.data, &self.trans, trans, &self.dim);
    }

    pub fn add_inplace(&mut self, other: &mut Self) {
        cpu::add_inplace(&mut self.data, &other.data, &self.trans, &other.trans, &self.dim, &other.dim);
    }

    pub fn as_row_major(&self) -> Vec<f32> {
        if self.trans[0] == 1 {
            return self.data.clone();
        } else {
            return self.transpose(&get_default_transposition(self.rank)).data.clone();
        }
    }

    pub fn get_data_clone(&self) -> (Vec<f32>,Vec<usize>) {
        (self.data.clone(), self.trans.clone())
    }

    pub fn get_data_ref(&self) -> (&Vec<f32>,&Vec<usize>) {
        (&self.data, &self.trans)
    }
}