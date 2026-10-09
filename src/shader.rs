use include_bytes_plus::include_bytes;

pub const ADD_ONE_SHADER_SPV: &[u32] = include_bytes!("data/shader/add_one.spv" as u32).as_slice();
pub const SUM_ARRAY_SHADER_SPV: &[u32] = include_bytes!("data/shader/sum_array.spv" as u32).as_slice();
