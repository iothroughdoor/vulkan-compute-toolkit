pub struct ComputeShader {
    pub local_size_x: u32,
    pub local_size_y: u32,
    pub local_size_z: u32,
    pub(crate) spirv: Vec<u32>,
}

pub struct ComputeShaderBuilder {
    comp_shader: ComputeShader
}

impl ComputeShader {
    pub fn builder() -> ComputeShaderBuilder {
        ComputeShaderBuilder {
            comp_shader: ComputeShader {
                local_size_x: 1,
                local_size_y: 1,
                local_size_z: 1,
                spirv: vec![],
            }
        }
    }
}

impl ComputeShaderBuilder {
    pub fn local_size_x(mut self, sz: u32) -> ComputeShaderBuilder {
        self.comp_shader.local_size_x = sz;
        self
    }

    pub fn local_size_y(mut self, sz: u32) -> ComputeShaderBuilder {
        self.comp_shader.local_size_y = sz;
        self
    }

    pub fn local_size_z(mut self, sz: u32) -> ComputeShaderBuilder {
        self.comp_shader.local_size_z = sz;
        self
    }

    pub fn spirv(mut self, byte_code: Vec<u32>) -> ComputeShaderBuilder {
        self.comp_shader.spirv = byte_code;
        self
    }

    pub fn build(self) -> ComputeShader {
        self.comp_shader
    }
}
