mod vk_facade;

use vk_facade::VkContext;

pub fn init(app_name: &str, app_version: u32) -> VkContext {
    match VkContext::new(app_name, app_version) {
        Ok(vk_ctx) => vk_ctx,
        Err(err) => panic!("Context creation failed {err:?}"),
    }
}
