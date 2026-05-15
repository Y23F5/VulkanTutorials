/** @file VulkanSDKCompat.h
 * Compatibility shim for VulkanSDK 1.4.328+ which removed
 * ShaderDescriptorSetAndBindingMappingInfoEXT.
 * Defines an empty stub so submodule PipelineBuilderBase compiles.
 */
#pragma once
#include "vulkan/vulkan.hpp"

namespace vk {
	struct ShaderDescriptorSetAndBindingMappingInfoEXT {};
}
