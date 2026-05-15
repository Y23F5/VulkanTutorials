/** @file VulkanSDKCompat.h
 * Compatibility shim for VulkanSDK 1.4.328+ which removed several types
 * and enum values. Uses a preprocessor rename trick to inject the missing
 * eDescriptorHeapEXT into the PipelineCreateFlagBits2 enum wrapper.
 *
 * DO NOT include vulkan.hpp before this header.
 */
#pragma once

#define PipelineCreateFlagBits2 PipelineCreateFlagBits2_Renamed
#define PipelineCreateFlagBits2KHR PipelineCreateFlagBits2KHR_Renamed
#include "vulkan/vulkan.hpp"
#undef PipelineCreateFlagBits2KHR
#undef PipelineCreateFlagBits2

namespace vk {
	using PipelineCreateFlagBits2KHR = PipelineCreateFlagBits2_Renamed;
	using PipelineCreateFlags2KHR   = PipelineCreateFlags2;

	namespace PipelineCreateFlagBits2 {
		using enum PipelineCreateFlagBits2_Renamed;
		constexpr PipelineCreateFlagBits2_Renamed eDescriptorHeapEXT =
			static_cast<PipelineCreateFlagBits2_Renamed>(0);
	}

	struct ShaderDescriptorSetAndBindingMappingInfoEXT {};
}
