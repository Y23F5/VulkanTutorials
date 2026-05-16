/** @file GuiWrapper.h
 * Minimal ImGui wrapper compatible with docking branch API.
 */
#pragma once
#ifdef USE_IMGUI
#include <vector>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <chrono>
#include <functional>
#include <string>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "vulkan/vulkan.hpp"
#include "imgui.h"
#include "VulkanRenderer.h"
#include "VulkanUtils.h"

namespace NCL::Rendering::Vulkan {

class GuiWrapper {
public:
	void Init(HWND window, VulkanRenderer* renderer);
	void StartNewFrame();
	void SyncInput(HWND hwnd);
	void Render(vk::CommandBuffer cmdBuffer);
	void Destroy();

private:
	VulkanRenderer* m_renderer = nullptr;
};

}
#endif
