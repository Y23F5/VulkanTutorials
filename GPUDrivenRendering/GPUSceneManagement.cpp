/** @file GPUSceneManagement.cpp
 * Implements three GPU-driven rendering scheme comparisons.
 * Uses split-SSBO: CullingData (24B) + RenderData (16B) per instance.
 *
 * Scheme 1: CPU traverses visible chunks, per-chunk vkCmdDrawIndexed.
 * Scheme 2: CPU frustum-culls chunks, fills indirect buffer, 2 indirect draws.
 * Scheme 3: Compute shader frustum-culls chunks, fills indirect buffer, 2 indirect draws.
 *
 * Measurement: QPC (CPU) + vkCmdWriteTimestamp2 (GPU), CSV output.
 * Two modes: interactive (window + optional ImGui monitor) and headless (-Benchmark).
 */
#include "GPUSceneManagement.h"
#include "VulkanVMAMemoryManager.h"
#include "MshLoader.h"

using namespace NCL;
using namespace Rendering;
using namespace Vulkan;

GPUSceneManagement::GPUSceneManagement(Window& window, VulkanInitialisation& vkInit)
	: m_hostWindow(window), m_controller(*window.GetKeyboard(), *window.GetMouse())
	, m_vkInit(vkInit), m_totalInstances(0), m_gridChunks(0)
	, m_cubeIndexCount(0), m_sphereIndexCount(0)
	, m_isRecording(false), m_benchmarkComplete(false), m_currentFrame(0)
	, m_monitor(nullptr), m_offscreenColour(nullptr), m_offscreenDepth(nullptr) {

	Initialise();
	CreatePipelines();
}

GPUSceneManagement::~GPUSceneManagement() {
	m_renderer->GetDevice().waitIdle();
	m_memoryManager->DiscardBuffer(m_cameraBuffer, DiscardMode::Immediate);
	delete m_memoryManager;
	delete m_renderer;
	if (m_monitor) delete m_monitor;
	if (m_offscreenColour) delete m_offscreenColour;
	if (m_offscreenDepth) delete m_offscreenDepth;
}

void GPUSceneManagement::Finish() {
	m_renderer->GetDevice().waitIdle();
}

void GPUSceneManagement::Initialise() {
	m_renderer      = new VulkanRenderer(m_hostWindow, m_vkInit);
	m_memoryManager = new VulkanVMAMemoryManager(m_renderer->GetDevice(),
		m_renderer->GetPhysicalDevice(), m_renderer->GetVulkanInstance(), m_vkInit);
	BuildCamera();

	QueryPerformanceFrequency(&m_qpcFrequency);
	m_cpuTimestampPeriod = 1'000'000'000 / m_qpcFrequency.QuadPart;

	FrameContext const& ctx = m_renderer->GetFrameContext();
	m_defaultSampler = ctx.device.createSamplerUnique(vk::SamplerCreateInfo()
		.setMinFilter(vk::Filter::eLinear)
		.setMagFilter(vk::Filter::eLinear));

	m_cameraLayout = DescriptorSetLayoutBuilder(ctx.device)
		.WithUniformBuffers(0, 1, vk::ShaderStageFlagBits::eVertex)
		.Build("CameraMatrices");
	m_cameraDescriptor = CreateDescriptorSet(ctx.device, ctx.descriptorPool, *m_cameraLayout);
	WriteBufferDescriptor(ctx.device, *m_cameraDescriptor, 0, vk::DescriptorType::eUniformBuffer, m_cameraBuffer);

	m_cubeMesh   = LoadMesh("Cube.msh");
	m_sphereMesh = LoadMesh("Sphere.msh");
	m_cubeIndexCount   = m_cubeMesh->GetIndexCount();
	m_sphereIndexCount = m_sphereMesh->GetIndexCount();

	vk::PhysicalDeviceProperties props = m_renderer->GetPhysicalDevice().getProperties();
	m_gpuTimestampPeriod = props.limits.timestampPeriod;

	vk::QueryPoolCreateInfo qpCreate(vk::QueryType::eTimestamp, 4);
	m_queryPool = ctx.device.createQueryPoolUnique(qpCreate);

	m_camera.SetFieldOfVision(45.0f).SetNearPlane(0.1f).SetFarPlane(2000.0f);
	m_camera.SetPosition(Vector3(128, 160, -64));
	m_camera.SetPitch(-55.0f).SetYaw(0.0f);
	m_camera.SetController(m_controller);

	m_controller.MapAxis(0, "Sidestep");
	m_controller.MapAxis(1, "UpDown");
	m_controller.MapAxis(2, "Forward");
	m_controller.MapAxis(3, "XLook");
	m_controller.MapAxis(4, "YLook");
}

UniqueVulkanMesh GPUSceneManagement::LoadMesh(const string& filename) {
	VulkanMesh* newMesh = new VulkanMesh();
	MshLoader::LoadMesh(filename, *newMesh);
	UploadMeshWait(*newMesh);
	return UniqueVulkanMesh(newMesh);
}

void GPUSceneManagement::UploadMeshWait(VulkanMesh& m) {
	FrameContext const& ctx = m_renderer->GetFrameContext();
	vk::UniqueCommandBuffer cmdBuffer = CmdBufferCreateBegin(ctx.device,
		ctx.commandPools[CommandType::Graphics], "Mesh Upload");
	m.UploadToGPU(*cmdBuffer, m_memoryManager);
	CmdBufferEndSubmitWait(*cmdBuffer, ctx.device, ctx.queues[CommandType::Graphics]);
}

void GPUSceneManagement::BuildCamera() {
	m_cameraBuffer = m_memoryManager->CreateBuffer(
		{ .size = sizeof(Matrix4) * 2,
		  .usage = vk::BufferUsageFlagBits::eUniformBuffer },
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		"Camera Buffer");
}

void GPUSceneManagement::UploadCameraUniform() {
	Matrix4* cameraMatrices = m_cameraBuffer.Map<Matrix4>();
	cameraMatrices[0] = m_camera.BuildViewMatrix();
	cameraMatrices[1] = m_camera.BuildProjectionMatrix(m_hostWindow.GetScreenAspect());
	m_cameraBuffer.Unmap();
}

void GPUSceneManagement::SetBenchmarkConfig(const BenchmarkConfig& config) {
	m_benchConfig = config;

	m_vkInit.autoBeginDynamicRendering = false;

	GenerateScene();
	CreateBuffers();
	CreateDescriptorSets();

	float sceneHalf = m_benchConfig.gridSize * 2.0f * 0.5f;
	m_camera.SetPosition(Vector3(sceneHalf, sceneHalf * 1.2f, -sceneHalf * 0.5f));
	m_camera.SetPitch(-55.0f).SetYaw(0.0f);
}

void GPUSceneManagement::RunFrame(float dt) {
	if (m_hostWindow.IsMinimised()) return;

	m_renderer->BeginFrame();
	m_controller.Update(dt);
	m_camera.UpdateCamera(dt);
	UploadCameraUniform();
	m_memoryManager->Update();

	switch (m_benchConfig.scheme) {
		case RenderScheme::CPU_Instanced:    RenderScheme1(dt); break;
		case RenderScheme::CPU_CullIndirect: RenderScheme2(dt); break;
		case RenderScheme::GPU_CullIndirect: RenderScheme3(dt); break;
	}

	m_renderer->EndFrame();
	m_renderer->SwapBuffers();
	m_currentFrame++;
}

