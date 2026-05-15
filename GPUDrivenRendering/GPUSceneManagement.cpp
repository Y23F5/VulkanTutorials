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
#include "WFCGenerator.h"
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

void GPUSceneManagement::GenerateScene() {
	WFCGenerator gen;
	WFCConfig wfcCfg;
	wfcCfg.gridSize = m_benchConfig.gridSize;
	wfcCfg.seed     = m_benchConfig.seed;

	switch (m_benchConfig.density) {
		case 20: wfcCfg.emptyWeight = 8.0f; wfcCfg.otherWeight = 2.0f; break;
		case 50: wfcCfg.emptyWeight = 5.0f; wfcCfg.otherWeight = 5.0f; break;
		case 80: wfcCfg.emptyWeight = 2.0f; wfcCfg.otherWeight = 10.0f; break;
	}

	m_tileGrid = gen.Generate(wfcCfg);
	auto instances = gen.TileGridToInstances(m_tileGrid, wfcCfg.gridSize, 2.0f);

	const uint32_t chunkDim = m_benchConfig.chunkSize;
	m_gridChunks = wfcCfg.gridSize / chunkDim;

	std::vector<std::vector<WFCInstance>> buckets(m_gridChunks * m_gridChunks);
	float cellSize = 2.0f;
	for (const auto& inst : instances) {
		uint32_t cx = std::min((uint32_t)(inst.posX / (chunkDim * cellSize)), m_gridChunks - 1);
		uint32_t cy = std::min((uint32_t)(inst.posZ / (chunkDim * cellSize)), m_gridChunks - 1);
		buckets[cy * m_gridChunks + cx].push_back(inst);
	}

	std::vector<WFCInstance> sorted;
	sorted.reserve(instances.size());
	m_chunks.clear();
	m_chunks.reserve(m_gridChunks * m_gridChunks);

	for (uint32_t cy = 0; cy < m_gridChunks; ++cy) {
		for (uint32_t cx = 0; cx < m_gridChunks; ++cx) {
			auto& bucket = buckets[cy * m_gridChunks + cx];
			auto mid = std::stable_partition(bucket.begin(), bucket.end(),
				[](const WFCInstance& i) { return i.isCube; });

			ChunkInfo chunk = {};
			chunk.gridX          = cx;
			chunk.gridY          = cy;
			chunk.instanceOffset = (uint32_t)sorted.size();
			chunk.instanceCount  = (uint32_t)bucket.size();
			chunk.cubeCount      = (uint32_t)std::distance(bucket.begin(), mid);
			chunk.sphereCount    = (uint32_t)std::distance(mid, bucket.end());

			sorted.insert(sorted.end(), bucket.begin(), bucket.end());
			m_chunks.push_back(chunk);
		}
	}
	m_chunkVisible.assign(m_chunks.size(), true);

	m_totalInstances = (uint32_t)sorted.size();
	m_cullingData.resize(m_totalInstances);
	m_renderData.resize(m_totalInstances);
	for (uint32_t i = 0; i < m_totalInstances; ++i) {
		const auto& inst = sorted[i];
		m_cullingData[i] = { inst.posX, inst.posY, inst.posZ,
			inst.scaleX * 0.5f, inst.scaleY * 0.5f, inst.scaleZ * 0.5f };
		m_renderData[i]  = { inst.r, inst.g, inst.b, 1.0f };
	}

	ComputeChunkAABBs();
}

void GPUSceneManagement::ComputeChunkAABBs() {
	for (auto& chunk : m_chunks) {
		float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
		float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;
		for (uint32_t i = 0; i < chunk.instanceCount; ++i) {
			const auto& c = m_cullingData[chunk.instanceOffset + i];
			float cx = c.centerX, cy = c.centerY, cz = c.centerZ;
			float hx = c.halfX, hy = c.halfY, hz = c.halfZ;
			minX = std::min(minX, cx - hx); maxX = std::max(maxX, cx + hx);
			minY = std::min(minY, cy - hy); maxY = std::max(maxY, cy + hy);
			minZ = std::min(minZ, cz - hz); maxZ = std::max(maxZ, cz + hz);
		}
		chunk.aabbMinX = minX; chunk.aabbMinY = minY; chunk.aabbMinZ = minZ;
		chunk.aabbMaxX = maxX; chunk.aabbMaxY = maxY; chunk.aabbMaxZ = maxZ;
	}
}

void GPUSceneManagement::CreateBuffers() {
	FrameContext const& ctx = m_renderer->GetFrameContext();

	m_cullingBuffer = m_memoryManager->CreateBuffer(
		{ sizeof(CullingDatum) * m_totalInstances, vk::BufferUsageFlagBits::eStorageBuffer },
		vk::MemoryPropertyFlagBits::eDeviceLocal, "CullingData");

	m_renderBuffer = m_memoryManager->CreateBuffer(
		{ sizeof(RenderDatum) * m_totalInstances, vk::BufferUsageFlagBits::eStorageBuffer },
		vk::MemoryPropertyFlagBits::eDeviceLocal, "RenderData");

	uint32_t indirectSize = m_gridChunks * m_gridChunks * 2 * 5 * sizeof(uint32_t);
	m_indirectBuffer = m_memoryManager->CreateBuffer(
		{ indirectSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer },
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		"IndirectBuffer");

	m_chunkBuffer = m_memoryManager->CreateBuffer(
		{ sizeof(ChunkInfo) * m_chunks.size(), vk::BufferUsageFlagBits::eStorageBuffer },
		vk::MemoryPropertyFlagBits::eDeviceLocal, "ChunkBuffer");

	WriteInstanceData();
}

void GPUSceneManagement::WriteInstanceData() {
	FrameContext const& ctx = m_renderer->GetFrameContext();

	auto stagingCull = m_memoryManager->CreateBuffer(
		{ sizeof(CullingDatum) * m_totalInstances, vk::BufferUsageFlagBits::eTransferSrc },
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, "StgCull");
	memcpy(stagingCull.Map<CullingDatum>(), m_cullingData.data(), sizeof(CullingDatum) * m_totalInstances);
	stagingCull.Unmap();

	auto stagingRender = m_memoryManager->CreateBuffer(
		{ sizeof(RenderDatum) * m_totalInstances, vk::BufferUsageFlagBits::eTransferSrc },
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, "StgRender");
	memcpy(stagingRender.Map<RenderDatum>(), m_renderData.data(), sizeof(RenderDatum) * m_totalInstances);
	stagingRender.Unmap();

	auto stagingChunk = m_memoryManager->CreateBuffer(
		{ sizeof(ChunkInfo) * m_chunks.size(), vk::BufferUsageFlagBits::eTransferSrc },
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, "StgChunk");
	memcpy(stagingChunk.Map<ChunkInfo>(), m_chunks.data(), sizeof(ChunkInfo) * m_chunks.size());
	stagingChunk.Unmap();

	vk::UniqueCommandBuffer cmd = CmdBufferCreateBegin(ctx.device, ctx.commandPools[CommandType::Graphics], "Upload");
	cmd->copyBuffer(stagingCull.buffer, m_cullingBuffer.buffer, 1,
		&vk::BufferCopy{0, 0, sizeof(CullingDatum) * m_totalInstances});
	cmd->copyBuffer(stagingRender.buffer, m_renderBuffer.buffer, 1,
		&vk::BufferCopy{0, 0, sizeof(RenderDatum) * m_totalInstances});
	cmd->copyBuffer(stagingChunk.buffer, m_chunkBuffer.buffer, 1,
		&vk::BufferCopy{0, 0, sizeof(ChunkInfo) * m_chunks.size()});
	CmdBufferEndSubmitWait(*cmd, ctx.device, ctx.queues[CommandType::Graphics]);

	m_memoryManager->DiscardBuffer(stagingCull, DiscardMode::Immediate);
	m_memoryManager->DiscardBuffer(stagingRender, DiscardMode::Immediate);
	m_memoryManager->DiscardBuffer(stagingChunk, DiscardMode::Immediate);
}

void GPUSceneManagement::CreatePipelines() {
	FrameContext const& ctx = m_renderer->GetFrameContext();

	m_sceneLayout = DescriptorSetLayoutBuilder(ctx.device)
		.WithUniformBuffers(0, 1, vk::ShaderStageFlagBits::eVertex)
		.WithStorageBuffers(1, 1, vk::ShaderStageFlagBits::eVertex)
		.WithStorageBuffers(2, 1, vk::ShaderStageFlagBits::eVertex)
		.Build("Scene Data");

	m_graphicsPipeline = PipelineBuilder(ctx.device)
		.WithVertexInputState(m_cubeMesh->GetVertexInputState())
		.WithTopology(vk::PrimitiveTopology::eTriangleList)
		.WithColourAttachment(ctx.colourFormat)
		.WithDepthAttachment(ctx.depthFormat)
		.WithDescriptorSetLayout(0, *m_sceneLayout)
		.WithShaderBinary("Scene.vert.spv", vk::ShaderStageFlagBits::eVertex)
		.WithShaderBinary("Scene.frag.spv", vk::ShaderStageFlagBits::eFragment)
		.Build("Scene Pipeline");

	m_computeLayout = DescriptorSetLayoutBuilder(ctx.device)
		.WithStorageBuffers(0, 1, vk::ShaderStageFlagBits::eCompute)
		.WithStorageBuffers(1, 1, vk::ShaderStageFlagBits::eCompute)
		.Build("Compute Layout");

	m_computePipeline = ComputePipelineBuilder(ctx.device)
		.WithDescriptorSetLayout(0, *m_computeLayout)
		.WithShaderBinary("Culling.comp.spv")
		.Build("GPU Culling");
}

void GPUSceneManagement::CreateDescriptorSets() {
	FrameContext const& ctx = m_renderer->GetFrameContext();

	m_sceneDescriptor = CreateDescriptorSet(ctx.device, ctx.descriptorPool, *m_sceneLayout);
	WriteBufferDescriptor(ctx.device, *m_sceneDescriptor, 0, vk::DescriptorType::eUniformBuffer, m_cameraBuffer);
	WriteBufferDescriptor(ctx.device, *m_sceneDescriptor, 1, vk::DescriptorType::eStorageBuffer, m_cullingBuffer);
	WriteBufferDescriptor(ctx.device, *m_sceneDescriptor, 2, vk::DescriptorType::eStorageBuffer, m_renderBuffer);

	m_computeDescriptor = CreateDescriptorSet(ctx.device, ctx.descriptorPool, *m_computeLayout);
	WriteBufferDescriptor(ctx.device, *m_computeDescriptor, 0, vk::DescriptorType::eStorageBuffer, m_chunkBuffer);
	WriteBufferDescriptor(ctx.device, *m_computeDescriptor, 1, vk::DescriptorType::eStorageBuffer, m_indirectBuffer);
}

