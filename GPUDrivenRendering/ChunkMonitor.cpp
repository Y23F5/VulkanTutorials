/** @file ChunkMonitor.cpp
 * Renders NxN colored matrix where each cell represents a spatial chunk.
 * Green = visible/populated, Red = culled/populated, Dark violet = empty.
 */
#include "ChunkMonitor.h"
#include "imgui.h"

ChunkMonitor::ChunkMonitor(uint32_t gridChunks, uint32_t chunkSize, uint32_t gridSize)
	: m_gridChunks(gridChunks), m_chunkSize(chunkSize), m_gridSize(gridSize)
	, m_cells(gridChunks * gridChunks), m_panelOpen(true) {
}

ChunkMonitor::~ChunkMonitor() {}

void ChunkMonitor::Update(const ChunkMonitorCell* cells, uint32_t& visibleCount, uint32_t& totalInstances) {
	visibleCount = 0;
	totalInstances = 0;
	for (uint32_t i = 0; i < m_gridChunks * m_gridChunks; ++i) {
		m_cells[i] = cells[i];
		if (cells[i].visible && cells[i].instanceCount > 0) {
			visibleCount++;
			totalInstances += cells[i].instanceCount;
		}
	}
}

void ChunkMonitor::Render() {
	if (!m_panelOpen) return;

	ImGui::Begin("Chunk Culling Monitor", &m_panelOpen);

	uint32_t total = m_gridChunks * m_gridChunks;
	uint32_t visible = 0, totalInst = 0;
	for (const auto& c : m_cells) {
		if (c.visible && c.instanceCount > 0) { visible++; totalInst += c.instanceCount; }
	}
	ImGui::Text("Grid: %dx%d | Chunk: %d^2 | Visible: %d/%d | Instances: %d",
		m_gridSize, m_gridSize, m_chunkSize, visible, total, totalInst);

	ImDrawList* drawList = ImGui::GetWindowDrawList();
	ImVec2 cursor = ImGui::GetCursorScreenPos();
	float cellPx = ImGui::GetContentRegionAvail().x / m_gridChunks;
	cellPx = std::min(cellPx, 24.0f);
	float totalW = cellPx * m_gridChunks;

	for (uint32_t y = 0; y < m_gridChunks; ++y) {
		for (uint32_t x = 0; x < m_gridChunks; ++x) {
			const auto& cell = m_cells[y * m_gridChunks + x];
			ImU32 color;
			if (cell.instanceCount == 0) {
				color = IM_COL32(40, 30, 40, 80);
			} else if (cell.visible) {
				color = IM_COL32(0, 160, 0, 140);
			} else {
				color = IM_COL32(160, 0, 0, 100);
			}

			ImVec2 tl(cursor.x + x * cellPx, cursor.y + y * cellPx);
			ImVec2 br(tl.x + cellPx - 1, tl.y + cellPx - 1);
			drawList->AddRectFilled(tl, br, color);

			if (cell.instanceCount > 0 && cellPx > 12) {
				char buf[8];
				snprintf(buf, 8, "%u", cell.instanceCount);
				drawList->AddText(ImVec2(tl.x + 1, tl.y + 1), IM_COL32(255, 255, 255, 200), buf);
			}
		}
	}
	ImGui::Dummy(ImVec2(totalW, totalW));

	ImGui::TextColored(ImVec4(0, 0.63f, 0, 1), "■ Visible");
	ImGui::SameLine();
	ImGui::TextColored(ImVec4(0.63f, 0, 0, 1), "■ Culled");
	ImGui::SameLine();
	ImGui::TextColored(ImVec4(0.16f, 0.12f, 0.16f, 1), "■ Empty");

	ImGui::End();
}
