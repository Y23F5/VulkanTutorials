/** @file ChunkMonitor.h
 * ImGui-based 2D visualization panel for real-time chunk culling inspection.
 * Maps chunks to an NxN colored matrix -- inspired by reactor monitoring panels.
 */
#pragma once
#include <vector>
#include <cstdint>

struct ChunkMonitorCell {
	uint32_t instanceCount;
	bool visible;
};

class ChunkMonitor {
public:
	ChunkMonitor(uint32_t gridChunks, uint32_t chunkSize, uint32_t gridSize);
	~ChunkMonitor();

	void Update(const ChunkMonitorCell* cells, uint32_t& visibleCount, uint32_t& totalInstances);
	void Render();

private:
	uint32_t m_gridChunks;
	uint32_t m_chunkSize;
	uint32_t m_gridSize;
	std::vector<ChunkMonitorCell> m_cells;
	bool m_panelOpen;
};
