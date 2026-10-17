#include "application.hpp"

#include <cstring>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include "math.hpp"

namespace application {

namespace {

// ---------------------------------------------------------------------------
// Данные, которые лежат в памяти GPU
// ---------------------------------------------------------------------------

// Вершина: позиция и нормаль (атрибуты location = 0 и 1)
struct Vertex {
	math::Vec3 position;
	math::Vec3 normal;
};

// Набор 0, binding 0: uniform-буфер сцены (std140, всё по 16 байт)
struct alignas(16) SceneUniforms {
	math::Mat4 view;
	math::Mat4 projection;
	float camera_position[4];
	float ambient[4];        // rgb – цвет * интенсивность
	float sun_direction[4];  // xyz – куда светит направленный источник
	float sun_color[4];      // rgb – цвет, a – интенсивность
	uint32_t counts[4];      // x – точечных источников, y – прожекторов
};

// Набор 1, binding 0: uniform-буфер модели
struct alignas(16) ModelUniforms {
	math::Mat4 model;
	float albedo[4];         // rgb – диффузный цвет
	float specular[4];       // rgb – цвет блика, a – коэффициент блеска
};

// Элементы storage-буферов (std430). Раскладка совпадает со структурами
// PointLight и SpotLight во фрагментном шейдере.
struct GpuPointLight {
	float position[4];
	float color[4];          // rgb – цвет, a – интенсивность
};

struct GpuSpotLight {
	float position[4];
	float direction[4];
	float color[4];          // rgb – цвет, a – интенсивность
	float cone[4];           // x – cos внутреннего угла, y – cos внешнего
};

// ---------------------------------------------------------------------------
// Состояние сцены, которым управляет интерфейс
// ---------------------------------------------------------------------------

constexpr uint32_t object_count = 3;        // два цилиндра и пол
constexpr uint32_t cylinder_segments = 24; // 25 вершин в каждом основании, всего 50 из тз

// Участок общего вершинного и индексного буфера, где лежит одна фигура
struct Mesh {
	uint32_t first_index;   // с какого индекса начинается фигура
	uint32_t index_count;   // сколько индексов рисовать
	int32_t vertex_offset;  // сколько прибавить к каждому индексу
};

struct Object {
	const char* name = "";
	const Mesh* mesh = nullptr;
	bool visible = true;

	float position[3] = { 0.0f, 0.0f, 0.0f };
	float rotation[3] = { 0.0f, 0.0f, 0.0f }; // в градусах
	float scale[3] = { 1.0f, 1.0f, 1.0f };

	// Материал по модели Блинна-Фонга
	float albedo[3] = { 0.8f, 0.8f, 0.8f };
	float specular[3] = { 0.5f, 0.5f, 0.5f };
	float shininess = 32.0f;

	math::Mat4 model = math::Mat4::identity();
};

// Свободная камера: позиция и поворот (yaw – влево-вправо, pitch – вверх-вниз)
struct Camera {
	float position[3] = { 0.0f, 2.0f, 7.0f };
	float yaw = 0.0f;
	float pitch = -15.0f;
	float fov = 60.0f;
	float near = 0.1f;
	float far = 100.0f;
	float move_speed = 3.0f;     // единиц в секунду
	float look_speed = 0.2f;     // градусов на пиксель движения мыши
};

struct AmbientLight {
	float color[3] = { 0.55f, 0.65f, 1.0f };
	float intensity = 0.08f;
};

struct DirectionalLight {
	bool enabled = true;
	float direction[3] = { -0.5f, -1.0f, -0.4f };
	float color[3] = { 1.0f, 0.95f, 0.85f };
	float intensity = 0.5f;
};

struct PointLight {
	float position[3] = { 0.0f, 1.5f, 0.0f };
	float color[3] = { 1.0f, 1.0f, 1.0f };
	float intensity = 5.0f;
};

struct SpotLight {
	float position[3] = { 0.0f, 4.0f, 0.0f };
	float direction[3] = { 0.0f, -1.0f, 0.0f };
	float color[3] = { 1.0f, 1.0f, 1.0f };
	float intensity = 25.0f;
	float inner_angle = 15.0f;   // градусы от оси до начала затухания
	float outer_angle = 25.0f;   // градусы от оси до полной темноты
};

Mesh cylinder_mesh;
Mesh floor_mesh;

Object objects[object_count];
Camera camera;
AmbientLight ambient;
DirectionalLight sun;
std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;

math::Mat4 view_matrix = math::Mat4::identity();
math::Mat4 projection_matrix = math::Mat4::identity();
double last_time = 0.0;

// ---------------------------------------------------------------------------
// Объекты Vulkan
// ---------------------------------------------------------------------------

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

VkBuffer vk_vertex_buffer;
VmaAllocation vma_vertex_allocation;
VkBuffer vk_index_buffer;
VmaAllocation vma_index_allocation;

VkDescriptorSetLayout vk_scene_set_layout;  // набор 0: сцена
VkDescriptorSetLayout vk_model_set_layout;  // набор 1: модель
VkDescriptorPool vk_descriptor_pool;

// Набор 0 – один на всю сцену
VkDescriptorSet vk_scene_set;
VkBuffer vk_scene_buffer;
VmaAllocation vma_scene_allocation;
void* scene_mapped;

// Набор 1 – свой у каждого объекта
VkDescriptorSet vk_model_sets[object_count];
VkBuffer vk_model_buffers[object_count];
VmaAllocation vma_model_allocations[object_count];
void* model_mapped[object_count];

// Storage-буфер с массивом источников. Может пересоздаваться большего
// размера, когда источников становится больше, чем в него помещается.
struct StorageBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	void* mapped = nullptr;
	uint32_t capacity = 0;   // на сколько элементов выделена память
};

StorageBuffer point_light_buffer;  // набор 0, binding 1
StorageBuffer spot_light_buffer;   // набор 0, binding 2

// ---------------------------------------------------------------------------
// Геометрия
// ---------------------------------------------------------------------------

// Цилиндр радиуса 1 и высоты 2 с осью вдоль Y.
// Чтобы у дна, крышки и стенки были разные нормали, точки на краях
// оснований хранятся дважды: с нормалью основания и с нормалью стенки.
// Вершины: [0] центр дна, [1..S] кольцо дна (нормаль -Y),
//          [S+1] центр крышки, [S+2..2S+1] кольцо крышки (нормаль +Y),
//          [2S+2..3S+1] низ стенки, [3S+2..4S+1] верх стенки (нормаль наружу).
// Треугольники заданы против часовой стрелки, если смотреть снаружи.
void buildCylinder(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
	const uint32_t S = cylinder_segments;

	const uint32_t bottom_center = 0;
	const uint32_t bottom_ring = 1;
	const uint32_t top_center = S + 1;
	const uint32_t top_ring = S + 2;
	const uint32_t side_bottom = 2 * S + 2;
	const uint32_t side_top = 3 * S + 2;

	const math::Vec3 down = { 0.0f, -1.0f, 0.0f };
	const math::Vec3 up = { 0.0f, 1.0f, 0.0f };

	// Дно
	vertices.push_back({ { 0.0f, -1.0f, 0.0f }, down });
	for (uint32_t i = 0; i < S; ++i) {
		const float a = 2.0f * math::pi * float(i) / float(S);
		vertices.push_back({ { std::cos(a), -1.0f, std::sin(a) }, down });
	}

	// Крышка
	vertices.push_back({ { 0.0f, 1.0f, 0.0f }, up });
	for (uint32_t i = 0; i < S; ++i) {
		const float a = 2.0f * math::pi * float(i) / float(S);
		vertices.push_back({ { std::cos(a), 1.0f, std::sin(a) }, up });
	}

	// Стенка: нормаль смотрит от оси наружу, (cos a, 0, sin a).
	// Соседние дольки делят вершины, поэтому стенка освещается гладко.
	for (uint32_t i = 0; i < S; ++i) {
		const float a = 2.0f * math::pi * float(i) / float(S);
		vertices.push_back({ { std::cos(a), -1.0f, std::sin(a) }, { std::cos(a), 0.0f, std::sin(a) } });
	}
	for (uint32_t i = 0; i < S; ++i) {
		const float a = 2.0f * math::pi * float(i) / float(S);
		vertices.push_back({ { std::cos(a), 1.0f, std::sin(a) }, { std::cos(a), 0.0f, std::sin(a) } });
	}

	for (uint32_t i = 0; i < S; ++i) {
		const uint32_t next = (i + 1) % S;

		// Дно и крышка
		indices.insert(indices.end(), { bottom_center, bottom_ring + i, bottom_ring + next });
		indices.insert(indices.end(), { top_center, top_ring + next, top_ring + i });

		// Стенка: прямоугольник из двух треугольников
		const uint32_t b0 = side_bottom + i, b1 = side_bottom + next;
		const uint32_t t0 = side_top + i, t1 = side_top + next;
		indices.insert(indices.end(), { b0, t0, b1 });
		indices.insert(indices.end(), { b1, t0, t1 });
	}
}

// Пол: квадрат от -1 до 1 в плоскости XZ с нормалью вверх.
// Размер и положение задаются матрицей модели.
void buildFloor(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
	const math::Vec3 up = { 0.0f, 1.0f, 0.0f };

	vertices.push_back({ { -1.0f, 0.0f, -1.0f }, up });
	vertices.push_back({ { -1.0f, 0.0f, 1.0f }, up });
	vertices.push_back({ { 1.0f, 0.0f, 1.0f }, up });
	vertices.push_back({ { 1.0f, 0.0f, -1.0f }, up });

	indices.insert(indices.end(), { 0, 1, 2, 0, 2, 3 });
}

// ---------------------------------------------------------------------------
// Вспомогательные функции
// ---------------------------------------------------------------------------

VkShaderModule loadShader(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file " << path << '\n';
		return VK_NULL_HANDLE;
	}

	const size_t size = size_t(file.tellg());
	std::vector<uint32_t> code((size + 3) / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return module;
}

// Создаёт буфер в памяти, доступной и CPU, и GPU, и копирует туда данные
bool createBuffer(VkBufferUsageFlags usage, VkDeviceSize size, const void* data,
                  VkBuffer& buffer, VmaAllocation& allocation, void** mapped = nullptr) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo info;
	if (vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &allocation_info,
	                    &buffer, &allocation, &info) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan buffer\n";
		return false;
	}

	if (data != nullptr) {
		std::memcpy(info.pMappedData, data, size_t(size));
	}

	if (mapped != nullptr) {
		*mapped = info.pMappedData;
	}

	return true;
}

// Связывает буфер с binding'ом набора дескрипторов
void writeDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                     VkBuffer buffer, VkDeviceSize range) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer,
		.offset = 0,
		.range = range,
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

// Гарантирует, что в storage-буфере хватит места на count элементов.
// Если не хватает – создаёт новый буфер вдвое больше и подключает его
// к тому же binding'у набора сцены вместо старого.
bool ensureCapacity(StorageBuffer& storage, uint32_t count, size_t element_size, uint32_t binding) {
	if (count <= storage.capacity) {
		return true;
	}

	uint32_t capacity = std::max<uint32_t>(storage.capacity, 4);
	while (capacity < count) {
		capacity *= 2;
	}

	auto& context = graphics::internal::context;

	// Старый буфер мог использоваться в последнем отправленном кадре,
	// поэтому перед удалением дожидаемся, пока GPU закончит работу
	if (storage.buffer != VK_NULL_HANDLE) {
		vkQueueWaitIdle(context.graphics_queue);
		vmaDestroyBuffer(context.allocator, storage.buffer, storage.allocation);
	}

	if (!createBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, capacity * element_size, nullptr,
	                  storage.buffer, storage.allocation, &storage.mapped)) {
		storage = {};
		return false;
	}

	storage.capacity = capacity;
	writeDescriptor(vk_scene_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                storage.buffer, VK_WHOLE_SIZE);
	return true;
}

bool createGeometry() {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	// Обе фигуры лежат в одних и тех же буферах друг за другом.
	// Mesh запоминает, где начинается каждая из них.
	cylinder_mesh.first_index = uint32_t(indices.size());
	cylinder_mesh.vertex_offset = int32_t(vertices.size());
	buildCylinder(vertices, indices);
	cylinder_mesh.index_count = uint32_t(indices.size()) - cylinder_mesh.first_index;

	floor_mesh.first_index = uint32_t(indices.size());
	floor_mesh.vertex_offset = int32_t(vertices.size());
	buildFloor(vertices, indices);
	floor_mesh.index_count = uint32_t(indices.size()) - floor_mesh.first_index;

	return createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.size() * sizeof(Vertex),
	                    vertices.data(), vk_vertex_buffer, vma_vertex_allocation) &&
	       createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.size() * sizeof(uint32_t),
	                    indices.data(), vk_index_buffer, vma_index_allocation);
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	// Шаблон набора 0 (сцена): uniform-буфер сцены и два storage-буфера с источниками
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
	                                &vk_scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	// Шаблон набора 1 (модель): uniform-буфер с матрицей и материалом
	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
	                                &vk_model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	// Пул: 1 набор сцены + object_count наборов моделей.
	// Uniform-буферов: 1 у сцены + по 1 у каждой модели; storage-буферов: 2 у сцены.
	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 + object_count },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + object_count,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	// Набор сцены
	const VkDescriptorSetAllocateInfo scene_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_allocate_info, &vk_scene_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	if (!createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(SceneUniforms), nullptr,
	                  vk_scene_buffer, vma_scene_allocation, &scene_mapped)) {
		return false;
	}

	writeDescriptor(vk_scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	                vk_scene_buffer, sizeof(SceneUniforms));

	// Storage-буферы создаются с небольшим запасом и растут при необходимости
	if (!ensureCapacity(point_light_buffer, 1, sizeof(GpuPointLight), 1) ||
	    !ensureCapacity(spot_light_buffer, 1, sizeof(GpuSpotLight), 2)) {
		return false;
	}

	// Наборы моделей: все по одному шаблону, у каждого свой uniform-буфер
	VkDescriptorSetLayout model_layouts[object_count];
	for (uint32_t i = 0; i < object_count; ++i) {
		model_layouts[i] = vk_model_set_layout;
	}

	const VkDescriptorSetAllocateInfo model_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = object_count,
		.pSetLayouts = model_layouts,
	};

	if (vkAllocateDescriptorSets(context.device, &model_allocate_info, vk_model_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate model descriptor sets\n";
		return false;
	}

	for (uint32_t i = 0; i < object_count; ++i) {
		if (!createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(ModelUniforms), nullptr,
		                  vk_model_buffers[i], vma_model_allocations[i], &model_mapped[i])) {
			return false;
		}

		writeDescriptor(vk_model_sets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		                vk_model_buffers[i], sizeof(ModelUniforms));
	}

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = loadShader("shaders/lit.vert.spv");
	VkShaderModule fragment_shader = loadShader("shaders/lit.frag.spv");
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		return false;
	}

	const VkPipelineShaderStageCreateInfo stage_infos[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	// Один вершинный буфер, в котором атрибуты чередуются (interleaved)
	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
	};

	const VkPipelineVertexInputStateCreateInfo input_state_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo assembly_state_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// Треугольники заданы против часовой стрелки (если смотреть снаружи).
	// Матрица проекции переворачивает Y под Vulkan, и в координатах
	// фреймбуфера (Y вниз) такие треугольники имеют положительную площадь,
	// что по спецификации Vulkan соответствует COUNTER_CLOCKWISE.
	const VkPipelineRasterizationStateCreateInfo raster_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo sample_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = true,
		.depthWriteEnable = true,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState attachment_info = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo blend_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &attachment_info,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	// Конвейер знает оба шаблона: set = 0 – сцена, set = 1 – модель
	const VkDescriptorSetLayout set_layouts[] = {
		vk_scene_set_layout,
		vk_model_set_layout,
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stage_infos,
		.pVertexInputState = &input_state_info,
		.pInputAssemblyState = &assembly_state_info,
		.pViewportState = &viewport_info,
		.pRasterizationState = &raster_info,
		.pMultisampleState = &sample_info,
		.pDepthStencilState = &depth_info,
		.pColorBlendState = &blend_info,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &vk_pipeline);

	// После создания конвейера шейдерные модули больше не нужны
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

// ---------------------------------------------------------------------------
// Камера
// ---------------------------------------------------------------------------

// Направления камеры в мире. Камера смотрит вдоль своей -Z,
// поэтому «вперёд» – это (0, 0, -1), повёрнутый на yaw и pitch.
math::Vec3 cameraForward() {
	const float yaw = math::radians(camera.yaw);
	const float pitch = math::radians(camera.pitch);
	return { -std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch) };
}

math::Vec3 cameraRight() {
	const float yaw = math::radians(camera.yaw);
	return { std::cos(yaw), 0.0f, -std::sin(yaw) };
}

// Управление с клавиатуры и мыши, когда курсор не над окном интерфейса:
// WASD – вперёд/влево/назад/вправо, E/Q – вверх/вниз,
// зажатая правая кнопка мыши – поворот камеры
void handleCameraInput(float dt) {
	using namespace math;

	const ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureKeyboard) {
		const Vec3 forward = cameraForward();
		const Vec3 right = cameraRight();
		const Vec3 up = { 0.0f, 1.0f, 0.0f };

		Vec3 move = { 0.0f, 0.0f, 0.0f };
		if (ImGui::IsKeyDown(ImGuiKey_W)) move = move + forward;
		if (ImGui::IsKeyDown(ImGuiKey_S)) move = move - forward;
		if (ImGui::IsKeyDown(ImGuiKey_D)) move = move + right;
		if (ImGui::IsKeyDown(ImGuiKey_A)) move = move - right;
		if (ImGui::IsKeyDown(ImGuiKey_E)) move = move + up;
		if (ImGui::IsKeyDown(ImGuiKey_Q)) move = move - up;

		const Vec3 step = normalize(move) * (camera.move_speed * dt);
		camera.position[0] += step.x;
		camera.position[1] += step.y;
		camera.position[2] += step.z;
	}

	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		camera.yaw -= io.MouseDelta.x * camera.look_speed;
		camera.pitch -= io.MouseDelta.y * camera.look_speed;
		camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);
	}
}

// ---------------------------------------------------------------------------
// Интерфейс
// ---------------------------------------------------------------------------

void drawCameraUI() {
	ImGui::DragFloat3("Position", camera.position, 0.05f);
	ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.0f deg");
	ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.0f deg");
	ImGui::SliderFloat("FOV", &camera.fov, 20.0f, 120.0f, "%.0f deg");
	ImGui::SliderFloat("Move speed", &camera.move_speed, 0.5f, 20.0f);
	ImGui::TextDisabled("WASD / Q E - move, right mouse - look");
}

void drawLightsUI() {
	ImGui::SeparatorText("Ambient");
	ImGui::ColorEdit3("Ambient color", ambient.color);
	ImGui::SliderFloat("Ambient intensity", &ambient.intensity, 0.0f, 1.0f);

	ImGui::SeparatorText("Directional");
	ImGui::Checkbox("Enabled", &sun.enabled);
	ImGui::DragFloat3("Direction", sun.direction, 0.01f, -1.0f, 1.0f);
	ImGui::ColorEdit3("Color", sun.color);
	ImGui::SliderFloat("Intensity", &sun.intensity, 0.0f, 5.0f);
}

void drawPointLightsUI() {
	int remove = -1;

	for (size_t i = 0; i < point_lights.size(); ++i) {
		PointLight& light = point_lights[i];

		ImGui::PushID(int(i));
		if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Point light %zu", i + 1)) {
			ImGui::DragFloat3("Position", light.position, 0.05f);
			ImGui::ColorEdit3("Color", light.color);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);
			if (ImGui::Button("Remove")) {
				remove = int(i);
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	// Удаляем после цикла, чтобы не менять массив во время обхода
	if (remove >= 0) {
		point_lights.erase(point_lights.begin() + remove);
	}

	if (ImGui::Button("Add point light")) {
		point_lights.push_back({});
	}
}

void drawSpotLightsUI() {
	int remove = -1;

	for (size_t i = 0; i < spot_lights.size(); ++i) {
		SpotLight& light = spot_lights[i];

		ImGui::PushID(int(i));
		if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Spot light %zu", i + 1)) {
			ImGui::DragFloat3("Position", light.position, 0.05f);
			ImGui::DragFloat3("Direction", light.direction, 0.01f, -1.0f, 1.0f);
			ImGui::ColorEdit3("Color", light.color);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f);
			ImGui::SliderFloat("Inner angle", &light.inner_angle, 0.0f, 89.0f, "%.0f deg");
			ImGui::SliderFloat("Outer angle", &light.outer_angle, 0.0f, 89.0f, "%.0f deg");
			if (ImGui::Button("Remove")) {
				remove = int(i);
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	if (remove >= 0) {
		spot_lights.erase(spot_lights.begin() + remove);
	}

	if (ImGui::Button("Add spot light")) {
		spot_lights.push_back({});
	}
}

void drawObjectUI(Object& object) {
	ImGui::Checkbox("Visible", &object.visible);

	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", object.position, 0.05f);
	ImGui::DragFloat3("Rotation", object.rotation, 1.0f, -360.0f, 360.0f, "%.0f deg");
	ImGui::DragFloat3("Scale", object.scale, 0.01f, 0.05f, 20.0f);

	ImGui::SeparatorText("Material");
	ImGui::ColorEdit3("Albedo", object.albedo);
	ImGui::ColorEdit3("Specular", object.specular);
	ImGui::SliderFloat("Shininess", &object.shininess, 1.0f, 256.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
}

void drawUI() {
	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400, 680), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab #2: Camera and lighting");

	ImGui::Text("%.1f FPS", double(ImGui::GetIO().Framerate));

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("camera");
		drawCameraUI();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Ambient and directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::PushID("sun");
		drawLightsUI();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Point lights")) {
		ImGui::PushID("point");
		drawPointLightsUI();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Spot lights")) {
		ImGui::PushID("spot");
		drawSpotLightsUI();
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Objects")) {
		if (ImGui::BeginTabBar("objects")) {
			for (uint32_t i = 0; i < object_count; ++i) {
				if (ImGui::BeginTabItem(objects[i].name)) {
					ImGui::PushID(int(i));
					drawObjectUI(objects[i]);
					ImGui::PopID();
					ImGui::EndTabItem();
				}
			}
			ImGui::EndTabBar();
		}
	}

	ImGui::End();
}

void updateObject(Object& object) {
	using namespace math;

	// Порядок применения: растяжение -> поворот -> перенос (справа налево)
	object.model = translate({ object.position[0], object.position[1], object.position[2] })
	             * rotateZ(radians(object.rotation[2]))
	             * rotateY(radians(object.rotation[1]))
	             * rotateX(radians(object.rotation[0]))
	             * scale({ object.scale[0], object.scale[1], object.scale[2] });
}

// Записать цвет rgb и четвёртое число a в массив из 4 float
void setColor(float* dst, const float* rgb, float a) {
	dst[0] = rgb[0];
	dst[1] = rgb[1];
	dst[2] = rgb[2];
	dst[3] = a;
}

} // namespace

bool initialize() {
	// Два цилиндра и пол под ними
	objects[0].name = "Cylinder 1";
	objects[0].mesh = &cylinder_mesh;
	objects[0].position[0] = -1.6f;
	objects[0].albedo[0] = 0.85f;
	objects[0].albedo[1] = 0.35f;
	objects[0].albedo[2] = 0.30f;
	objects[0].shininess = 64.0f;

	objects[1].name = "Cylinder 2";
	objects[1].mesh = &cylinder_mesh;
	objects[1].position[0] = 1.6f;
	objects[1].scale[0] = objects[1].scale[2] = 0.6f;
	objects[1].albedo[0] = 0.30f;
	objects[1].albedo[1] = 0.55f;
	objects[1].albedo[2] = 0.90f;
	objects[1].shininess = 128.0f;

	objects[2].name = "Floor";
	objects[2].mesh = &floor_mesh;
	objects[2].position[1] = -1.0f;
	objects[2].scale[0] = objects[2].scale[2] = 8.0f;
	objects[2].albedo[0] = objects[2].albedo[1] = objects[2].albedo[2] = 0.75f;
	objects[2].specular[0] = objects[2].specular[1] = objects[2].specular[2] = 0.15f;
	objects[2].shininess = 16.0f;

	// Один тёплый точечный источник перед цилиндрами
	PointLight point;
	point.position[0] = 0.0f;
	point.position[1] = 0.3f;
	point.position[2] = 1.5f;
	point.color[0] = 1.0f;
	point.color[1] = 0.6f;
	point.color[2] = 0.25f;
	point.intensity = 3.0f;
	point_lights.push_back(point);

	// Один холодный прожектор над вторым цилиндром
	SpotLight spot;
	spot.position[0] = 1.6f;
	spot.position[1] = 3.5f;
	spot.position[2] = 0.0f;
	spot.color[0] = 0.5f;
	spot.color[1] = 0.8f;
	spot.color[2] = 1.0f;
	spot.intensity = 12.0f;
	spot.inner_angle = 18.0f;
	spot.outer_angle = 28.0f;
	spot_lights.push_back(spot);

	return createGeometry() && createDescriptors() && createPipeline();
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);

	for (uint32_t i = 0; i < object_count; ++i) {
		vmaDestroyBuffer(context.allocator, vk_model_buffers[i], vma_model_allocations[i]);
	}

	vmaDestroyBuffer(context.allocator, spot_light_buffer.buffer, spot_light_buffer.allocation);
	vmaDestroyBuffer(context.allocator, point_light_buffer.buffer, point_light_buffer.allocation);
	vmaDestroyBuffer(context.allocator, vk_scene_buffer, vma_scene_allocation);

	vmaDestroyBuffer(context.allocator, vk_index_buffer, vma_index_allocation);
	vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vma_vertex_allocation);
}

void update(double time) {
	using namespace math;

	const float dt = last_time == 0.0 ? 0.0f : float(time - last_time);
	last_time = time;

	drawUI();
	handleCameraInput(dt);

	for (Object& object : objects) {
		updateObject(object);
	}

	// Матрица вида – обратная к матрице «модели» камеры T(pos) * Ry(yaw) * Rx(pitch).
	// Обратная к произведению – произведение обратных в обратном порядке:
	// (T * Ry * Rx)^-1 = Rx^-1 * Ry^-1 * T^-1, а обратный поворот – поворот на минус угол.
	view_matrix = rotateX(radians(-camera.pitch))
	            * rotateY(radians(-camera.yaw))
	            * translate({ -camera.position[0], -camera.position[1], -camera.position[2] });

	const VkExtent2D extent = graphics::internal::context.swapchain_extent;
	const float aspect = float(extent.width) / float(std::max(extent.height, 1u));
	projection_matrix = perspective(radians(camera.fov), aspect, camera.near, camera.far);
}

void render(const graphics::internal::FrameData& fd) {
	using namespace math;
	auto& context = graphics::internal::context;

	// К этому моменту GPU закончил предыдущий кадр (prepare() дождался fence),
	// поэтому буферы можно безопасно перезаписать

	// Источники света – в storage-буферы
	ensureCapacity(point_light_buffer, uint32_t(point_lights.size()), sizeof(GpuPointLight), 1);
	ensureCapacity(spot_light_buffer, uint32_t(spot_lights.size()), sizeof(GpuSpotLight), 2);

	auto* gpu_points = static_cast<GpuPointLight*>(point_light_buffer.mapped);
	for (size_t i = 0; i < point_lights.size(); ++i) {
		const PointLight& light = point_lights[i];
		gpu_points[i] = {
			.position = { light.position[0], light.position[1], light.position[2], 1.0f },
			.color = { light.color[0], light.color[1], light.color[2], light.intensity },
		};
	}

	auto* gpu_spots = static_cast<GpuSpotLight*>(spot_light_buffer.mapped);
	for (size_t i = 0; i < spot_lights.size(); ++i) {
		const SpotLight& light = spot_lights[i];
		const Vec3 direction = normalize({ light.direction[0], light.direction[1], light.direction[2] });
		// Внешний угол не может быть меньше внутреннего – иначе края не будет
		const float outer = std::max(light.outer_angle, light.inner_angle);
		gpu_spots[i] = {
			.position = { light.position[0], light.position[1], light.position[2], 1.0f },
			.direction = { direction.x, direction.y, direction.z, 0.0f },
			.color = { light.color[0], light.color[1], light.color[2], light.intensity },
			.cone = { std::cos(radians(light.inner_angle)), std::cos(radians(outer)), 0.0f, 0.0f },
		};
	}

	// Данные сцены
	SceneUniforms scene = {
		.view = view_matrix,
		.projection = projection_matrix,
		.camera_position = { camera.position[0], camera.position[1], camera.position[2], 1.0f },
		.ambient = { ambient.color[0] * ambient.intensity, ambient.color[1] * ambient.intensity,
		             ambient.color[2] * ambient.intensity, 0.0f },
		.counts = { uint32_t(point_lights.size()), uint32_t(spot_lights.size()), 0, 0 },
	};

	const Vec3 sun_direction = normalize({ sun.direction[0], sun.direction[1], sun.direction[2] });
	scene.sun_direction[0] = sun_direction.x;
	scene.sun_direction[1] = sun_direction.y;
	scene.sun_direction[2] = sun_direction.z;
	setColor(scene.sun_color, sun.color, sun.enabled ? sun.intensity : 0.0f);

	std::memcpy(scene_mapped, &scene, sizeof(scene));

	// Данные каждой модели
	for (uint32_t i = 0; i < object_count; ++i) {
		const Object& object = objects[i];

		ModelUniforms model = { .model = object.model };
		setColor(model.albedo, object.albedo, 1.0f);
		setColor(model.specular, object.specular, object.shininess);

		std::memcpy(model_mapped[i], &model, sizeof(model));
	}

	VkCommandBuffer cmd = fd.command_buffer;

	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(cmd, &begin_info);

	// Значения для очистки: цвет фона и максимальная глубина 1.0
	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.02f, 0.02f, 0.03f, 1.0f } } },
		{ .depthStencil = { .depth = 1.0f, .stencil = 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(cmd, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = { .extent = context.swapchain_extent };

	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vk_vertex_buffer, &offset);
	vkCmdBindIndexBuffer(cmd, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

	// Набор 0 (сцена) подключается один раз на весь кадр
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
	                        0, 1, &vk_scene_set, 0, nullptr);

	// Набор 1 (модель) – перед каждым объектом свой
	for (uint32_t i = 0; i < object_count; ++i) {
		const Object& object = objects[i];
		if (!object.visible) {
			continue;
		}

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
		                        1, 1, &vk_model_sets[i], 0, nullptr);
		vkCmdDrawIndexed(cmd, object.mesh->index_count, 1,
		                 object.mesh->first_index, object.mesh->vertex_offset, 0);
	}

	vkCmdEndRenderPass(cmd);

	vkEndCommandBuffer(cmd);
}

} // namespace application
