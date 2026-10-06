#include "application.hpp"

#include <cstdio>
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

// Вершина: позиция и процедурный цвет (атрибуты location = 0 и 1)
struct Vertex {
	math::Vec3 position;
	math::Vec3 color;
};

// Uniform-буфер объекта. Раскладка соответствует std140 в шейдере:
// mat4 занимает 64 байта, vec4 – 16 байт, выравнивание 16 байт.
struct alignas(16) ObjectUniforms {
	math::Mat4 model;
	math::Mat4 view;
	math::Mat4 projection;
	float color[4];
	float params[4]; // x – использовать процедурные цвета вершин
};

// ---------------------------------------------------------------------------
// Состояние сцены, которым управляет интерфейс
// ---------------------------------------------------------------------------

constexpr uint32_t object_count = 2;
constexpr uint32_t cylinder_segments = 24; // 25 вершин в каждом основании, всего 50

enum Projection : int {
	projection_perspective = 0,
	projection_orthographic = 1,
};

struct Object {
	bool visible = true;

	// Базовое преобразование (доп. задание 2)
	float position[3] = { 0.0f, 0.0f, 0.0f };
	float rotation[3] = { 0.0f, 0.0f, 0.0f }; // в градусах
	float scale[3] = { 1.0f, 1.0f, 1.0f };

	// Цвет (доп. задания 4 и 5)
	float color[3] = { 1.0f, 1.0f, 1.0f };
	bool vertex_colors = true;

	// Анимация по траектории-«восьмёрке» (доп. задание 3)
	bool animate = false;
	float speed = 1.0f;            // скорость движения по траектории
	float radius = 1.5f;           // радиус траектории
	float height = 0.75f;          // амплитуда колебаний по Y
	float spin_speed = 90.0f;      // скорость вращения вокруг своей оси, град/с
	float path_time = 0.0f;        // параметр t траектории
	float spin_angle = 0.0f;       // накопленный угол вращения, градусы

	math::Mat4 model = math::Mat4::identity();
};

struct Camera {
	int projection = projection_perspective;
	float fov = 60.0f;             // вертикальный угол обзора, градусы
	float ortho_height = 6.0f;     // высота видимой области в орт. проекции
	float near = 0.1f;
	float far = 100.0f;
	float distance = 8.0f;
	float yaw = 0.0f;
	float pitch = 30.0f;
};

Object objects[object_count];
Camera camera;
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
uint32_t index_count;

VkDescriptorSetLayout vk_descriptor_set_layout;
VkDescriptorPool vk_descriptor_pool;

// Доп. задание 6: у каждого объекта свой uniform-буфер и свой набор дескрипторов
VkDescriptorSet vk_descriptor_sets[object_count];
VkBuffer vk_uniform_buffers[object_count];
VmaAllocation vma_uniform_allocations[object_count];
void* uniform_mapped[object_count];

// ---------------------------------------------------------------------------
// Геометрия цилиндра
// ---------------------------------------------------------------------------

// Цилиндр радиуса 1 и высоты 2 с осью вдоль Y.
// Вершины: [0] центр нижнего основания, [1..S] кольцо нижнего основания,
//          [S+1] центр верхнего основания, [S+2..2S+1] кольцо верхнего.
// Треугольники заданы против часовой стрелки, если смотреть снаружи.
void buildCylinder(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
	const uint32_t S = cylinder_segments;

	// Процедурный цвет: локальная позиция [-1, 1] переводится в RGB [0, 1]
	auto make_vertex = [](float x, float y, float z) {
		return Vertex{
			.position = { x, y, z },
			.color = { x * 0.5f + 0.5f, y * 0.5f + 0.5f, z * 0.5f + 0.5f },
		};
	};

	const uint32_t bottom_center = 0;
	const uint32_t bottom_ring = 1;
	const uint32_t top_center = S + 1;
	const uint32_t top_ring = S + 2;

	vertices.push_back(make_vertex(0.0f, -1.0f, 0.0f));
	for (uint32_t i = 0; i < S; ++i) {
		const float a = 2.0f * math::pi * float(i) / float(S);
		vertices.push_back(make_vertex(std::cos(a), -1.0f, std::sin(a)));
	}

	vertices.push_back(make_vertex(0.0f, 1.0f, 0.0f));
	for (uint32_t i = 0; i < S; ++i) {
		const float a = 2.0f * math::pi * float(i) / float(S);
		vertices.push_back(make_vertex(std::cos(a), 1.0f, std::sin(a)));
	}

	for (uint32_t i = 0; i < S; ++i) {
		const uint32_t next = (i + 1) % S;

		const uint32_t b0 = bottom_ring + i, b1 = bottom_ring + next;
		const uint32_t t0 = top_ring + i, t1 = top_ring + next;

		// Нижнее основание (нормаль -Y)
		indices.insert(indices.end(), { bottom_center, b0, b1 });
		// Верхнее основание (нормаль +Y)
		indices.insert(indices.end(), { top_center, t1, t0 });
		// Боковая грань: прямоугольник из двух треугольников
		indices.insert(indices.end(), { b0, t0, b1 });
		indices.insert(indices.end(), { b1, t0, t1 });
	}
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

bool createGeometry() {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	buildCylinder(vertices, indices);
	index_count = uint32_t(indices.size());

	return createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.size() * sizeof(Vertex),
	                    vertices.data(), vk_vertex_buffer, vma_vertex_allocation) &&
	       createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.size() * sizeof(uint32_t),
	                    indices.data(), vk_index_buffer, vma_index_allocation);
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	// Описание набора: один uniform-буфер на binding = 0 для вершинного шейдера
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
	                                &vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	// Пул: N (=1 дескриптор в наборе) * S (=object_count наборов)
	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1 * object_count,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = object_count,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	// Все наборы создаются по одному и тому же описанию
	VkDescriptorSetLayout layouts[object_count];
	for (uint32_t i = 0; i < object_count; ++i) {
		layouts[i] = vk_descriptor_set_layout;
	}

	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = object_count,
		.pSetLayouts = layouts,
	};

	if (vkAllocateDescriptorSets(context.device, &allocate_info, vk_descriptor_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	// Для каждого объекта: свой uniform-буфер, связанный со своим набором
	for (uint32_t i = 0; i < object_count; ++i) {
		if (!createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(ObjectUniforms), nullptr,
		                  vk_uniform_buffers[i], vma_uniform_allocations[i], &uniform_mapped[i])) {
			return false;
		}

		const VkDescriptorBufferInfo buffer_info = {
			.buffer = vk_uniform_buffers[i],
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};

		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = vk_descriptor_sets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
		};

		vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	}

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	VkShaderModule vertex_shader = loadShader("shaders/cylinder.vert.spv");
	VkShaderModule fragment_shader = loadShader("shaders/cylinder.frag.spv");
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
			.offset = offsetof(Vertex, color),
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

	// Треугольники модели заданы против часовой стрелки (если смотреть снаружи).
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

	// Конвейер знает, что в set = 0 лежит наш набор дескрипторов
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
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
// Интерфейс
// ---------------------------------------------------------------------------

void drawCameraUI() {
	// Доп. задание 1: переключение матрицы проекции
	ImGui::SeparatorText("Projection");
	ImGui::RadioButton("Perspective", &camera.projection, projection_perspective);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &camera.projection, projection_orthographic);

	if (camera.projection == projection_perspective) {
		ImGui::SliderFloat("FOV", &camera.fov, 20.0f, 120.0f, "%.0f deg");
	} else {
		ImGui::SliderFloat("Ortho height", &camera.ortho_height, 1.0f, 20.0f);
	}

	ImGui::SeparatorText("Camera");
	ImGui::SliderFloat("Distance", &camera.distance, 2.0f, 30.0f);
	ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.0f deg");
	ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.0f deg");
}

void drawObjectUI(Object& object) {
	ImGui::Checkbox("Visible", &object.visible);

	// Доп. задание 2: позиция, поворот и растяжение
	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", object.position, 0.05f);
	ImGui::DragFloat3("Rotation", object.rotation, 1.0f, -360.0f, 360.0f, "%.0f deg");
	ImGui::DragFloat3("Scale", object.scale, 0.01f, 0.05f, 5.0f);
	if (ImGui::Button("Reset rotation & scale")) {
		const Object defaults;
		std::memcpy(object.rotation, defaults.rotation, sizeof(object.rotation));
		std::memcpy(object.scale, defaults.scale, sizeof(object.scale));
	}

	// Доп. задания 4 и 5: цвет фигуры и процедурные цвета вершин
	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Color", object.color);
	ImGui::Checkbox("Procedural vertex colors", &object.vertex_colors);

	// Доп. задание 3: анимация по траектории
	ImGui::SeparatorText("Animation");
	if (ImGui::Button(object.animate ? "Pause" : "Play")) {
		object.animate = !object.animate;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
		object.path_time = 0.0f;
		object.spin_angle = 0.0f;
	}
	ImGui::SliderFloat("Speed", &object.speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Path radius", &object.radius, 0.0f, 5.0f);
	ImGui::SliderFloat("Path height", &object.height, 0.0f, 3.0f);
	ImGui::SliderFloat("Spin speed", &object.spin_speed, -360.0f, 360.0f, "%.0f deg/s");
}

void updateObject(Object& object, float dt) {
	using namespace math;

	if (object.animate) {
		object.path_time += dt * object.speed;
		object.spin_angle += dt * object.spin_speed;
	}

	// Траектория – пространственная «восьмёрка» (кривая Лиссажу):
	// x = R sin t, y = H sin 2t, z = R sin t cos t
	const float t = object.path_time;
	const Vec3 path = {
		object.radius * std::sin(t),
		object.height * std::sin(2.0f * t),
		object.radius * std::sin(t) * std::cos(t),
	};

	// Порядок применения: растяжение -> поворот -> перенос (справа налево)
	object.model = translate({ object.position[0] + path.x,
	                           object.position[1] + path.y,
	                           object.position[2] + path.z })
	             * rotateY(radians(object.spin_angle))
	             * rotateZ(radians(object.rotation[2]))
	             * rotateY(radians(object.rotation[1]))
	             * rotateX(radians(object.rotation[0]))
	             * scale({ object.scale[0], object.scale[1], object.scale[2] });
}

} // namespace

bool initialize() {
	// Первый цилиндр стоит в центре, второй – меньше, справа и сразу анимирован
	objects[0].position[0] = 0.5f;
	objects[0].rotation[0] = 20.0f;

	objects[1].position[0] = 3.5f;
	objects[1].scale[0] = objects[1].scale[2] = 0.6f;
	objects[1].color[0] = 1.0f;
	objects[1].color[1] = 0.7f;
	objects[1].color[2] = 0.3f;
	objects[1].animate = true;

	return createGeometry() && createDescriptors() && createPipeline();
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);

	for (uint32_t i = 0; i < object_count; ++i) {
		vmaDestroyBuffer(context.allocator, vk_uniform_buffers[i], vma_uniform_allocations[i]);
	}

	vmaDestroyBuffer(context.allocator, vk_index_buffer, vma_index_allocation);
	vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vma_vertex_allocation);
}

void update(double time) {
	using namespace math;

	const float dt = last_time == 0.0 ? 0.0f : float(time - last_time);
	last_time = time;

	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab #1: Cylinder");

	ImGui::Text("%.1f FPS", double(ImGui::GetIO().Framerate));
	drawCameraUI();

	if (ImGui::BeginTabBar("objects")) {
		for (uint32_t i = 0; i < object_count; ++i) {
			char label[32];
			std::snprintf(label, sizeof(label), "Cylinder %u", i + 1);
			if (ImGui::BeginTabItem(label)) {
				ImGui::PushID(int(i));
				drawObjectUI(objects[i]);
				ImGui::PopID();
				ImGui::EndTabItem();
			}
		}
		ImGui::EndTabBar();
	}

	ImGui::End();

	for (Object& object : objects) {
		updateObject(object, dt);
	}

	// Камера вращается вокруг начала координат: сначала поворачиваем сцену
	// на yaw и pitch, затем отодвигаем её от камеры на distance
	view_matrix = translate({ 0.0f, 0.0f, -camera.distance })
	            * rotateX(radians(camera.pitch))
	            * rotateY(radians(-camera.yaw));

	const VkExtent2D extent = graphics::internal::context.swapchain_extent;
	const float aspect = float(extent.width) / float(std::max(extent.height, 1u));

	if (camera.projection == projection_perspective) {
		projection_matrix = perspective(radians(camera.fov), aspect, camera.near, camera.far);
	} else {
		const float h = camera.ortho_height * 0.5f;
		const float w = h * aspect;
		projection_matrix = orthographic(-w, w, -h, h, camera.near, camera.far);
	}
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	// К этому моменту GPU закончил предыдущий кадр (prepare() дождался fence),
	// поэтому uniform-буферы можно безопасно перезаписать
	for (uint32_t i = 0; i < object_count; ++i) {
		const Object& object = objects[i];

		const ObjectUniforms uniforms = {
			.model = object.model,
			.view = view_matrix,
			.projection = projection_matrix,
			.color = { object.color[0], object.color[1], object.color[2], 1.0f },
			.params = { object.vertex_colors ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f },
		};

		std::memcpy(uniform_mapped[i], &uniforms, sizeof(uniforms));
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
		{ .color = { .float32 = { 0.08f, 0.09f, 0.12f, 1.0f } } },
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

	// Одна и та же геометрия рисуется дважды, но с разными наборами
	// дескрипторов – то есть с разными матрицами и цветом
	for (uint32_t i = 0; i < object_count; ++i) {
		if (!objects[i].visible) {
			continue;
		}

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
		                        0, 1, &vk_descriptor_sets[i], 0, nullptr);
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);

	vkEndCommandBuffer(cmd);
}

} // namespace application
