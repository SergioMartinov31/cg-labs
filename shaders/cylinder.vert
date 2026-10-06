#version 450

// Атрибуты вершины: должны совпадать с VkVertexInputAttributeDescription
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

// Цвет передаётся во фрагментный шейдер и интерполируется по треугольнику
layout(location = 0) out vec3 out_color;

// Uniform-буфер объекта: binding = 0 в наборе дескрипторов (set = 0)
layout(set = 0, binding = 0, std140) uniform ObjectUniforms {
	mat4 model;      // локальные координаты -> мировые
	mat4 view;       // мировые -> координаты камеры
	mat4 projection; // координаты камеры -> clip space
	vec4 color;      // цвет, выбранный в интерфейсе
	vec4 params;     // x = 1, если используем процедурные цвета вершин
} ubo;

void main() {
	gl_Position = ubo.projection * ubo.view * ubo.model * vec4(in_position, 1.0);

	// Если процедурные цвета выключены, цвет вершины считается белым
	vec3 vertex_color = mix(vec3(1.0), in_color, ubo.params.x);
	out_color = vertex_color * ubo.color.rgb;
}
