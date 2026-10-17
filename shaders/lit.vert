#version 450

// Атрибуты вершины: должны совпадать с VkVertexInputAttributeDescription
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

// Позиция и нормаль в мировых координатах для расчёта освещения
// во фрагментном шейдере. GPU интерполирует их по треугольнику.
layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;

// Набор 0: данные сцены, общие для всех объектов (камера и свет)
layout(set = 0, binding = 0, std140) uniform SceneUniforms {
	mat4 view;             // мировые -> координаты камеры
	mat4 projection;       // координаты камеры -> clip space
	vec4 camera_position;  // xyz – позиция камеры в мире
	vec4 ambient;          // rgb – рассеянный свет (цвет * интенсивность)
	vec4 sun_direction;    // xyz – куда светит направленный источник
	vec4 sun_color;        // rgb – цвет, a – интенсивность
	uvec4 counts;          // x – точечных источников, y – прожекторов
} scene;

// Набор 1: данные конкретной модели
layout(set = 1, binding = 0, std140) uniform ModelUniforms {
	mat4 model;            // локальные координаты -> мировые
	vec4 albedo;           // rgb – диффузный цвет материала
	vec4 specular;         // rgb – цвет блика, a – коэффициент блеска
} object;

void main() {
	const vec4 world_position = object.model * vec4(in_position, 1.0);
	out_world_position = world_position.xyz;

	// Нормаль преобразуется обратной транспонированной матрицей модели:
	// при неравномерном растяжении обычная матрица модели наклонила бы
	// нормаль, и она перестала бы быть перпендикулярной поверхности.
	// Сдвиг на нормаль не влияет, поэтому берём только верхний блок 3x3.
	out_world_normal = transpose(inverse(mat3(object.model))) * in_normal;

	gl_Position = scene.projection * scene.view * world_position;
}
