#version 450

// Должны зеркалить тип и location выходов вершинного шейдера
layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;

// Цвет пикселя в первое (и единственное) цветное изображение
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
	mat4 view;
	mat4 projection;
	vec4 camera_position;
	vec4 ambient;
	vec4 sun_direction;
	vec4 sun_color;
	uvec4 counts;
} scene;

// Storage-буферы: массивы без фиксированной длины. Сколько элементов
// реально заполнено, шейдер узнаёт из scene.counts.
struct PointLight {
	vec4 position;   // xyz – позиция
	vec4 color;      // rgb – цвет, a – интенсивность
};

struct SpotLight {
	vec4 position;   // xyz – позиция
	vec4 direction;  // xyz – куда светит
	vec4 color;      // rgb – цвет, a – интенсивность
	vec4 cone;       // x – cos внутреннего угла, y – cos внешнего угла
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(set = 0, binding = 2, std430) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

layout(set = 1, binding = 0, std140) uniform ModelUniforms {
	mat4 model;
	vec4 albedo;
	vec4 specular;
} object;

// Модель Блинна-Фонга для одного источника:
// N – нормаль, V – направление на камеру, L – направление на источник,
// radiance – сколько света пришло от источника в эту точку.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
	const float n_dot_l = max(dot(N, L), 0.0);

	// Диффузная часть по закону Ламберта: чем круче падает свет, тем ярче
	const vec3 diffuse = object.albedo.rgb * n_dot_l;

	// Блик: половинчатый вектор H между направлением на свет и на камеру.
	// Чем ближе H к нормали, тем ярче блик. Блика нет, если свет сзади.
	const vec3 H = normalize(L + V);
	const float highlight = n_dot_l > 0.0 ? pow(max(dot(N, H), 0.0), object.specular.a) : 0.0;
	const vec3 specular = object.specular.rgb * highlight;

	return (diffuse + specular) * radiance;
}

// Закон обратных квадратов: интенсивность падает как 1 / D^2.
// max() защищает от деления на ноль, если точка совпала с источником.
float inverseSquare(float distance) {
	return 1.0 / max(distance * distance, 0.0001);
}

void main() {
	// После интерполяции длина нормали может стать меньше 1 – нормализуем
	const vec3 N = normalize(in_world_normal);
	const vec3 V = normalize(scene.camera_position.xyz - in_world_position);

	// Рассеянный свет: приходит отовсюду, направления не имеет
	vec3 color = scene.ambient.rgb * object.albedo.rgb;

	// Направленный источник: лучи параллельны, затухания нет
	const vec3 sun_L = normalize(-scene.sun_direction.xyz);
	color += blinnPhong(N, V, sun_L, scene.sun_color.rgb * scene.sun_color.a);

	// Точечные источники
	for (uint i = 0; i < scene.counts.x; ++i) {
		const PointLight light = point_lights[i];

		const vec3 to_light = light.position.xyz - in_world_position;
		const float distance = length(to_light);
		const vec3 L = to_light / distance;

		const vec3 radiance = light.color.rgb * light.color.a * inverseSquare(distance);
		color += blinnPhong(N, V, L, radiance);
	}

	// Прожекторы: как точечный источник, но только внутри конуса
	for (uint i = 0; i < scene.counts.y; ++i) {
		const SpotLight light = spot_lights[i];

		const vec3 to_light = light.position.xyz - in_world_position;
		const float distance = length(to_light);
		const vec3 L = to_light / distance;

		// Косинус угла между осью прожектора и направлением на точку.
		// Внутри внутреннего угла – полная яркость, за внешним – темно,
		// между ними яркость плавно спадает: это и есть мягкий край.
		const float cos_angle = dot(-L, normalize(light.direction.xyz));
		const float cos_inner = light.cone.x;
		const float cos_outer = light.cone.y;
		const float edge = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);

		const vec3 radiance = light.color.rgb * light.color.a * inverseSquare(distance) * edge;
		color += blinnPhong(N, V, L, radiance);
	}

	out_color = vec4(color, 1.0);
}
