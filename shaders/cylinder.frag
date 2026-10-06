#version 450

// Должен зеркалить тип и location выхода вершинного шейдера
layout(location = 0) in vec3 in_color;

// Цвет пикселя в первое (и единственное) цветное изображение
layout(location = 0) out vec4 out_color;

void main() {
	out_color = vec4(in_color, 1.0);
}
