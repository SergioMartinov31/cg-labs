#pragma once

#include <cmath>

// Минимальная библиотека линейной алгебры для лабораторной.
// Матрицы хранятся по столбцам (column-major), как mat4 в GLSL,
// поэтому их можно копировать в uniform-буфер без транспонирования.

namespace math {

constexpr float pi = 3.14159265358979323846f;

inline float radians(float degrees) {
	return degrees * pi / 180.0f;
}

struct Vec3 {
	float x, y, z;
};

struct Mat4 {
	float m[16]; // m[column * 4 + row]

	float& at(int row, int column) { return m[column * 4 + row]; }
	float at(int row, int column) const { return m[column * 4 + row]; }

	static Mat4 identity() {
		Mat4 r{};
		r.at(0, 0) = r.at(1, 1) = r.at(2, 2) = r.at(3, 3) = 1.0f;
		return r;
	}
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r{};
	for (int row = 0; row < 4; ++row) {
		for (int column = 0; column < 4; ++column) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.at(row, k) * b.at(k, column);
			}
			r.at(row, column) = sum;
		}
	}
	return r;
}

// Перемещение: в однородных координатах сдвиг лежит в последнем столбце
inline Mat4 translate(Vec3 t) {
	Mat4 r = Mat4::identity();
	r.at(0, 3) = t.x;
	r.at(1, 3) = t.y;
	r.at(2, 3) = t.z;
	return r;
}

// Растяжение по осям
inline Mat4 scale(Vec3 s) {
	Mat4 r = Mat4::identity();
	r.at(0, 0) = s.x;
	r.at(1, 1) = s.y;
	r.at(2, 2) = s.z;
	return r;
}

// Повороты вокруг осей (угол в радианах, правая система координат)
inline Mat4 rotateX(float a) {
	Mat4 r = Mat4::identity();
	const float c = std::cos(a), s = std::sin(a);
	r.at(1, 1) = c; r.at(1, 2) = -s;
	r.at(2, 1) = s; r.at(2, 2) = c;
	return r;
}

inline Mat4 rotateY(float a) {
	Mat4 r = Mat4::identity();
	const float c = std::cos(a), s = std::sin(a);
	r.at(0, 0) = c;  r.at(0, 2) = s;
	r.at(2, 0) = -s; r.at(2, 2) = c;
	return r;
}

inline Mat4 rotateZ(float a) {
	Mat4 r = Mat4::identity();
	const float c = std::cos(a), s = std::sin(a);
	r.at(0, 0) = c; r.at(0, 1) = -s;
	r.at(1, 0) = s; r.at(1, 1) = c;
	return r;
}

// Перспективная проекция под Vulkan:
// камера смотрит вдоль -Z, глубина отображается в [0, 1],
// ось Y в clip space Vulkan направлена вниз, поэтому берём -f.
inline Mat4 perspective(float fov_y, float aspect, float near, float far) {
	const float f = 1.0f / std::tan(fov_y * 0.5f);

	Mat4 r{};
	r.at(0, 0) = f / aspect;
	r.at(1, 1) = -f;
	r.at(2, 2) = far / (near - far);
	r.at(2, 3) = near * far / (near - far);
	r.at(3, 2) = -1.0f; // w' = -z, затем GPU делит x, y, z на w
	return r;
}

// Ортографическая проекция: композиция перемещения и растяжения,
// которая переводит параллелепипед [l, r] x [b, t] x [-near, -far]
// в канонический объём Vulkan [-1, 1] x [-1, 1] x [0, 1].
inline Mat4 orthographic(float left, float right, float bottom, float top,
                         float near, float far) {
	Mat4 r = Mat4::identity();
	r.at(0, 0) = 2.0f / (right - left);
	r.at(1, 1) = -2.0f / (top - bottom); // переворот Y под Vulkan
	r.at(2, 2) = -1.0f / (far - near);
	r.at(0, 3) = -(right + left) / (right - left);
	r.at(1, 3) = (top + bottom) / (top - bottom);
	r.at(2, 3) = -near / (far - near);
	return r;
}

} // namespace math
