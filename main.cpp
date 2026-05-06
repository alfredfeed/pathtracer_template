#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <map>
#include <string>
#include <fstream>


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323856
#endif

static std::default_random_engine engine[32];
static std::uniform_real_distribution<double> uniform(0, 1);

double sqr(double x) { return x * x; };

void boxMuller(double sigma, double& dx, double& dy) {
	double r1 = uniform(engine[omp_get_thread_num()]);
	double r2 = uniform(engine[omp_get_thread_num()]);
	double r = sigma * sqrt(-2 * log(r1));
	dx = r * cos(2 * M_PI * r2);
	dy = r * sin(2 * M_PI * r2);
}

class Vector {
public:
	explicit Vector(double x = 0, double y = 0, double z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	double norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	double norm() const {
		return sqrt(norm2());
	}
	void normalize() {
		double n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
	}
	double operator[](int i) const { return data[i]; };
	double& operator[](int i) { return data[i]; };
	double data[3];
};

Vector operator+(const Vector& a, const Vector& b) {
	return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
	return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
Vector operator*(const Vector& a, const double b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
Vector operator*(const Vector& a, const Vector& b) {
	return Vector(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
Vector operator/(const Vector& a, const double b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

Vector random_cos(const Vector& N) {
	double r1 = uniform(engine[omp_get_thread_num()]);
	double r2 = uniform(engine[omp_get_thread_num()]);
	double s = sqrt(1 - r2);
	double x = cos(2 * M_PI * r1) * s;
	double y = sin(2 * M_PI * r1) * s;
	double z = sqrt(r2);

	Vector T1;
	if (std::abs(N[0]) <= std::abs(N[1]) && std::abs(N[0]) <= std::abs(N[2]))
		T1 = Vector(0, -N[2], N[1]);
	else if (std::abs(N[1]) <= std::abs(N[0]) && std::abs(N[1]) <= std::abs(N[2]))
		T1 = Vector(-N[2], 0, N[0]);
	else
		T1 = Vector(-N[1], N[0], 0);
	T1.normalize();
	Vector T2 = cross(N, T1);
	return x * T1 + y * T2 + z * N;
}

class Ray {
public:
	Ray(const Vector& origin, const Vector& unit_direction) : O(origin), u(unit_direction) {};
	Vector O, u;
};

class Object {
public:
	Object(const Vector& albedo, bool mirror = false, bool transparent = false) : albedo(albedo), mirror(mirror), transparent(transparent) {};

	virtual bool intersect(const Ray& ray, Vector& P, double& t, Vector& N) const = 0;

	Vector albedo;
	bool mirror, transparent;
};

class Sphere : public Object {
public:
	Sphere(const Vector& center, double radius, const Vector& albedo, bool mirror = false, bool transparent = false) : ::Object(albedo, mirror, transparent), C(center), R(radius) {};

	// returns true iif there is an intersection between the ray and the sphere
	// if there is an intersection, also computes the point of intersection P, 
	// t>=0 the distance between the ray origin and P (i.e., the parameter along the ray)
	// and the unit normal N
	bool intersect(const Ray& ray, Vector& P, double &t, Vector& N) const {
		// DONE (lab 1) : compute the intersection (just true/false at the begining of lab 1, then P, t and N as well)
		// delta = (dot(u, OC))^2 - (||O-C||^2 - R^2)
		// t = dot(u, OC) +/- sqrt(delta)
		Vector OC = ray.O - C;
		double delta = sqr(dot(ray.u, OC)) - (OC.norm2() - sqr(R));
		if (delta < 0) return false;
		double t1 = -dot(ray.u, OC) - sqrt(delta);
		double t2 = -dot(ray.u, OC) + sqrt(delta);
		double eps = 1.0e-3;
		if (t1 > eps) t = t1;
		else if (t2 > eps) t = t2;
		else return false;
		P = ray.O + t * ray.u;
		N = (P - C) / R;
		return true;
	}

	double R;
	Vector C;
};


// Class only used in labs 3 and 4 
class TriangleIndices {
public:
	TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1) {
		vtx[0] = vtxi; vtx[1] = vtxj; vtx[2] = vtxk;
		uv[0] = uvi; uv[1] = uvj; uv[2] = uvk;
		n[0] = ni; n[1] = nj; n[2] = nk;
		this->group = group;
	};
	int vtx[3]; // indices within the vertex coordinates array
	int uv[3];  // indices within the uv coordinates array
	int n[3];   // indices within the normals array
	int group;  // face group
};

// Class only used in labs 3 and 4 

class BVHNode {
public:
	Vector bbox_min, bbox_max;
	BVHNode *left = nullptr, *right = nullptr;
	int start = 0, end = 0;
	~BVHNode() { delete left; delete right; }

	bool intersectBBox(const Ray& ray) const {
		double tx1 = (bbox_min[0] - ray.O[0]) / ray.u[0];
		double tx2 = (bbox_max[0] - ray.O[0]) / ray.u[0];
		double ty1 = (bbox_min[1] - ray.O[1]) / ray.u[1];
		double ty2 = (bbox_max[1] - ray.O[1]) / ray.u[1];
		double tz1 = (bbox_min[2] - ray.O[2]) / ray.u[2];
		double tz2 = (bbox_max[2] - ray.O[2]) / ray.u[2];
		double t_near = std::max({std::min(tx1, tx2), std::min(ty1, ty2), std::min(tz1, tz2)});
		double t_far  = std::min({std::max(tx1, tx2), std::max(ty1, ty2), std::max(tz1, tz2)});
		return t_far >= t_near && t_far >= 0;
	}
};

class TriangleMesh : public Object {
public:
	TriangleMesh(const Vector& albedo, bool mirror = false, bool transparent = false) : ::Object(albedo, mirror, transparent) {};
	~TriangleMesh() { delete root; }

	// first scale and then translate the current object
	void scale_translate(double s, const Vector& t) {
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i] = vertices[i] * s + t;
		}
		buildBVH();
	}

	// read an .obj file
	void readOBJ(const char* obj) {
		std::ifstream f(obj);
		if (!f) return;

		std::map<std::string, int> mtls;
		int curGroup = -1, maxGroup = -1;

		// OBJ indices are 1-based and can be negative (relative), this normalizes them
		auto resolveIdx = [](int i, int size) {
			return i < 0 ? size + i : i - 1;
		};

		auto setFaceVerts = [&](TriangleIndices& t, int i0, int i1, int i2) {
			t.vtx[0] = resolveIdx(i0, vertices.size());
			t.vtx[1] = resolveIdx(i1, vertices.size());
			t.vtx[2] = resolveIdx(i2, vertices.size());
		};
		auto setFaceUVs = [&](TriangleIndices& t, int j0, int j1, int j2) {
			t.uv[0] = resolveIdx(j0, uvs.size());
			t.uv[1] = resolveIdx(j1, uvs.size());
			t.uv[2] = resolveIdx(j2, uvs.size());
		};
		auto setFaceNormals = [&](TriangleIndices& t, int k0, int k1, int k2) {
			t.n[0] = resolveIdx(k0, normals.size());
			t.n[1] = resolveIdx(k1, normals.size());
			t.n[2] = resolveIdx(k2, normals.size());
		};

		std::string line;
		while (std::getline(f, line)) {
			// Trim trailing whitespace
			line.erase(line.find_last_not_of(" \r\t\n") + 1);
			if (line.empty()) continue;

			const char* s = line.c_str();

			if (line.rfind("usemtl ", 0) == 0) {
				std::string matname = line.substr(7);
				auto result = mtls.emplace(matname, maxGroup + 1);
				if (result.second) {
					curGroup = ++maxGroup;
				} else {
					curGroup = result.first->second;
				}
			} else if (line.rfind("vn ", 0) == 0) {
				Vector v;
				sscanf(s, "vn %lf %lf %lf", &v[0], &v[1], &v[2]);
				normals.push_back(v);
			} else if (line.rfind("vt ", 0) == 0) {
				Vector v;
				sscanf(s, "vt %lf %lf", &v[0], &v[1]);
				uvs.push_back(v);
			} else if (line.rfind("v ", 0) == 0) {
				Vector pos, col;
				if (sscanf(s, "v %lf %lf %lf %lf %lf %lf", &pos[0], &pos[1], &pos[2], &col[0], &col[1], &col[2]) == 6) {
					for (int i = 0; i < 3; i++) col[i] = std::min(1.0, std::max(0.0, col[i]));
					vertexcolors.push_back(col);
				} else {
					sscanf(s, "v %lf %lf %lf", &pos[0], &pos[1], &pos[2]);
				}
				vertices.push_back(pos);
			}
			else if (line[0] == 'f') {
				int i[4], j[4], k[4], offset, nn;
				const char* cur = s + 1;
				TriangleIndices t;
				t.group = curGroup;

				// Try each face format: v/vt/vn, v/vt, v//vn, v
				if ((nn = sscanf(cur, "%d/%d/%d %d/%d/%d %d/%d/%d%n", &i[0], &j[0], &k[0], &i[1], &j[1], &k[1], &i[2], &j[2], &k[2], &offset)) == 9) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceUVs(t, j[0], j[1], j[2]); 
					setFaceNormals(t, k[0], k[1], k[2]);
				} else if ((nn = sscanf(cur, "%d/%d %d/%d %d/%d%n", &i[0], &j[0], &i[1], &j[1], &i[2], &j[2], &offset)) == 6) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceUVs(t, j[0], j[1], j[2]);
				} else if ((nn = sscanf(cur, "%d//%d %d//%d %d//%d%n", &i[0], &k[0], &i[1], &k[1], &i[2], &k[2], &offset)) == 6) {
					setFaceVerts(t, i[0], i[1], i[2]); 
					setFaceNormals(t, k[0], k[1], k[2]);
				} else if ((nn = sscanf(cur, "%d %d %d%n", &i[0], &i[1], &i[2], &offset)) == 3) {
					setFaceVerts(t, i[0], i[1], i[2]);
				}
				else continue;

				indices.push_back(t);
				cur += offset;

				// Fan triangulation for polygon faces (4+ vertices)
				while (*cur && *cur != '\n') {
					TriangleIndices t2;
					t2.group = curGroup;
					if ((nn = sscanf(cur, " %d/%d/%d%n", &i[3], &j[3], &k[3], &offset)) == 3) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceUVs(t2, j[0], j[2], j[3]); 
						setFaceNormals(t2, k[0], k[2], k[3]);
					} else if ((nn = sscanf(cur, " %d/%d%n", &i[3], &j[3], &offset)) == 2) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceUVs(t2, j[0], j[2], j[3]);
					} else if ((nn = sscanf(cur, " %d//%d%n", &i[3], &k[3], &offset)) == 2) {
						setFaceVerts(t2, i[0], i[2], i[3]); 
						setFaceNormals(t2, k[0], k[2], k[3]);
					} else if ((nn = sscanf(cur, " %d%n", &i[3], &offset)) == 1) {
						setFaceVerts(t2, i[0], i[2], i[3]);
					} else { 
						cur++; 
						continue; 
					}

					indices.push_back(t2);
					cur += offset;
					i[2] = i[3]; j[2] = j[3]; k[2] = k[3];
				}
			}
		}
		buildBVH();
	}


	void buildBVH() {
		delete root;
		root = buildBVHRecursive(0, (int)indices.size());
	}

	BVHNode* buildBVHRecursive(int start, int end) {
		BVHNode* node = new BVHNode();
		node->start = start;
		node->end = end;

		// bounding
		if (start < end) {
			node->bbox_min = node->bbox_max = vertices[indices[start].vtx[0]];
			for (int i = start; i < end; i++) {
				for (int j = 0; j < 3; j++) {
					const Vector& v = vertices[indices[i].vtx[j]];
					for (int k = 0; k < 3; k++) {
						node->bbox_min[k] = std::min(node->bbox_min[k], v[k]);
						node->bbox_max[k] = std::max(node->bbox_max[k], v[k]);
					}
				}
			}
		}

		const int leaf_threshold = 5;
		if (end - start <= leaf_threshold) return node;

		Vector diag = node->bbox_max - node->bbox_min;
		int axis = 0;
		if (diag[1] > diag[axis]) axis = 1;
		if (diag[2] > diag[axis]) axis = 2;
		double mid = (node->bbox_min[axis] + node->bbox_max[axis]) / 2.0;

		int pivot = start;
		for (int i = start; i < end; i++) {
			const TriangleIndices& tri = indices[i];
			double c = (vertices[tri.vtx[0]][axis] + vertices[tri.vtx[1]][axis] + vertices[tri.vtx[2]][axis]) / 3.0;
			if (c < mid) {
				std::swap(indices[i], indices[pivot]);
				pivot++;
			}
		}

		if (pivot == start || pivot == end) return node;

		node->left = buildBVHRecursive(start, pivot);
		node->right = buildBVHRecursive(pivot, end);
		return node;
	}

	void intersectNode(const BVHNode* node, const Ray& ray, double& closest_t, Vector& best_N, bool& hit) const {
		if (!node || !node->intersectBBox(ray)) return;
		if (!node->left && !node->right) {
			for (int i = node->start; i < node->end; i++) {
				const Vector& A = vertices[indices[i].vtx[0]];
				const Vector& B = vertices[indices[i].vtx[1]];
				const Vector& C = vertices[indices[i].vtx[2]];
				Vector e1 = B - A;
				Vector e2 = C - A;
				Vector triN = cross(e1, e2);
				double denom = dot(ray.u, triN);
				if (std::abs(denom) < 1e-12) continue;
				Vector AmO = A - ray.O;
				Vector AmO_x_u = cross(AmO, ray.u);
				double beta  =  dot(e2, AmO_x_u) / denom;
				double gamma = -dot(e1, AmO_x_u) / denom;
				double alpha = 1.0 - beta - gamma;
				if (alpha < 0 || beta < 0 || gamma < 0) continue;
				double tt = dot(AmO, triN) / denom;
				if (tt < 1e-3) continue;
				if (tt < closest_t) {
					closest_t = tt;
					best_N = triN;
					hit = true;
				}
			}
		} else {
			intersectNode(node->left, ray, closest_t, best_N, hit);
			intersectNode(node->right, ray, closest_t, best_N, hit);
		}
	}

	bool intersect(const Ray& ray, Vector& P, double& t, Vector& N) const {
		if (!root) return false;
		double closest_t = INFINITY;
		Vector best_N;
		bool hit = false;
		intersectNode(root, ray, closest_t, best_N, hit);
		if (hit) {
			t = closest_t;
			P = ray.O + t * ray.u;
			best_N.normalize();
			N = best_N;
		}
		return hit;
	}


	std::vector<TriangleIndices> indices;
	std::vector<Vector> vertices;
	std::vector<Vector> normals;
	std::vector<Vector> uvs;
	std::vector<Vector> vertexcolors;
	BVHNode* root = nullptr;
};


class Scene {
public:
	Scene() {};
	void addObject(const Object* obj) {
		objects.push_back(obj);
	}

	// returns true iif there is an intersection between the ray and any object in the scene
    // if there is an intersection, also computes the point of the *nearest* intersection P, 
    // t>=0 the distance between the ray origin and P (i.e., the parameter along the ray)
    // and the unit normal N. 
	// Also returns the index of the object within the std::vector objects in object_id
	bool intersect(const Ray& ray, Vector& P, double& t, Vector& N, int &object_id) const  {

		// DONE (lab 1): iterate through the objects and check the intersections with all of them, 
		// and keep the closest intersection, i.e., the one if smallest positive value of t

		double closest_t = INFINITY;
		Vector closest_P, closest_N;
		for (int i = 0; i < objects.size(); i++) {
			if (objects[i]->intersect(ray, P, t, N)) {
				if (t < closest_t) {
					closest_t = t;
					object_id = i;
					closest_P = P;
					closest_N = N;
				}
			}
		}
		P = closest_P;
		N = closest_N;
		t = closest_t;
		return closest_t != INFINITY;
		
	}


	// return the radiance (color) along ray
	Vector getColor(const Ray& ray, int recursion_depth) {

		if (recursion_depth >= max_light_bounce) return Vector(0, 0, 0);

		// DONE (lab 1) : if intersect with ray, use the returned information to compute the color ; otherwise black 
		// in lab 1, the color only includes direct lighting with shadows		

		Vector P, N;
		double t;
		int object_id;
		if (intersect(ray, P, t, N, object_id)) {

			if (objects[object_id]->mirror) {
				Vector r = ray.u - 2 * dot(ray.u, N) * N;
      			return getColor(Ray(P + 0.001 * N, r), recursion_depth + 1);
				// return getColor in the reflected direction, with recursion_depth+1 (recursively)
			} // else

			if (objects[object_id]->transparent) { // optional

				// return getColor in the refraction direction, with recursion_depth+1 (recursively)
			} // else

			Vector l = light_position - P;
			double d2 = l.norm2();
			l = l / sqrt(d2);
			Vector P_off = P + 0.001 * N;
			Vector direct(0, 0, 0);
			Vector sP, sN; double st; int sid;
			if (!(intersect(Ray(P_off, l), sP, st, sN, sid) && st < sqrt(d2))) {
				double cos_t = std::max(0., dot(N, l));
				direct = (light_intensity / (4 * M_PI * d2)) * (objects[object_id]->albedo / M_PI) * cos_t;
			}

			Vector wi = random_cos(N);
			Vector indirect = objects[object_id]->albedo * getColor(Ray(P_off, wi), recursion_depth + 1);
			return direct + indirect;
		}

		

		return Vector(0, 0, 0);
	}

	std::vector<const Object*> objects;

	Vector camera_center, light_position;
	double fov, gamma, light_intensity;
	int max_light_bounce;
};


int main() {
	int W = 512;
	int H = 512;

	for (int i = 0; i<32; i++) {
		engine[i].seed(i);
	}

	TriangleMesh cat(Vector(0.3, 0.3, 0.3));
	cat.readOBJ("cat.obj");
	cat.scale_translate(0.3, Vector(0, -10, 0));

	Sphere wall_left(Vector(-1000, 0, 0), 940, Vector(0.5, 0.8, 0.1));
	Sphere wall_right(Vector(1000, 0, 0), 940, Vector(0.9, 0.2, 0.3));
	Sphere wall_front(Vector(0, 0, -1000), 940, Vector(0.1, 0.6, 0.7));
	Sphere wall_behind(Vector(0, 0, 1000), 940, Vector(0.8, 0.2, 0.9));
	Sphere ceiling(Vector(0, 1000, 0), 940, Vector(0.3, 0.5, 0.3));
	Sphere floor(Vector(0, -1000, 0), 990, Vector(0.6, 0.5, 0.7));

	Scene scene;
	scene.camera_center = Vector(0, 0, 55);
	scene.light_position = Vector(-10,20,40);
	scene.light_intensity = 3E7;
	scene.fov = 60 * M_PI / 180.;
	scene.gamma = 2.2;    // DONE (lab 1) : play with gamma ; typically, gamma = 2.2
	scene.max_light_bounce = 5;

	scene.addObject(&cat);
	scene.addObject(&wall_left);
	scene.addObject(&wall_right);
	scene.addObject(&wall_front);
	scene.addObject(&wall_behind);
	scene.addObject(&ceiling);
	scene.addObject(&floor);

	std::vector<unsigned char> image(W * H * 3, 0);

	int nb_samples = 2048;
	double aperture = 0.3;
	double focus_distance = 60.0;

#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			Vector color(0, 0, 0);

			for (int s = 0; s < nb_samples; s++) {
				double dx, dy;
				boxMuller(0.5, dx, dy);
				Vector dir(
					j - W / 2.0 + 0.5 + dx,
					-(i - H / 2.0 + 0.5 + dy),
					-W / (2.0 * tan(scene.fov / 2.0))
				);
				dir.normalize();

				double t_focus = focus_distance / std::abs(dir[2]);
				Vector focusPoint = scene.camera_center + t_focus * dir;
				double du, dv;
				boxMuller(aperture, du, dv);
				Vector origin = scene.camera_center + Vector(du, dv, 0);
				Vector newDir = focusPoint - origin;
				newDir.normalize();

				color = color + scene.getColor(Ray(origin, newDir), 0);
			}
			color = color / nb_samples;

			image[(i * W + j) * 3 + 0] = std::min(255., std::max(0., 255. * std::pow(color[0] / 255., 1. / scene.gamma)));
			image[(i * W + j) * 3 + 1] = std::min(255., std::max(0., 255. * std::pow(color[1] / 255., 1. / scene.gamma)));
			image[(i * W + j) * 3 + 2] = std::min(255., std::max(0., 255. * std::pow(color[2] / 255., 1. / scene.gamma)));
		}
	}
	stbi_write_png("image.png", W, H, 3, &image[0], 0);

	return 0;
}