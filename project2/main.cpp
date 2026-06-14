#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include <sstream>

#include <vector>
#include <random>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "lbfgs.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double sqr(double x) { return x * x; };

class Vector {
public:
    explicit Vector(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    double data[2];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1];
}


class Polygon {
public:

    double area() {
        if (vertices.size() < 3) return 0;
        // DONE Lab 2
        // Compute the area of the polygon
        double a = 0;
        int n = (int)vertices.size();
        for (int i = 0; i < n; i++) {
            const Vector& A = vertices[i];
            const Vector& B = vertices[(i + 1) % n];
            a += A[0] * B[1] - B[0] * A[1];
        }
        return std::abs(a) * 0.5;
    }

    Vector centroid() {
        if (vertices.size() < 3) return Vector(0, 0);
        // DONE Lab 3
        // Compute the centroid of the polygon
        double cx = 0, cy = 0, a = 0;
        int n = (int)vertices.size();
        for (int i = 0; i < n; i++) {
            const Vector& A = vertices[i];
            const Vector& B = vertices[(i + 1) % n];
            double cross = A[0] * B[1] - B[0] * A[1];
            cx += (A[0] + B[0]) * cross;
            cy += (A[1] + B[1]) * cross;
            a += cross;
        }
        a *= 0.5;
        if (std::abs(a) < 1e-12) return vertices[0];
        return Vector(cx / (6 * a), cy / (6 * a));
    }

    double integral_square_distance(const Vector& Pi) {
        if (vertices.size() < 3) return 0;

        // DONE Lab 2
        // Compute the integral of ||x-Pi||^2 over the polygon

        double total = 0;
        int n = (int)vertices.size();
        for (int j = 1; j < n - 1; j++) {
            Vector c[3] = { vertices[0], vertices[j], vertices[j + 1] };
            double areaT = 0.5 * std::abs((c[1][0] - c[0][0]) * (c[2][1] - c[0][1])
                                        - (c[1][1] - c[0][1]) * (c[2][0] - c[0][0]));
            double sum = 0;
            for (int k = 0; k < 3; k++)
                for (int l = k; l < 3; l++)
                    sum += dot(c[k] - Pi, c[l] - Pi);
            total += areaT / 6.0 * sum;
        }
        return total;
    }

    std::vector<Vector> vertices;
};


void save_frame(const std::vector<Polygon>& cells, std::string filename, int frameid = 0) {
    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;

    std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto& V = cells[i].vertices;
        const int n = (int)V.size();
        if (n < 3) continue;

        std::vector<double> xs(n), ys(n);
        double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
        for (int j = 0; j < n; ++j) {
            xs[j] = V[j][0] * W;
            ys[j] = V[j][1] * H;
            xmin = std::min(xmin, xs[j]);
            ymin = std::min(ymin, ys[j]);
            xmax = std::max(xmax, xs[j]);
            ymax = std::max(ymax, ys[j]);
        }

        int x0 = std::max(0, (int)std::floor(xmin - edge_width));
        int y0 = std::max(0, (int)std::floor(ymin - edge_width));
        int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
        int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;

                int prev_sign = 0;
                bool isInside = true;
                bool isEdge = false;

                for (int j = 0; j < n; ++j) {
                    int k = (j + 1) % n;

                    double ax = xs[j], ay = ys[j];
                    double bx = xs[k], by = ys[k];
                    double dx = bx - ax, dy = by - ay;
                    double qx = px - ax, qy = py - ay;

                    double det = qx * dy - qy * dx;
                    int s = (det > 1e-12) - (det < -1e-12);

                    if (s != 0) {
                        if (prev_sign != 0 && s != prev_sign) {
                            isInside = false;
                            break;
                        }
                        prev_sign = s;
                    }

                    double len2 = dx * dx + dy * dy;
                    double dot = qx * dx + qy * dy;
                    if (dot >= 0.0 && dot <= len2 && det * det <= edge_width2 * len2)
                        isEdge = true;
                }

                if (isInside) {
                    int id = (H - 1 - y) * W + x;
                    inside[id] = 1;
                    if (isEdge) edge[id] = 1;
                }
            }
        }
    }

    std::vector<unsigned char> image(W * H * 3, 255);

#pragma omp parallel for
    for (int i = 0; i < W * H; ++i) {
        if (edge[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 0;
        }
        else if (inside[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}


// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon>& polygons, std::string filename, const std::vector<Vector>* points = NULL, std::string fillcol = "none") {
    FILE* f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }

    if (points) {
        fprintf(f, "<g>\n");
        for (int i = 0; i < points->size(); i++) {
            fprintf(f, "<circle cx = \"%3.3f\" cy = \"%3.3f\" r = \"3\" />\n", (*points)[i][0] * 1000., 1000. - (*points)[i][1] * 1000);
        }
        fprintf(f, "</g>\n");

    }

    fprintf(f, "</svg>\n");
    fclose(f);
}


class VoronoiDiagram {

public:

    VoronoiDiagram() {
    };


    void compute() {

        // DONE Lab 1 (Voronoi)
        // For all sites Pi (in parallel) :
        //      Start with a unit square
        //      For all other sites Pj (optionally, only k nearest neighbors) :
        //          Clip it with bisector of [Pi,Pj]
        //      (Lab 3, fluids) : also clip it by a disk of radius sqrt(w_i - w_air) centered at Pi
        Polygon square;
        square.vertices.push_back(Vector(0, 0));
        square.vertices.push_back(Vector(1, 0));
        square.vertices.push_back(Vector(1, 1));
        square.vertices.push_back(Vector(0, 1));

        cells.resize(points.size());

        bool has_air = (weights.size() == points.size() + 1);
        double w_air = has_air ? weights.back() : 0.0;

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)points.size(); i++) {
            Polygon cell = square;
            double wi = weights.empty() ? 0.0 : weights[i];
            for (int j = 0; j < (int)points.size(); j++) {
                if (i == j) continue;
                double wj = weights.empty() ? 0.0 : weights[j];
                cell = clip_by_bisector(cell, points[i], points[j], wi, wj);
            }

            if (has_air) {
                double R2 = wi - w_air;
                if (R2 <= 0) {
                    // edgecase
                    cell.vertices.clear();
                } else {
                    double R = sqrt(R2);
                    const int K = 50;
                    for (int k = 0; k < K; k++) {
                        double a0 = 2.0 * M_PI * k / K;
                        double a1 = 2.0 * M_PI * (k + 1) / K;
                        Vector edge_u = points[i] + Vector(R * cos(a0), R * sin(a0));
                        Vector edge_v = points[i] + Vector(R * cos(a1), R * sin(a1));
                        cell = clip_by_edge(cell, edge_u, edge_v);
                    }
                }
            }

            cells[i] = cell;
        }
    }


    static Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {

        // DONE Lab 3 (fluids)
        // Clip a polygon by an edge defined by vertices u and v
        // Will be used to clip a polygon (a cell) by all the edges of a (discretized) disk

        Vector edge = v - u;
        Vector N(edge[1], -edge[0]);

        Polygon result;
        int n = (int)V.vertices.size();
        for (int i = 0; i < n; i++) {
            const Vector& B = V.vertices[i];
            const Vector& A = V.vertices[(i > 0) ? (i - 1) : (n - 1)];

            bool B_inside = dot(u - B, N) >= 0;
            bool A_inside = dot(u - A, N) >= 0;

            if (B_inside) {
                if (!A_inside) {
                    double t = dot(u - A, N) / dot(B - A, N);
                    result.vertices.push_back(A + t * (B - A));
                }
                result.vertices.push_back(B);
            } else if (A_inside) {
                double t = dot(u - A, N) / dot(B - A, N);
                result.vertices.push_back(A + t * (B - A));
            }
        }

        return result;
    }

    static Polygon clip_by_bisector(const Polygon& V, const Vector& P0, const Vector& Pi, double w0, double wi) {

        // DONE Lab 1 (Voronoi) : in Lab 1, we assume w0 = w1 = 0
        // Clip a polygon by the bisector of the segment defined by P0 (the current site of the Voronoi cell being computed) and Pi (another site)
        
        // DONE Lab 2 (Semi-Discrete Optimal Transport) : extend to Laguerre cells, i.e., w0 != w1

        Vector dir = Pi - P0;
        Vector M = (P0 + Pi) * 0.5 + ((w0 - wi) / (2.0 * dir.norm2())) * dir;

        Polygon result;
        int n = (int)V.vertices.size();
        for (int i = 0; i < n; i++) {
            const Vector& B = V.vertices[i];
            const Vector& A = V.vertices[(i > 0) ? (i - 1) : (n - 1)];

            bool B_inside = dot(B - M, dir) < 0;
            bool A_inside = dot(A - M, dir) < 0;

            if (B_inside) {
                if (!A_inside) {
                    double t = dot(M - A, dir) / dot(B - A, dir);
                    result.vertices.push_back(A + t * (B - A));
                }
                result.vertices.push_back(B);
            } else if (A_inside) {
                double t = dot(M - A, dir) / dot(B - A, dir);
                result.vertices.push_back(A + t * (B - A));
            }
        }

        return result;
    }


    std::vector<Vector> points;    // Lab 1 (Voronoi) : the sites to consider

    std::vector<double> weights;   // Lab 2 (OT) : the weight associated to each site (the Laguerre weight, i.e. the dual optimal transport variables to be optimized)
    
    std::vector<Polygon> cells;   // Lab 1 : the polygons representing each individual cell

};


// Lab 2 
class OptimalTransport {

public:
    OptimalTransport() {};

    void optimize();

    VoronoiDiagram vor;

    bool fluid_mode = false;
    double fluid_volume = 1.0;
};


// Labs 2 and 3
static lbfgsfloatval_t evaluate(
    void* instance,
    const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step
)
{
    OptimalTransport* ot = (OptimalTransport*)(instance);

    // first compute the Voronoi diagram at the current optimization step
    memcpy(&ot->vor.weights[0], x, n * sizeof(x[0]));
    ot->vor.compute();
  
   
    // Lab 2 (Optimal transport) : compute the function to be minimized (fx) and its gradient (g[i], i=0..n-1)
    // Lab 3 (fluid) : adapt these functions to support partial optimal transport (now "n" has been increased by 1 to account for the air variable)

    lbfgsfloatval_t fx = 0.0;

    if (ot->fluid_mode) {
        int Nf = n - 1;
        double vol_air = 1.0 - ot->fluid_volume;
        double lambda = ot->fluid_volume / Nf;
        double estimated_fluid = 0.0;
        for (int i = 0; i < Nf; i++) {
            double A_i = ot->vor.cells[i].area();
            double I_i = ot->vor.cells[i].integral_square_distance(ot->vor.points[i]);
            fx += -I_i + x[i] * A_i - lambda * x[i];
            g[i] = A_i - lambda;
            estimated_fluid += A_i;
        }
        double estimated_air = 1.0 - estimated_fluid;
        fx += -x[Nf] * (vol_air - estimated_air);
        g[Nf] = estimated_air - vol_air;
        return fx;
    }

    double lambda = 1.0 / n;
    for (int i = 0; i < n; i++) {
        double A_i = ot->vor.cells[i].area();
        double I_i = ot->vor.cells[i].integral_square_distance(ot->vor.points[i]);
        fx += -I_i + x[i] * A_i - lambda * x[i];
        g[i] = A_i - lambda;
    }

    return fx;
}

// Labs 2 and 3 : you may use this function to print debugging info.
static int progress(
    void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
    int n, int k, int ls) {
    if (((OptimalTransport*)instance)->fluid_mode) return 0;
    printf("Iteration %d:\n", k);
    printf("  fx = %f\n", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}


// Lab 2
void OptimalTransport::optimize() {

    lbfgsfloatval_t fx;
    std::vector<double> weights(vor.weights);

    lbfgs_parameter_t param;
    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&param);

    // run the LBFGS optimizer
    int ret = lbfgs(weights.size(), &weights[0], &fx, evaluate, progress, (void*)this, &param);

    // copy the result back to the voronoi structure
    vor.weights = weights;

    // finally recompute the Voronoi diagram with the final optimized weights
    vor.compute();
}


// Lab 3 (fluids)
class Fluid {
public:
    Fluid(int N_particles = 1000) : N_particles(N_particles) {
    }

    // Lab 3 : advance the simulation dt in time
    void time_step(double dt) {

        double epsilon2 = 0.004 * 0.004;
        Vector g(0, -9.81);
        double m_i = 200;

        // DONE Lab 3 :
        // Compute semi-discrete partial optimal transport
        // for all particles, add gravity and spring force towards cell centroid, integrate acceleration->velocity and velocity->position

        ot.vor.points = particles;
        double w0 = fluid_volume / (N_particles * M_PI);
        ot.vor.weights.assign(N_particles + 1, 0.0);
        for (int i = 0; i < N_particles; i++) ot.vor.weights[i] = w0;
        ot.optimize();

        // I added this to make energy disappear, because the water was bouncing endlessly
        double restitution = use_damping ? 0.8 : 1.0;
        double viscosity = use_damping ? 0.997 : 1.0;

        for (int i = 0; i < N_particles; i++) {
            Vector C = (ot.vor.cells[i].vertices.size() >= 3) ? ot.vor.cells[i].centroid() : particles[i];
            Vector F_spring = (1.0 / epsilon2) * (C - particles[i]);
            Vector F = F_spring + m_i * g;
            velocities[i] = (velocities[i] + (dt / m_i) * F) * viscosity;
            particles[i] = particles[i] + dt * velocities[i];

            for (int d = 0; d < 2; d++) {
                if (particles[i][d] < 0)      { particles[i][d] = -particles[i][d];     velocities[i][d] = -restitution * velocities[i][d]; }
                else if (particles[i][d] > 1) { particles[i][d] = 2.0 - particles[i][d]; velocities[i][d] = -restitution * velocities[i][d]; }
            }
        }
    }

    // just run the full simulation
    void run_simulation() {

        fluid_volume = 0.45;

        particles.resize(N_particles);
        velocities.resize(N_particles, Vector(0, 0));

        std::default_random_engine engine(42);
        std::uniform_real_distribution<double> jitter(-0.2, 0.2);
        double L = sqrt(fluid_volume);
        int nc = (int)ceil(sqrt((double)N_particles));
        double spacing = L / nc;
        double x0 = 0.5 - L / 2;
        double y0 = 0.98 - L;
        for (int i = 0; i < N_particles; i++) {
            int r = i / nc, c = i % nc;
            double x = x0 + (c + 0.5 + jitter(engine)) * spacing;
            double y = y0 + (r + 0.5 + jitter(engine)) * spacing;
            particles[i] = Vector(x, y);
        }

        ot.fluid_mode = true;
        ot.fluid_volume = fluid_volume;
        ot.vor.points = particles;

        ot.vor.weights.assign(N_particles + 1, 0.0);
        double w0 = fluid_volume / (N_particles * M_PI);
        for (int i = 0; i < N_particles; i++) ot.vor.weights[i] = w0;

        double dt = 0.002;
        for (int i = 0; i < 2000; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "fluid", i);
            printf("frame %d done\n", i);
        }
    }

    int N_particles;
    bool use_damping = false; // set false for the original elastic fluid

    OptimalTransport ot;
    std::vector<Vector> particles;  // the position of all particles
    std::vector<Vector> velocities; // the velocities of all particles
    double fluid_volume; // you decide the fraction of the unit square occupied by the fluid
};








int main() {

    Fluid fluid(300);
    fluid.run_simulation();

    return 0;
}