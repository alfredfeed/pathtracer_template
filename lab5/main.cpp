#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"



int main() {

	int W, H, C;
	
	//stbi_set_flip_vertically_on_load(true);
	unsigned char *image = stbi_load("8733654151_b9422bb2ec_k.jpg",
                                 &W,
                                 &H,
                                 &C,
                                 STBI_rgb);

	int Wm, Hm, Cm;
	unsigned char *model = stbi_load("redim.jpg", &Wm, &Hm, &Cm, STBI_rgb);

	std::vector<double> image_double(W*H*3);
	for (int i=0; i<W*H*3; i++)
		image_double[i] = image[i];

	std::vector<double> model_double(W*H*3);
	for (int i=0; i<W*H*3; i++)
		model_double[i] = model[i];

	int N = W*H;
	for (int iter=0; iter<100; iter++) {

		double vx = rand()/(double)RAND_MAX - 0.5;
		double vy = rand()/(double)RAND_MAX - 0.5;
		double vz = rand()/(double)RAND_MAX - 0.5;
		double len = sqrt(vx*vx + vy*vy + vz*vz);
		vx/=len; vy/=len; vz/=len;

		std::vector<std::pair<double,int> > pI(N);
		std::vector<double> pM(N);
		for (int i=0; i<N; i++) {
			pI[i] = std::make_pair(image_double[i*3]*vx + image_double[i*3+1]*vy + image_double[i*3+2]*vz, i);
			pM[i] = model_double[i*3]*vx + model_double[i*3+1]*vy + model_double[i*3+2]*vz;
		}
		std::sort(pI.begin(), pI.end());
		std::sort(pM.begin(), pM.end());

		for (int k=0; k<N; k++) {
			int id = pI[k].second;
			double d = pM[k] - pI[k].first;
			image_double[id*3+0] += d*vx;
			image_double[id*3+1] += d*vy;
			image_double[id*3+2] += d*vz;
		}
	}

	for (int i=0; i<W*H*3; i++) {
		if (image_double[i] < 0) image_double[i] = 0;
		if (image_double[i] > 255) image_double[i] = 255;
	}

	std::vector<unsigned char> image_result(W*H * 3, 0);
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {

			image_result[(i*W + j) * 3 + 0] = image_double[(i*W+j)*3+0];
			image_result[(i*W + j) * 3 + 1] = image_double[(i*W+j)*3+1];
			image_result[(i*W + j) * 3 + 2] = image_double[(i*W+j)*3+2];
		}
	}
	stbi_write_png("image.png", W, H, 3, &image_result[0], 0);

	return 0;
}
