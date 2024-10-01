#include <iostream>
#include <math.h>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// I have 3584 cores on my GPU (nvidia-settings -q CUDACores -t)
static const constexpr int NUM_CORES = 3584;
static const constexpr int EDGE = 256;
static const constexpr double XMIN = -2.0;
static const constexpr double YMAX = 1.25;
static const constexpr double WIDTH = 2.5;
static const constexpr int N = EDGE * EDGE;

uint16_t vals[EDGE * EDGE];
uint32_t palette[1000];
uint32_t img[EDGE * EDGE];

#define RGB(r,g,b) \
	(((r) & 0xff) | (((g) & 0xff) << 8) | (((b) & 0xff) << 16) | \
	 0xff000000)

void tight_palette(void)
{
	int i;
	for (i = 0; i < 32; ++i) {
		palette[i] = RGB(0,0,64+i*4);
	}
	for (i = 0; i < 64; ++i) {
		palette[i+ 32] = RGB(i*3,0,192);
		palette[i+ 96] = RGB(192+i,0,192-i*3);
		palette[i+160] = RGB(255,i*4,0);
		palette[i+224] = RGB(255,255,i*4);
	}
	for (i = 280; i < 1000; ++i) {
		palette[i] = RGB(255,255,255);
	}
}

__global__ void mandelbrot_gpu(double xmin, double ymax, double width, int edge, int *output){
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = global_index;
    int out_index = iy * edge;

    double y = ymax - iy * (width / edge);
    for (int ix = 0; ix < edge; ++ix) {
        double x = xmin + ix * (width / edge);
        double a = 0.0;
        double b = 0.0;
        int n = 0;
        for (n = 0; n < 1000; ++n) {
            double a2 = a*a, b2 = b*b;
            if (a2 + b2 > 4.0) {
                break;
            }
            b = 2 * a * b + y;
            a = a2 + x - b2;
        }
        output[out_index++] = n;
  }
}

__global__ void test(int* output, int edge){
    int global_index = threadIdx.x + (blockDim.x * blockIdx.x);
    int iy = global_index;
    int out_index = iy * edge;

    for (int ix = 0; ix < edge; ++ix) {
        output[out_index++] = out_index;
  }
}

__global__ void mandelbrot_gpu_single(double xmin, double ymax, double width, int edge, int *output)
{
	int ix, iy, out_index;
	out_index = 0;
	for (iy = 0; iy < edge; ++iy) {
		double y = ymax - iy * (width / edge);
		for (ix = 0; ix < edge; ++ix) {
			double x = xmin + ix * (width / edge);
			double a = 0.0;
			double b = 0.0;
			uint16_t n;
			for (n = 0; n < 1000; ++n) {
				double a2 = a*a, b2 = b*b;
				if (a2 + b2 > 4.0) {
					break;
				}
				b = 2 * a * b + y;
				a = a2 + x - b2;
			}
			output[out_index++] = n;
		}
	}
}

void mandelbrot_ref(double xmin, double ymax, double width, int edge, uint16_t *output)
{
	int ix, iy, out_index;
	out_index = 0;
	for (iy = 0; iy < edge; ++iy) {
		double y = ymax - iy * (width / edge);
		for (ix = 0; ix < edge; ++ix) {
			double x = xmin + ix * (width / edge);
			double a = 0.0;
			double b = 0.0;
			uint16_t n = 0;
			for (n = 0; n < 1000; ++n) {
				double a2 = a*a, b2 = b*b;
				if (a2 + b2 > 4.0) {
					break;
				}
				b = 2 * a * b + y;
				a = a2 + x - b2;
			}
			output[out_index++] = n;
		}
	}
}

uint16_t result_ref[N];

int main(){
    int* result;
    const int blockSize = 256;
    const int blockCount = EDGE / 256;
    cudaMallocManaged(&result, N*sizeof(int));

    // compute with CUDA
    #if 1
    auto now = []{
        return std::chrono::system_clock::now();
    };
    auto start = now();

    mandelbrot_gpu_single<<<1,1>>>(XMIN, YMAX, WIDTH, EDGE, result);
    cudaDeviceSynchronize();

    auto cudaTime = std::chrono::duration_cast<std::chrono::milliseconds>((now() - start));
    std:: cout << "Finished cuda calculation. Took " << cudaTime.count() << " ms." << std::endl;

    // double check against ref impl
    
    mandelbrot_ref(XMIN, YMAX, WIDTH, EDGE, result_ref);

    auto refTime = (std::chrono::duration_cast<std::chrono::milliseconds>((now() - start)) - cudaTime);
    std:: cout << "Finished ref calculation. Took " << refTime.count() << " ms." << std::endl;

    int error_cnt = 0;
    for(int i = 0; i < N; i++){
        if(result[i] != result_ref[i]){            
            if(++error_cnt < 100){
                std::cout << "result[" << i << "] AKA " << "result[" << i / EDGE << ", " << i % EDGE << "]= " 
                        << result[i] << ", " << "result_ref = " << result_ref[i] << "\n";
            }
        }
    }

    std::cout << "total errors: " << error_cnt << " out of " << N << "\n";
    #endif

    tight_palette();
	for (int i = 0; i < EDGE * EDGE; ++i) {
		if (result[i] >= 1000) {
			img[i] = 0xff000000;
		} else {
			int v = result[i] * 3;
			if (v > 999) { v = 999; }
			img[i] = palette[v];
		}
	}
	stbi_write_png("mandelbrot.png", EDGE, EDGE, 4, img, EDGE*4);
    

    cudaFree(result);

    return 0;
}



