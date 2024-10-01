#include <iostream>
#include <math.h>
#include <chrono>

// I have 3584 cores on my GPU (nvidia-settings -q CUDACores -t)
// static const constexpr int NUM_CORES = 3584;
static const constexpr int EDGE = 3584;
static const constexpr double XMIN = -2.0;
static const constexpr double YMAX = 1.25;
static const constexpr double WIDTH = 2.5;
static const constexpr int N = EDGE * EDGE;

#include "image_write.h"

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
    auto now = []{
        return std::chrono::system_clock::now();
    };
    auto start = now();

    mandelbrot_gpu<<<blockCount,blockSize>>>(XMIN, YMAX, WIDTH, EDGE, result);
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

    generate_image(result);    

    cudaFree(result);

    return 0;
}



