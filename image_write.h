#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

void generate_image(int* result){
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
}
