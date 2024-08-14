#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MASK_WIDTH 15

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

void check_cuda(cudaError_t error, const char *filename, const int line)
{
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d: %s: %s\n", filename, line,
                 cudaGetErrorName(error), cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

typedef struct {
  unsigned char red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n')
    ;
  img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

void writePPM(PPMImage *img) {

  fprintf(stdout, "P6\n");
  fprintf(stdout, "# %s\n", COMMENT);
  fprintf(stdout, "%d %d\n", img->x, img->y);
  fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

  fwrite(img->data, 3 * img->x, img->y, stdout);
  fclose(stdout);
}

// Implement this!
__global__ void smoothing_kernel(const PPMPixel *input, PPMPixel *output, int width, int height) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < height && j < width) {
    int total_red = 0, total_blue = 0, total_green = 0;

    for (int y = i - ((MASK_WIDTH - 1) / 2); y <= (i + ((MASK_WIDTH - 1) / 2)); y++) {
      for (int x = j - ((MASK_WIDTH - 1) / 2); x <= (j + ((MASK_WIDTH - 1) / 2)); x++) {
        if (x >= 0 && y >= 0 && y < height && x < width) {
          total_red += input[(y * width) + x].red;
          total_blue += input[(y * width) + x].blue;
          total_green += input[(y * width) + x].green;
        }
      }
    }

    output[(i * width) + j].red = total_red / (MASK_WIDTH * MASK_WIDTH);
    output[(i * width) + j].blue = total_blue / (MASK_WIDTH * MASK_WIDTH);
    output[(i * width) + j].green = total_green / (MASK_WIDTH * MASK_WIDTH);
  }
}

void Smoothing(PPMImage *image, PPMImage *image_copy, PPMPixel *d_input, PPMPixel *d_output) {
  dim3 dimBlock(16, 16);
  dim3 dimSize((image->x + dimBlock.x - 1) / dimBlock.x, (image->y + dimBlock.y - 1) / dimBlock.y);

  smoothing_kernel<<<dimSize, dimBlock>>>(d_input, d_output, image->x, image->y);
}

int main(int argc, char *argv[]) {
  FILE *input;
  char filename[255];
  double t;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open input file!\n");
    return 1;
  }

  // Read input filename
  fscanf(input, "%s\n", filename);

  // Read input file
  PPMImage *image = readPPM(filename);
  PPMImage *image_output = readPPM(filename);

  // Allocate pointers
  PPMPixel *d_input, *d_output;
  size_t imageSize = image->x * image->y * sizeof(PPMPixel);

  // Allocate device memory
  cudaMalloc((void **)&d_input, imageSize);
  cudaMalloc((void **)&d_output, imageSize);

  // Copy data to device
  cudaMemcpy(d_input, image->data, imageSize, cudaMemcpyHostToDevice);

  // Call Smoothing Kernel
  t = omp_get_wtime();
  Smoothing(image_output, image, d_input, d_output);
  t = omp_get_wtime() - t;

  // Copy data back to host
  cudaMemcpy(image_output->data, d_output, imageSize, cudaMemcpyDeviceToHost);

  // Write result to stdout
  writePPM(image_output);

  // Print time to stderr
  fprintf(stderr, "%lf\n", t);

  // Cleanup
  free(image);
  free(image_output);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
