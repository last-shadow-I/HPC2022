#include "EasyBMP.h"
#include <iostream>
#include <vector>
#include <algorithm> 
#include <string>
#include <iomanip>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

int MedianFilter(vector<vector<int>> image, int verctical, int gorizontal) {
	vector<int> array;

	for (int i = verctical - 1; i <= verctical + 1; i++)
	{
		for (int j = gorizontal - 1; j <= gorizontal + 1; j++)
		{
			array.push_back(image[i][j]);
		}
	}
	sort(array.begin(), array.end());
	return array[4];
}

vector<vector<int>> transformationImage(vector<vector<int>> image, float limitX, float limitY) {
	vector<vector<int>> output(image.size(), vector <int>(image[0].size()));
	int procent = -1;
	int last_procent = -1;

	//float count = 0;
	//float h = image.size()/100;
	for (int i = 1; i < (image.size() - 1)*limitX; i++)
	{
		procent = (int)(((double)i / (double)(image.size() - 2)/limitX) * 100);
		if (last_procent != procent) {
			system("clear");
			last_procent = procent;
			std::cout << "Posl: " << procent << "%" << endl;
		}
		//count +=h;
		//cout << setprecision(2) << count << endl;
		for (int j = 1; j < (image[0].size() - 1)*limitY; j++)
		{
			output[i][j] = MedianFilter(image, i, j);
		}
	}
	return output;
}

__global__ void kernel(float* arrayOutput, cudaTextureObject_t texObj, int width, int height) {

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int array[9];
	int k = 0;
	if ((index_x < width) && (index_y < height))
	{
		for (int i = index_x-1; i <= index_x + 1; i++)
		{
			for (int j = index_y-1; j <= index_y + 1; j++)
			{
				array[k] = (int)tex2D<float>(texObj, i, j);
				k++;
			}
		}
		for (int q = 0; q < 9; q++) 
		{
        		for (int w = 0; w < 8; w++) 
			{
            			if (array[w] > array[w + 1]) 
				{
               				float b = array[w]; // создали дополнительную переменную
                			array[w] = array[w+ 1]; // меняем местами
               				array[w + 1] = b; // значения элементов
            			}
        		}
    		}
		arrayOutput[(index_x)  + (index_y)* (width)] = array[4];

	}
}

int main(int argc, char* argv[])
{
	float limit_X = atof(argv[1]);
  float limit_Y = atof(argv[2]);
	// declare and read the bitmap
	BMP Input;
	Input.ReadFromFile("input.bmp");
	int width = Input.TellWidth();
	int height = Input.TellHeight();

	vector<vector<int>> a(width + 2, vector <int>(height + 2));

	// convert each pixel to grayscale using RGB->YUV
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			int Temp = (int)floor(0.3 * Input(i, j)->Red + 0.59 * Input(i, j)->Green + 0.11 * Input(i, j)->Blue);
			a[i + 1][j + 1] = Temp;
		}
	}

	for (size_t j = 1; j < height - 1; j++)
	{
		a[0][j] = a[1][j];
		a[width - 1][j] = a[width - 2][j];
	}
	for (size_t i = 1; i < width - 1; i++)
	{
		a[i][0] = a[i][1];
		a[i][height - 1] = a[i][height - 2];
	}
	a[0][0] = a[1][1];
	a[0][height - 1] = a[1][height - 2];
	a[width - 1][0] = a[width - 2][1];
	a[width - 1][height - 1] = a[width - 2][height - 2];

	float* h_data = (float*)malloc(width * height * sizeof(float));
	for (int i = 1; i < width+1; ++i)
		for (int j = 1; j < height+1; ++j)
			h_data[i * height + j] = a[i+1][j+1];

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray_t arrayInput;
	float* arrayOutput;

	cudaError_t cuerr = cudaMalloc(&arrayOutput, width * height * sizeof(float));
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot allocate device Ouput array for a: %s\n", cudaGetErrorString(cuerr));
		return 0;
	}

	cuerr = cudaMallocArray(&arrayInput, &channelDesc, width, height);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot allocate device Input array for a: %s\n", cudaGetErrorString(cuerr));
		return 0;
	}

	cuerr = cudaMemcpy2DToArray(arrayInput, 0, 0, h_data, (width) * sizeof(float), (width) * sizeof(float), (height), cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot copy a array2D from host to device: %s\n", cudaGetErrorString(cuerr));
		return 0;
	}

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = arrayInput;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp; // За границей текстуры будет продолжаться граничное значение
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0; // не использовать нормализованную адресацию

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cuerr = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot create TextureObject: %s\n", cudaGetErrorString(cuerr));
		return 0;
	}
	// Создание обработчиков событий
  cudaEvent_t start, stop;
  float gpuTime = 0.0f;
  cuerr = cudaEventCreate(&start);
  if (cuerr != cudaSuccess) {
  	fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
  	return 0;
   }
   cuerr = cudaEventCreate(&stop);
   if (cuerr != cudaSuccess) {
     fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
     return 0;
  }

	dim3 BLOCK_SIZE(32, 32, 1);
	dim3 GRID_SIZE(height  / 32 + 1, width/ 32 + 1, 1);

	// Установка точки старта
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

	kernel << <GRID_SIZE, BLOCK_SIZE >> > (arrayOutput, texObj, width, height);
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	// Синхронизация устройств
	cuerr = cudaDeviceSynchronize();
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
			cudaGetErrorString(cuerr));
		return 0;
	}

	// Установка точки окончания
    cuerr = cudaEventRecord(stop, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

	cuerr = cudaMemcpy(h_data, arrayOutput, width * sizeof(float) * height, cudaMemcpyDeviceToHost);
	if (cuerr != cudaSuccess) {
		fprintf(stderr, "Cannot copy a array from device to host: %s\n", cudaGetErrorString(cuerr));
		return 0;
	}	

	struct timespec mt1, mt2; 
  //Переменная для расчета дельты времени
  long int tt;
	clock_gettime(CLOCK_REALTIME, &mt1);

	a = transformationImage(a, limit_X, limit_Y);
	
	clock_gettime(CLOCK_REALTIME, &mt2);
  tt=1000000000*(mt2.tv_sec - mt1.tv_sec)+(mt2.tv_nsec - mt1.tv_nsec);
  printf("time spent executing cpu: %ld nanoseconds\n",tt);
	
	// Расчет времени
  cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
  printf("time spent executing kernel: %10.5f seconds\n", gpuTime /1000);
	printf("SpeedUp: %.9f \n", tt/(gpuTime*1000000));
	printf("LimitX: %.2f \n", limit_X);
	printf("LimitY: %.2f \n", limit_Y);
	printf("Width: %d \n", width);
	printf("Height: %d \n", height);

	for (int j = 0; j < height*limit_Y; j++)
	{
		for (int i = 0; i < width*limit_X; i++)
		{	
			ebmpBYTE color = (ebmpBYTE)h_data[i*height + j];
			Input(i, j)->Red = color;
			Input(i, j)->Green = color;
			Input(i, j)->Blue = color;
		}
	}
	BMP Output;
	Output.ReadFromFile("input.bmp");
		
	for (int j = 0; j < height*limit_Y; j++)
	{
		for (int i = 0; i < width*limit_X; i++)
		{
			ebmpBYTE color = (ebmpBYTE)a[i + 1][j + 1];
			Output(i, j)->Red = color;
			Output(i, j)->Green = color;
			Output(i, j)->Blue = color;
		}
	}
	
/*	
	//Create a grayscale color table if necessary
	if (Input.TellBitDepth() < 16)
	{
		CreateGrayscaleColorTable(Input);
	}
	
*/
	//write the output file
	Input.WriteToFile("outputGPU.bmp");
	Output.WriteToFile("outputCPU.bmp");
	
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(arrayInput);
	cudaFree(arrayOutput);
	
	free(h_data);
	return 0;
}