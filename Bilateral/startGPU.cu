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

bool IsCudaSuccess(cudaError_t cudaError, const char* message)
{
	if (cudaError != cudaSuccess) {
		cout << message << cudaGetErrorString(cudaError) << endl;
		return false;
	}
	return true;
}

void GetG(float* G_d, float sigmaD) {
	int index = 0;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			G_d[index++] = exp((pow(i, 2) + pow(j, 2)) / (-1*pow(sigmaD,2)));
		}
	}
}

int BilateralFilter(vector<vector<int>> image, int verctical, int gorizontal, float sigmaR, float* G_h) {
	float a0 = image[verctical][gorizontal];
	float hai = 0;
	float ai;
	int index = 0;
	float k = 0;
	for (int i = verctical - 1; i <= verctical + 1; i++)
	{
		for (int j = gorizontal - 1; j <= gorizontal + 1; j++)
		{
			ai = image[i][j];
			float rai = exp((pow(ai - a0, 2)) / (pow(sigmaR, 2)));
			hai += ai * G_h[index] * rai;
			k += G_h[index] * rai;
			++index;
		}
	}
	return (int)(hai / k);
}

vector<vector<int>> transformationImage(vector<vector<int>> image, float limitX, float limitY, float sigmaR, float* G_h) {
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
			output[i][j] = BilateralFilter(image, i, j, sigmaR, G_h);
		}
	}
	return output;
}

__global__ void kernel(float* G, float* arrayOutput, cudaTextureObject_t texObj, int width, int height, float sigmaR) {

	int index_x = blockIdx.x * blockDim.x + threadIdx.x; 
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	float a0 = tex2D<float>(texObj, index_x, index_y);
	float hai = 0;
	float ai;
	int index = 0;
	float k = 0;
	if ((index_x < width) && (index_y < height))
	{
		for (int i = index_x - 1; i <= index_x + 1; i++)
		{
			for (int j = index_y-1; j <= index_y + 1; j++)
			{
				ai = tex2D<float>(texObj, i, j);
				float rai = exp((pow(ai - a0, 2.0)) / (pow(sigmaR, 2.0)));
				hai += ai * G[index] * rai;
				k += G[index] * rai;
				++index;			
			}
		}
		arrayOutput[(index_x)  + (index_y)* (width)] = (hai / k);
	}	
}

int main(int argc, char* argv[])
{
	float limit_X = atof(argv[1]);
  	float limit_Y = atof(argv[2]);
	float sigmaD = atof(argv[3]);
  	float sigmaR = atof(argv[4]);
	if (limit_X < 0 || limit_X > 1  || limit_Y < 0 || limit_Y > 1 || sigmaD < 0 || sigmaR < 0)
	{
		cout << "Parameters in the following range were expected:" << endl;
		cout << "0 < limit_x,limit_y <= 1 and 0 < sigmaD,sigmaR" << endl;
		cout << "Received:" << endl;
		cout << "limit_x: " <<limit_X<<"  limit_y: " <<limit_Y<<"  sigmaD: " <<sigmaD<<"  sigmaR: "<<sigmaR<< endl;
		return 0;
	}

	BMP Input;
	Input.ReadFromFile("input.bmp");
	int width = Input.TellWidth();
	int height = Input.TellHeight();

	vector<vector<int>> a(width + 2, vector <int>(height + 2));
	float* h_data = (float*)malloc(width * height * sizeof(float));
	float* G_h = (float*)malloc(9 * sizeof(float));
	float* G_d;

	GetG(G_h, sigmaD);

	// convert each pixel to grayscale using RGB->YUV
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			int Temp = (int)floor(0.3 * Input(i, j)->Red + 0.59 * Input(i, j)->Green + 0.11 * Input(i, j)->Blue);
			a[i + 1][j + 1] = Temp;
			h_data[i*height + j] = Temp;
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

	cudaError_t cuerr = cudaMalloc(&G_d, 9 * sizeof(float));
	if (!IsCudaSuccess(cuerr, "Cannot allocate device G_d for a: ")) return 0;

	cuerr = cudaMemcpy(G_d, G_h, 9 * sizeof(float), cudaMemcpyHostToDevice);
	if (!IsCudaSuccess(cuerr, "Cannot copy a array from host to device: ")) return 0;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray_t arrayInput;
	float* arrayOutput;

	cuerr = cudaMalloc((void**)&arrayOutput, width * height * sizeof(float));
	if (!IsCudaSuccess(cuerr, "Cannot allocate device arrayOutput for a: ")) return 0;

	cuerr = cudaMallocArray(&arrayInput, &channelDesc, width, height);
	if (!IsCudaSuccess(cuerr, "Cannot allocate device arrayInput for a:")) return 0;

	cuerr = cudaMemcpy2DToArray(arrayInput, 0, 0, h_data, (width) * sizeof(float), (width) * sizeof(float), (height), cudaMemcpyHostToDevice);
	if (!IsCudaSuccess(cuerr, "Cannot copy a array2D from host to device: ")) return 0;

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
	if (!IsCudaSuccess(cuerr, "Cannot create TextureObject: ")) return 0;
	
	// Создание обработчиков событий
  	cudaEvent_t start, stop;
  	float gpuTime = 0.0f;
  	cuerr = cudaEventCreate(&start);
  	if (!IsCudaSuccess(cuerr, "Cannot create CUDA start event: ")) return 0;

   	cuerr = cudaEventCreate(&stop);
   	if (!IsCudaSuccess(cuerr, "Cannot create CUDA end event: ")) return 0;

	dim3 BLOCK_SIZE(32, 32, 1);
	dim3 GRID_SIZE(width / 32 + 1, height/ 32 + 1, 1);

	// Установка точки старта
    cuerr = cudaEventRecord(start, 0);
	if (!IsCudaSuccess(cuerr, "Cannot record CUDA event: %s\n ")) return 0;
    
	kernel << <GRID_SIZE, BLOCK_SIZE >> > (G_d, arrayOutput, texObj, width, height, sigmaR);
	cuerr = cudaGetLastError();
	if (!IsCudaSuccess(cuerr, "Cannot launch CUDA kernel: %s\n ")) return 0;
	
	// Синхронизация устройств
	cuerr = cudaDeviceSynchronize();
	if (!IsCudaSuccess(cuerr, "Cannot synchronize CUDA kernel: ")) return 0;

	// Установка точки окончания
    cuerr = cudaEventRecord(stop, 0);
    if (!IsCudaSuccess(cuerr, "Cannot copy c array from device to host: ")) return 0;

	cuerr = cudaMemcpy(h_data, arrayOutput, width * sizeof(float) * height, cudaMemcpyDeviceToHost);
	if (!IsCudaSuccess(cuerr, "Cannot copy a array from device to host: ")) return 0;

	struct timespec mt1, mt2; 
  	//Переменная для расчета дельты времени
  	long int tt;
	clock_gettime(CLOCK_REALTIME, &mt1);

	a = transformationImage(a, limit_X, limit_Y, sigmaR, G_h);
	
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
	printf("sigmaD: %.2f \n", sigmaD);
	printf("sigmaR: %.2f \n", sigmaR);

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

	//write the output file
	Input.WriteToFile("outputGPU.bmp");
	Output.WriteToFile("outputCPU.bmp");
	
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(arrayInput);
	cudaFree(arrayOutput);
	cudaFree(G_d);
	
	free(h_data);
	free(G_h);
	return 0;
}