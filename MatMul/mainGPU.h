#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <cuda.h>
#include <time.h> //clock_gettime

//Умножение матриц последовательное. C = A*B, где размеры A ­- m*n, B - n*k, C - m*k
void MatMul (float* c, float* a, float* b, unsigned int m, unsigned int n, unsigned int k){
	for(int i = 0; i < m; i++){
		for(int j = 0; j < k; j++){
			for(int l = 0; l < n; l++){
				c[i*k+j] += a[i*n+l]*b[l*k+j];
			}
		}
	}
}

int main(int argc, char* argv[])
{
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
	bool autoFill = atoi(argv[4]); //флаг автоматического заполнения

    //Определяем размер грида и блоков
    dim3 BLOCK_SIZE(32 , 32, 1);
    dim3 GRID_SIZE(m / 32 + 1 , k / 32 + 1, 1);
   
    printf("m = %d\n", m);
    printf("n = %d\n", n);
    printf("k = %d\n", k);
	printf("\n");
    
    // Выделение памяти на хосте-CPU
    float* a = (float*)calloc(m*n*sizeof(int), sizeof(float));
    float* b = (float*)calloc(n*k*sizeof(int), sizeof(float));
    float* c = (float*)calloc(m*k*sizeof(int), sizeof(float));

	float* cCpu = (float*)calloc(m*k*sizeof(int), sizeof(float));

    // Инициализация массивов
	if (autoFill) {
		srand(time(NULL));
    	for (int i = 0; i < m; i++) {
        	for (int j=0; j<n; j++)	{
	   			a[i*n+j] = ((rand() % 10))/5.;
			}
    	}

    	for (int i = 0; i < n; i++) {
       		for (int j=0; j<k; j++) {
	  	  		b[i*k+j] = ((rand() % 10))/5.;
			}
    	}

    	for (int i = 0; i < m; i++) {
        	for (int j=0; j<k; j++) {
	   			c[i*k+j] = 0;
				cCpu[i*k+j] = 0;
			}
		}
		if(m < 10 && n < 10 && k <10) { //Если массивы слишком большие то не выводит
    		for (int i = 0; i < m; i++) {
        		for (int j=0; j<n; j++) {
             		printf("%5.3f", a[i*n+j]); 
             		printf("  ");
        		}   
			printf("\n");
    		}
    		printf("\n");
    		for (int i = 0; i < n; i++) {
        		for (int j=0; j<k; j++) {
             		printf("%5.3f", b[i*k+j]); 
             		printf("  ");
        		}
				printf("\n");
    		}
		}
    	printf("\n");
	}
    else {
		for (int i = 0; i < m; i++) {
        	for (int j=0; j<n; j++) {
	   			scanf("%f", &a[i*n+j]); 
			}
    	}
    	for (int i = 0; i < n; i++) {
       		for (int j=0; j<k; j++)	{
	  	  		scanf("%f", &b[i*k+j]);
			}
    	}
    	for (int i = 0; i < m; i++) {
        	for (int j=0; j<k; j++) {
	   			c[i*k+j]=0;
				cCpu[i*k+j] = 0; 
			}
		}    
    	printf("\n");
    	for (int i = 0; i < m; i++) {
        	for (int j=0; j<n; j++) {
             	printf("%5.3f", a[i*n+j]); 
             	printf("  ");
        	}   
		printf("\n");
    	}
    	printf("\n");
    	for (int i = 0; i < n; i++) {
        	for (int j=0; j<k; j++) {
             	printf("%5.3f", b[i*k+j]); 
             	printf("  ");
        	}
			printf("\n");
    	}
    	printf("\n");
	}
	
    // Выделение памяти на устройстве
    float* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, m*n*sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, n*k*sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    float* cdev = NULL;
    cuerr = cudaMalloc((void**)&cdev, m*k*sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
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

    // Копирование данных с хоста на девайс
    cuerr = cudaMemcpy(adev, a, m*n*sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaMemcpy(bdev, b, n*k*sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    cuerr = cudaMemcpy(cdev, c, m*k*sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy c array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Установка точки старта
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    //Запуск ядра
    //GRID_SIZE, BLOCK_SIZE
    MatMulKer <<< GRID_SIZE, BLOCK_SIZE >>> (cdev, adev, bdev, m, n, k);
	 
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
    // Синхронизация устройств
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
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

    // Копирование результата на хост
    cuerr = cudaMemcpy(c, cdev, m*k*sizeof(float), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }
	struct timespec mt1, mt2; 
    //Переменная для расчета дельты времени
    long int tt;
	clock_gettime(CLOCK_REALTIME, &mt1);

    MatMul(cCpu, a, b, m, n, k);

    clock_gettime(CLOCK_REALTIME, &mt2);
    tt=1000000000*(mt2.tv_sec - mt1.tv_sec)+(mt2.tv_nsec - mt1.tv_nsec);
    printf("time spent executing cpu: %ld nanoseconds\n",tt);
	
	if(m < 10 && n < 10 && k <10) {
    	for (int i = 0; i < m; i++) {
        	for (int j=0; j<k; j++) {
            	 printf("%5.3f", c[i*k+j]); 
            	 printf("  ");
        	}
		printf("\n");    
    	}
		printf("\n");
		for (int i = 0; i < m; i++) {
        	for (int j=0; j<k; j++) {
             	printf("%5.3f", cCpu[i*k+j]); 
             	printf("  ");
        	}
		printf("\n");    
    	}
	}

	printf("\n");
	float maxEl = 0;
	for(int i = 0; i< m*k; i++) {
		cCpu[i] -= c[i];
		if(abs(cCpu[i]) > maxEl) maxEl = abs(cCpu[i]);
	}

    // Расчет времени
    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing kernel: %10.5f seconds\n", gpuTime /1000);
	printf("maximum error: %.9f \n", maxEl);
	printf("SpeedUp: %.9f \n", tt/(gpuTime*1000000));


    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    free(a);
    free(b);
    free(c);
	free(cCpu);
    return 0;
}
