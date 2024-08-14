#include <random>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <curand.h>

using namespace std;

__global__ void PathGenKernel(
    float* Paths,
    float* Normals,
    float InitialStockPrice,
    float InterestRate,
    float Dt,
    float SqrtDt,
    float Volatility,
    int NumberOfSteps,
    int NumberOfPaths)
{
    int PathIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (PathIdx < NumberOfPaths) {
        float StockPrice = InitialStockPrice;
        int NormalIdx = PathIdx * NumberOfSteps;
        Paths[NormalIdx] = InitialStockPrice;
        for (int StepIdx = 1; StepIdx <= NumberOfSteps; StepIdx++)
        {
            StockPrice += StockPrice * InterestRate * Dt + StockPrice * Volatility * SqrtDt * Normals[NormalIdx];
            Paths[NormalIdx] = StockPrice;
            NormalIdx++;
        }
    }
}

__global__ void BarrierCallKernel(
    float* Paths,
    float* Payoff,
    float Strike,
    float Barrier,
    int NumberOfSteps,
    int NumberOfPaths)
{
    int PathIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (PathIdx < NumberOfPaths) 
    {
        int StepIdx = 1;
        bool Expired = false;
        float StockPrice = Paths[PathIdx * NumberOfSteps];
        while (!Expired && StepIdx < NumberOfSteps)
        {
          StockPrice = Paths[PathIdx * NumberOfSteps + StepIdx];
          if (StockPrice < Barrier)
              Expired = true;
          StepIdx++;
         }

        if (!Expired && StockPrice > Strike)
            Payoff[PathIdx] = StockPrice - Strike;
        else
            Payoff[PathIdx] = 0.0;
    }
}

int main()
{
    // Setting parameter values
    const int seed = 123456789;
    const int NumberOfPaths = 1'000'000;
    const int NumberOfSteps = 500;
    const float Maturity = 1.0;
    const float Strike = 85.0;
    const float Barrier = 90.0;
    const float InitialStockPrice = 100.0;
    const float Volatility = 0.2;
    const float InterestRate = 0.05;
    float Dt = float(Maturity) / float(NumberOfSteps);
    float SqrtDt = sqrt(Dt);
    const size_t size = NumberOfPaths * NumberOfSteps * sizeof(float);

    int ThreadsPerBlock = 512;
    int Blocks = ceil(NumberOfPaths / ThreadsPerBlock);

    // Generating random numbers
    double t1 = double(clock()) / CLOCKS_PER_SEC;
    float* devNormals;
    cudaMalloc(&devNormals, size);

    curandGenerator_t curandGenerator;
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, seed);

    curandGenerateNormal(curandGenerator, devNormals, NumberOfPaths * NumberOfSteps, 0.0, 1.0);
    double t2 = double(clock()) / CLOCKS_PER_SEC;

    // Path generation
    double t3 = double(clock()) / CLOCKS_PER_SEC;
    float* devPaths;
    cudaMalloc(&devPaths, size);

    PathGenKernel<<<Blocks, ThreadsPerBlock>>>(devPaths, devNormals, InitialStockPrice, InterestRate, Dt, SqrtDt, Volatility, NumberOfSteps, NumberOfPaths);
    cudaDeviceSynchronize();
    double t4 = double(clock()) / CLOCKS_PER_SEC;

    // Payoff evaluation
    double t5 = double(clock()) / CLOCKS_PER_SEC;
    float* devPayoff;
    cudaMalloc(&devPayoff, NumberOfPaths * sizeof(float));
    
    BarrierCallKernel<<<Blocks, ThreadsPerBlock>>>(devPaths, devPayoff, Strike, Barrier, NumberOfSteps, NumberOfPaths);
    cudaDeviceSynchronize();
    
    vector<float> Payoff(NumberOfPaths);
    cudaMemcpy(Payoff.data(), devPayoff, NumberOfPaths * sizeof(float), cudaMemcpyDeviceToHost);
    double t6 = double(clock()) / CLOCKS_PER_SEC;

    // Calculating average payoff
    float sum = 0;
    for (int PathIdx = 0; PathIdx < NumberOfPaths; PathIdx++)
    {
        sum += Payoff.at(PathIdx);
    }
    float average = sum / NumberOfPaths;

    cout << "Average Payoff: " << average << "\n";
    cout << "Random number generator: " << (t2 - t1) << "s\n";
    cout << "Path generation: " << (t4 - t3) << "s\n";
    cout << "Payoff evaluation: " << (t6 - t5) << "s\n";
    cout << "Total time: " << (t6 - t1) << "s\n";

    return 0;
}