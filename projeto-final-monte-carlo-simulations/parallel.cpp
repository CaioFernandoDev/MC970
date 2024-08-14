#include <random>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <omp.h>

using namespace std;

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

    // Creating 2d-vector for paths
    vector<vector<float>> Paths(NumberOfPaths, vector<float>(NumberOfSteps, InitialStockPrice));

    // Generating random numbers
    mt19937 generator(seed);
    normal_distribution<float> StdNormal(0, 1);
    vector<float> Normals(NumberOfPaths * NumberOfSteps);
    vector<float> Payoff(NumberOfPaths);

    double t1 = double(clock()) / CLOCKS_PER_SEC;
    #pragma omp parallel for
    for (int NormalIdx = 0; NormalIdx < NumberOfPaths * NumberOfSteps; NormalIdx++)
    {
        Normals.at(NormalIdx) = StdNormal(generator);
    }
    double t2 = double(clock()) / CLOCKS_PER_SEC;

    // Path generation
    double t3 = double(clock()) / CLOCKS_PER_SEC;
    for (int PathIdx = 0; PathIdx < NumberOfPaths; PathIdx++)
    {
        float StockPrice = InitialStockPrice;
        int NormalIdx = PathIdx * NumberOfSteps;
        for (int StepIdx = 1; StepIdx < NumberOfSteps; StepIdx++)
        {
            StockPrice += StockPrice * InterestRate * Dt + StockPrice * Volatility * SqrtDt * Normals.at(NormalIdx);
            Paths.at(PathIdx).at(StepIdx) = StockPrice;
            NormalIdx++;
        }
    }
    double t4 = double(clock()) / CLOCKS_PER_SEC;

    // Payoff evaluation
    double t5 = double(clock()) / CLOCKS_PER_SEC;
    for (int PathIdx = 0; PathIdx < NumberOfPaths; PathIdx++)
    {
        int StepIdx = 0;
        float StockPrice = InitialStockPrice;
        bool Expired = false;
        while (!Expired && StepIdx < NumberOfSteps)
        {
            StockPrice = Paths.at(PathIdx).at(StepIdx);
            if (StockPrice < Barrier)
                Expired = true;
            StepIdx++;
        }
        if (!Expired && StockPrice > Strike)
            Payoff.at(PathIdx) = StockPrice - Strike;
    }
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