#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>

// Function to generate random numbers between -1 and 1
double random_number() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<double> dis(-1.0, 1.0);
  return dis(gen);
}

// Function to estimate pi using the Monte Carlo method
void calculate_pi(int &count, int &n_points, int num_iterations) {
  int hits = 0;
  for (int i = 0; i < num_iterations; ++i) {
    n_points++; // count every try

    double x = random_number();
    double y = random_number();

    if (x * x + y * y <= 1.0) {
      ++hits;
    }
  }

  count += hits;

  std::cout << "hits: " << hits << " of " << num_iterations << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: ./program_name num_iterations num_threads"
              << std::endl;
    return 1;
  }

  const int num_iterations = std::atoi(argv[1]);
  const int num_threads = std::atoi(argv[2]);

  assert(num_iterations > 0);
  assert(num_threads > 0);

  int count = 0;
  int n_points = 0;

  for (int i = 0; i < num_threads; ++i) {
    calculate_pi(count, n_points, num_iterations / num_threads);
  }

  std::cout << "count: " << count << " of " << n_points << std::endl;
  double pi = 4.0 * (double)count / (double)num_iterations;
  std::cout << "Used " << n_points << " points to estimate pi: " << pi
            << std::endl;

  assert(n_points = num_iterations);
  return 0;
}
