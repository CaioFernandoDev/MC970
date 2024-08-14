#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <mutex>

std::mutex mtx;

typedef struct
{
  int num_iteration_per_thread;
  int *count;
  int *n_points;
} WorkerArgs;

// Function to generate random numbers between -1 and 1
double random_number()
{
  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  thread_local std::uniform_real_distribution<double> dis(-1.0, 1.0);
  return dis(gen);
}

// Function to estimate pi using the Monte Carlo method
void calculate_pi(WorkerArgs const *args)
{
  int num_iterations = args->num_iteration_per_thread;

  int hits = 0;
  int n_points_local = 0;

  for (int i = 0; i < num_iterations; i++)
  {
    n_points_local++; // count every try

    double x = random_number();
    double y = random_number();

    if (x * x + y * y <= 1.0)
    {
      hits++;
    }
  }

  int *count = args->count;
  int *n_points = args->n_points;

  // mutex for safety
  mtx.lock();
  *count += hits;
  *n_points += n_points_local;
  mtx.unlock();

  std::cout << "hits: " << hits << " of " << num_iterations << std::endl;
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
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

  std::thread workers[num_threads];
  WorkerArgs args[num_threads];

  for (int i = 0; i < num_threads; i++)
  {
    args[i].num_iteration_per_thread = num_iterations / num_threads;
    args[i].count = &count;
    args[i].n_points = &n_points;
  }

  for (int i = 0; i < num_threads; i++)
  {
    workers[i] = std::thread(calculate_pi, &args[i]);
  }

  for (int i = 0; i < num_threads; i++)
  {
    workers[i].join();
  }

  std::cout << "count: " << count << " of " << n_points << std::endl;
  double pi = 4.0 * (double)count / (double)num_iterations;
  std::cout << "Used " << n_points << " points to estimate pi: " << pi
            << std::endl;

  assert(n_points = num_iterations);
  return 0;
}
