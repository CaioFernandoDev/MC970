#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Print hello world from a thread
void *printHello(void *threadId) {
  long tid = (long)threadId;

  printf("Hello Worlds from thread #%ld!\n", tid);

  // Pause for 3 seconds
  sleep(3);

  printf("Goodbye from thread #%ld!\n", tid);

  return NULL;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <num_threads>\n", argv[0]);
    return 1;
  }

  int num_threads = atoi(argv[1]);
  long t;
  pthread_t th[num_threads];

  for (t = 0; t < num_threads; t++) {
    printf("Creating thread #%ld\n", t);
    pthread_create(&th[t], NULL, printHello, (void *)t);
  }

  // Wait for all threads to finish
  for (t = 0; t < num_threads; t++) {
    pthread_join(th[t], NULL);
  }

  return 0;
}