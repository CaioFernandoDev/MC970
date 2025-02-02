OpenMP Tasks: ZIP
================================================================================

ZIP is an archive file format that supports lossless data compression. In this
assignment you are in charge of brute-forcing a ZIP file which was compressed
with a numeric key. You must parallelize the code using OpenMP Tasks in a way
that you can obtain the key as soon as possible.

[wiki]: https://en.wikipedia.org/wiki/Zip_(file_format)

### Input

This program takes two lines as input:

- Number of threads to be used;
- Path to the ZIP file to be brute-forced.

**Note:** be sure to have `unzip` installed in your system, and that run your binary from the assignment folder (`01-OmpTask-Zip`), as instructed in this README.

### Output

This program outputs the secret key and the time it took to find that key.

Tasks & Rules
--------------------------------------------------------------------------------

You should do the following tasks:

- [ ] Understand the serial code in `src/zip-serial.c`
- [ ] Parallelize the code using OpenMP Tasks in the file `src/zip-parallel.c`
- [ ] Run both versions and compare them. Did you get any speedup?

You must **not** change the serial implementation, only the parallel one.

Grading
--------------------------------------------------------------------------------

Your assignment will be evaluated in terms of:

- Correctness: your program returns the correct result;
- Performance: your program runs faster than the serial implementation.

In order to test your solution, you can use one of the inputs available inside
the `tests/` directory. 
Your grade will be computed using an automated routine restricted to the
instructors and TAs. This routine will be run after the assignment deadline,
using the latest commit push to the branch `main` before the deadline. Your
code will be ensured to run in an environment with no competition for resources.

**Note:** The automatic grading routine expect your the output of
your program to be formatted correctly. For that reason, you should not add
`printf`s or any other function that writes to `stdout`, otherwise your
assignment will be considered incorrect.

**Note:** Tampering with the serial implementation or the tests is considered
cheating and will result in disqualification of the assignment.

Compiling & Running
--------------------------------------------------------------------------------

After you have accepted this assignment on the course's GitHub Classroom page,
clone it to your machine. First you have to generate the build system using
[CMake](https://cmake.org/). Make sure you have it installed! You will also need
an OpenMP compatible compiler. If you use Linux, you are good to go. For MacOS
users, you can install the OpenMP libraries via [Homebrew](https://brew.sh/)
with the following command:

```bash
# Only for MacOS users
brew install libomp
```

Then, run the following commands:

```bash
# Where the build will live
mkdir build && cd build

# Generate the Makefiles
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Having done that, still inside the `build` directory, run `make` to compile
everything. Finally, from the root directory of the repo you can execute the
serial and parallel versions like so:

```bash
build/serial tests/1.in
build/parallel tests/2.in
```

If you have any doubts or run into problems, please contact the TAs. Happy
coding! :smile: :keyboard:

Try to answer the following questions
--------------------------------------------------------------------------------

- Can parallelizing the code with OpenMP Tasks help you to find the key faster?
- Is it possible to achieve super-linear speedup in this assignment? Why?
- What are the advantages of using tasks in this kind of problem?

Contribute
--------------------------------------------------------------------------------

Found a typo? Something is missing or broken? Have ideas for improvement? The
instructor and the TAs would love to hear from you!

About
--------------------------------------------------------------------------------

This repository is one of the assignments handed out to the students in course
"MC970 - Introduction to Parallel Programming" offered by the Institute of
Computing at Unicamp.
