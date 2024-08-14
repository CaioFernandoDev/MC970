import os
import subprocess
import time
import glob
import statistics

# Change to the script's directory to ensure paths are handled correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the exercises to be compiled and benchmarked
exercises = [
    {
        "name": "Exercise 1",
        "dir": "01-OmpTask-Zip",
    },
    {
        "name": "Exercise 2",
        "dir": "02-OmpTask-TreeReduce",
    },
]

# Minimum speedup required to consider the parallelization effective
MIN_SPEEDUP = 1.5
NUMBER_OF_RUNS = 3


def run_subprocess(command, cwd=None):
    """Execute a subprocess command via the shell, using the command as a string."""
    env = {
        'OMP_CANCELLATION': 'true'
    }

    start_time = time.time()
    par_or_ser = "par" if "parallel" in command else "ser"
    test_file_path = command.split()[1]
    test_file_name = os.path.basename(test_file_path)
    test_num = test_file_name.split(".")[0]
    process = subprocess.run(
        command +
        f" 2> build/{test_num}.{par_or_ser}.err > build/{test_num}.{par_or_ser}.std",
        shell=True,
        cwd=cwd,
        env = env,
    )
    end_time = time.time()
    if process.returncode != 0:
        print(f"Running command: {command}")
        print(f"Working directory: {cwd}")
        print(
            f"Command '{command}' failed with return code {process.returncode}")
        print(f"Error output:\n{process.stderr.decode()}")

    return end_time - start_time, process.returncode


def compile_project(exercise_dir):
    """Compile the exercise project."""
    build_dir = os.path.join(exercise_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    compile_commands = ["cmake -DCMAKE_BUILD_TYPE=Release ..", "make -j"]
    for cmd in compile_commands:
        process = subprocess.run(
            cmd, shell=True, stderr=subprocess.PIPE, cwd=build_dir)
        if process.returncode != 0:
            print(
                f"Compilation failed for {exercise_dir} with error:\n{process.stderr.decode()}")
            return False
    return True


def execute_and_compare(exercise_dir, test_files):
    """Execute serial and parallel versions of the exercise, compare their outputs, and take the median of multiple runs."""
    # build_dir = os.path.join(exercise_dir, "build")
    build_dir = "build"
    speedups_met_criteria = 0
    successful_compilation_and_output = True

    test_files.sort()  # Ensure alphabetical order

    for test_file in test_files:
        base_name = os.path.splitext(os.path.basename(test_file))[0]
        parallel_command = f"{build_dir}/parallel {os.path.abspath(test_file)}"
        serial_command = f"{build_dir}/serial {os.path.abspath(test_file)}"

        parallel_times = []
        serial_times = []

        for i in range(NUMBER_OF_RUNS):
            print(f"Run {i+1} of {NUMBER_OF_RUNS} for {test_file}")
            _, parallel_return_code = run_subprocess(
                parallel_command, cwd=exercise_dir)
            _, serial_return_code = run_subprocess(
                serial_command, cwd=exercise_dir)

            if parallel_return_code != 0 or serial_return_code != 0:
                successful_compilation_and_output = False
                break

            # compare output from serial and parallel version
            ser_output_path = f"{exercise_dir}/{build_dir}/{base_name}.ser.std"
            par_output_path = f"{exercise_dir}/{build_dir}/{base_name}.par.std"
            compare_command = f"diff {ser_output_path} {par_output_path}"
            process = subprocess.run(compare_command, shell=True)
            if process.returncode != 0:
                print(f"Output mismatch for {base_name}")
                successful_compilation_and_output = False
                break

            # get times from {test_num}.{ser/par}.err
            with open(f"{exercise_dir}/{build_dir}/{base_name}.ser.err") as f:
                serial_time = float(f.read())

            with open(f"{exercise_dir}/{build_dir}/{base_name}.par.err") as f:
                parallel_time = float(f.read())

            parallel_times.append(parallel_time)
            serial_times.append(serial_time)

        if not successful_compilation_and_output:
            continue

        median_parallel_time = statistics.median(parallel_times)
        median_serial_time = statistics.median(serial_times)

        speedup = median_serial_time / median_parallel_time if median_parallel_time > 0 else 0
        if speedup >= MIN_SPEEDUP:
            speedups_met_criteria += 1

        print(f"Test {base_name}: Median Speedup = {speedup:.2f}x")

    # Display emoticons based on the outcome
    if speedups_met_criteria >= 2 and successful_compilation_and_output:
        print("✅ Compilation and execution successful with minimum speedup achieved on at least two outputs.")
    else:
        print("❌ Did not achieve the minimum speedup on at least two outputs or had incorrect outputs or execution failures.")


def main():
    for exercise in exercises:
        print(f"Running {exercise['name']}...")
        if compile_project(exercise["dir"]):
            test_files = glob.glob(os.path.join(
                exercise["dir"], "tests", "*.in"))
            execute_and_compare(exercise["dir"], test_files)
        else:
            print("Skipping due to compilation failure.")


if __name__ == "__main__":
    main()
