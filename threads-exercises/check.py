import os
import subprocess
import time
import re

min_speedup = 1.5

def compile_project(exercise_path):
    build_path = os.path.join(exercise_path, "build")
    subprocess.run(["mkdir", "-p", build_path], check=True)
    subprocess.run("cmake ..", shell=True, cwd=build_path, check=True)
    subprocess.run("make -j", shell=True, cwd=build_path, check=True)
    print(f"Compilation successful for {exercise_path}.")
    return build_path


def run_command(command, cwd):
    start_time = time.time()
    result = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if result.returncode != 0:
        print(f"Error running command: {command}\n{result.stderr.decode()}")

    return result.stdout.decode(), elapsed_time, result.returncode


def benchmark_exercise(name, compile_path, commands, check_func, reports_speedup=False):
    print(f"\n{name}:")
    build_path = compile_project(compile_path)
    outputs, times, return_codes = zip(
        *[run_command(cmd, build_path) for cmd in commands])

    if any(return_codes):
        print(f"❌{name}: One or more commands returned non-zero exit code.")
        return

    success = check_func(outputs)
    emotion = "✅" if success else "❌"
    print(f"{emotion}{name}: {'Success' if success else 'Failure'}")
    if not success:
        return

    if not reports_speedup and len(times) == 2:
        # Assuming the first command is parallel and the second is serial
        speedup = times[1] / times[0]
        emotion = "✅" if float(speedup) > min_speedup else "❌"
        print(f"{emotion}Calculated Speedup: {speedup:.2f}x")
    elif reports_speedup:
        speedup = extract_speedup(outputs[0])
        emotion = "✅" if float(speedup) > min_speedup else "❌"
        print(f"{emotion}Reported Speedup: {extract_speedup(outputs[0])}x")


def check_unordered_output(outputs):
    assert len(outputs) == 2
    output1 = outputs[0]
    output2 = outputs[1]
    return set(output1.splitlines()) == set(output2.splitlines())


def check_output_return_zero(output):
    # Placeholder: The function assumes output parsing is handled separately
    return True


def check_hits_against_count(output):
    assert len(output) == 2
    output1 = output[0]
    output2 = output[1]

    hits1 = [int(x) for x in re.findall(r"hits: (\d+)", output1)]
    count1 = int(re.findall(r"count: (\d+)", output1)[0])
    pi1 = float(re.findall(r"pi: (\d+\.\d+)", output1)[0])

    hits2 = [int(x) for x in re.findall(r"hits: (\d+)", output2)]
    count2 = int(re.findall(r"count: (\d+)", output2)[0])
    pi2 = float(re.findall(r"pi: (\d+\.\d+)", output2)[0])

    correctness = True

    if sum(hits1) != count1:
        correctness = False

    if sum(hits2) != count2:
        correctness = False

    if (abs(pi1/pi2 - 1) > 0.4):
        correctness = False

    return correctness


def check_sum_line_match(output):
    assert len(output) == 2
    output1 = output[0]
    output2 = output[1]
    sum_line1 = [line for line in output1.splitlines()
                 if line.startswith("sum")][0]
    sum_line2 = [line for line in output2.splitlines()
                 if line.startswith("sum")][0]
    return sum_line1 == sum_line2


def extract_speedup(output):
    speedup_line = [line for line in output.splitlines()
                    if "speedup" in line][0]
    speedup = float(speedup_line.split()[0].strip("(x"))
    return speedup


def main():
    benchmark_exercise(
        "Exercise 01",
        "01-hello-threads",
        ["./hello_threads_solution 4", "./hello_threads 4"],
        check_unordered_output,
        reports_speedup=False
    )
    print("---------------------------------------------------")

    benchmark_exercise(
        "Exercise 2",
        "02-fractal-generation",
        ["./mandelbrot -t 4"],
        check_output_return_zero,
        reports_speedup=True
    )
    print("---------------------------------------------------")

    benchmark_exercise(
        "Exercise 3",
        "03-monte-carlo-pi",
        ["./monte_carlo_parallel 40000000 4",
            "./monte_carlo_serial 40000000 4"],
        check_hits_against_count,
        reports_speedup=False
    )
    print("---------------------------------------------------")

    benchmark_exercise(
        "Exercise 4",
        "04-false-sharing",
        ["./sum_scalar_solution 75746584 4", "./sum_scalar 75746584 4"],
        check_sum_line_match,
        reports_speedup=False
    )


if __name__ == "__main__":
    main()
