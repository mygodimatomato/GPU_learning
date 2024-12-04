import subprocess
import re
import statistics

# Function to run the nvprof command and parse the total profiling time
def run_profiling(command, num_runs=1):
    total_times = []
    profiling_regex = r"API calls:\s+.*?Time\(%\)\s+Time\s+Calls.*?\n(?:.*?\n)+?\s+(\d+\.\d+)ms.*?cudaMalloc"

    for i in range(num_runs):
        try:
            # Run the profiling command
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Get the output from stderr since nvprof writes profiling results there
            output = result.stderr
            print(f"Run {i+1} output: {output}")
            # Extract the total profiling time using regex (extracting cudaMalloc specifically)
            match = re.search(profiling_regex, output, re.MULTILINE)
            # print(f"Match: {match}")
            if match:
                profiling_time = float(match.group(1))
                total_times.append(profiling_time)
            else:
                print(f"Run {i+1}: Could not parse profiling time from output.")

        except Exception as e:
            print(f"Error during run {i+1}: {e}")

    return total_times

# Function to calculate average, max, and min times
def calculate_statistics(times):
    if not times:
        return None, None, None

    avg_time = statistics.mean(times)
    max_time = max(times)
    min_time = min(times)
    return avg_time, max_time, min_time

# # Run profiling for saxpy
# print("Running profiling for ./saxpy...")
# saxpy_command = ["nvprof", "./saxpy"]
# saxpy_times = run_profiling(saxpy_command, num_runs=5)
# saxpy_avg, saxpy_max, saxpy_min = calculate_statistics(saxpy_times)

# # Display statistics for saxpy
# print("\nStatistics for ./saxpy:")
# print(f"Average Time: {saxpy_avg:.2f} ms")
# print(f"Max Time: {saxpy_max:.2f} ms")
# print(f"Min Time: {saxpy_min:.2f} ms")

# Run profiling for saxpy_unified_init
print("\nRunning profiling for ./saxpy_unified_init...")
saxpy_unified_command = ["nvprof", "./saxpy_unified_init"]
saxpy_unified_times = run_profiling(saxpy_unified_command, num_runs=5)
saxpy_unified_avg, saxpy_unified_max, saxpy_unified_min = calculate_statistics(saxpy_unified_times)

# Display statistics for saxpy_unified
print("\nStatistics for ./saxpy_unified:")
print(f"Average Time: {saxpy_unified_avg:.2f} ms")
print(f"Max Time: {saxpy_unified_max:.2f} ms")
print(f"Min Time: {saxpy_unified_min:.2f} ms")
