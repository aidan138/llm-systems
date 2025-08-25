import os
import submitit
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


LOG_DIR = "./slurm_logs"
NSYS_DIR = "./nsys_logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(NSYS_DIR, exist_ok=True)

# Define the sweep 
sizes = ['small'
#, 'medium', 'large', 'xl'
]
context_lengths = [128
#,256, 512, 1024
]
mode = 'forward-backward'

jobs = []

for size in sizes:
    for cl in context_lengths:
        jobs.append((size, cl))

def run_benchmark(job):
    size, cl = job
    output = f"{NSYS_DIR}/{size}_{cl}"
    cmd = (
        # 
        f'uv run '
        f"nsys profile "

        f"--pytorch autograd-nvtx "
        f"--python-backtrace=cuda "
        # f"--sample=none --cpuctxsw=none "


        # Configs found online
        f"-w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cudabacktrace=true -x true "
        f"-o {output} "
        f"python ./cs336_systems/launch_benchmarks.py "
        f"--size {size} "
        f"--mode {mode} "
        f"--seq-len {cl} "
    )
    print(f"Running: {cmd}")
    os.system(cmd)


def main():
    # executor = submitit.AutoExecutor(folder=LOG_DIR)

    # executor.update_parameters(
    #     timeout_min=30,              # Max runtime per job
    #     slurm_partition="gpu",       # Change this to your cluster partition
    #     slurm_gres="gpu:1",          # Request 1 GPU
    #     slurm_mem="16G",             # Memory per job
    #     cpus_per_task=4,             # CPU threads
    # )

    # # Submit jobs
    # # jobs_list = executor.map_array(run_benchmark, jobs)
    jobs_list = []
    for job in jobs:
        run_benchmark(job)
        
    # print(f"Submitted {len(jobs)} jobs")

    # for j in jobs_list:
    #     try:
    #         print(f"Waiting for job {j.job_id} …")
    #         j.result()  # Blocks until job finishes
    #         print(f"Job {j.job_id} completed successfully.")
    #     except Exception as e:
    #         print(f"Job {j.job_id} failed with exception: {e}")
    #         print(j.stderr())

if __name__ == '__main__':
    main()

