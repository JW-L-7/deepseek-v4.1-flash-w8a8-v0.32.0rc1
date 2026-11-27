import argparse
import multiprocessing
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="./run_dp_template.sh")
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pcp-size", type=int, default=1)
    parser.add_argument("--dp-size-local", type=int, default=-1)
    parser.add_argument("--dp-rank-start", type=int, default=0)
    parser.add_argument("--dp-address", required=True)
    parser.add_argument("--dp-rpc-port", default="12325")
    parser.add_argument("--vllm-start-port", type=int, default=8000)
    args = parser.parse_args()
    if args.dp_size_local == -1:
        args.dp_size_local = args.dp_size
    return args


def run(args, devices, port, dp_rank):
    script_args = [
        devices,
        str(port),
        str(args.dp_size),
        str(dp_rank),
        args.dp_address,
        args.dp_rpc_port,
        str(args.tp_size),
        str(args.pcp_size),
    ]
    if args.dp_size == 1:
        with open(args.template, encoding="utf-8") as f:
            script = f.read()
        script = "\n".join(
            line for line in script.splitlines()
            if "--data-parallel-rank" not in line
            and "--data-parallel-external-lb" not in line
        )
        command = ["bash", "-o", "pipefail", "-c", script, args.template, *script_args]
    else:
        command = ["bash", "-o", "pipefail", args.template, *script_args]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    args = parse_args()
    processes = []
    devices_per_dp_rank = args.tp_size * args.pcp_size
    for i in range(args.dp_size_local):
        devices = ",".join(
            str(device)
            for device in range(i * devices_per_dp_rank, (i + 1) * devices_per_dp_rank)
        )
        process = multiprocessing.Process(
            target=run,
            args=(
                args,
                devices,
                args.vllm_start_port + i,
                args.dp_rank_start + i,
            ),
        )
        processes.append(process)
        process.start()
    for process in processes:
        process.join()