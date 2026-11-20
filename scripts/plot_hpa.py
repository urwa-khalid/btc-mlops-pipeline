"""Plot HPA autoscaling from a log produced by:

    while true; do echo "$(date +%T) $(kubectl get hpa btc-forecast-hpa --no-headers)"; \
        sleep 5; done | tee load/hpa_log.txt

Usage: python scripts/plot_hpa.py load/hpa_log.txt report/hpa_scaling.png
"""

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LINE = re.compile(
    r"^(\d\d:\d\d:\d\d)\s+\S+\s+\S+\s+(?:cpu:\s*)?(\S+?)/(\d+)%\s+(\d+)\s+(\d+)\s+(\d+)\s"
)


def parse(path):
    times, cpu, replicas, target = [], [], [], 60
    with open(path) as handle:
        for line in handle:
            match = LINE.match(line.strip() + " ")
            if not match:
                continue
            usage = match.group(2).rstrip("%")
            times.append(match.group(1))
            cpu.append(float(usage) if usage.replace(".", "").isdigit() else 0.0)
            target = int(match.group(3))
            replicas.append(int(match.group(6)))
    return times, cpu, replicas, target


def main(log_path, out_path):
    times, cpu, replicas, target = parse(log_path)
    if not times:
        sys.exit("no HPA lines found in the log")
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(range(len(times)), cpu, color="#B45309", label="CPU utilisation (%)")
    ax1.axhline(target, color="#B45309", ls="--", lw=1, label=f"target {target}%")
    ax1.set_ylabel("CPU utilisation (%)")
    ax2 = ax1.twinx()
    ax2.step(
        range(len(times)), replicas, where="post", color="#4338CA", label="replicas"
    )
    ax2.set_ylabel("Pod replicas")
    ax2.set_ylim(0, max(replicas) + 1)
    step = max(1, len(times) // 8)
    ax1.set_xticks(range(0, len(times), step))
    ax1.set_xticklabels(times[::step], rotation=30, ha="right")
    ax1.set_xlabel("time")
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper left")
    ax1.set_title("Horizontal Pod Autoscaler under load")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
