import subprocess
import sys

thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

for agents in [2, 3, 4]:
    for t in thresholds:
        label = f"{int(t*100):02d}"
        cmd = [
            "main.py",
            "--mode", "single",
            "--comm", "C3",
            "--loss", "0.0",
            "--latency", "0",
            "--seeds", "1000",
            "--grid", "50",
            "--episodes", "150",
            "--min-agents", str(agents),
            "--entropy-delta", str(t),
            "--output", f"06APR_agents{agents}_delta{label}.pkl",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run([sys.executable] + cmd, check=True)

print("All scripts completed.")