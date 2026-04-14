import subprocess
import sys

base_cmd = [
    "main.py",
    "--seeds", "1000",
    "--mode", "full",
    "--grid", "50",
    "--episodes", "200"
]

for agents in [2, 3, 4]:
    cmd = base_cmd + [
        "--min-agents", str(agents),
        "--output", f"09APR_{agents}.pkl"
    ]

    print("Running:", " ".join(cmd))
    subprocess.run([sys.executable] + cmd, check=True)

print("All scripts completed.")