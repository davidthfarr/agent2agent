import subprocess
import sys

base_cmd = [
    "main.py",
    "--seeds", "100",
    "--mode", "full",
    "--grid", "50",
    "--episodes", "150"
]

for agents in [2, 3, 4]:
    cmd = base_cmd + [
        "--min-agents", str(agents),
        "--output", f"10MAR_{agents}.pkl"
    ]

    print("Running:", " ".join(cmd))
    subprocess.run([sys.executable] + cmd, check=True)

print("All scripts completed.")