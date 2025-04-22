import yaml
from pathlib import Path
import subprocess

PROJECT = "COMP7055-Project"
LOG_DIR = "/Users/danielthompson/Documents/HKBU/COMP7055-Project/cache/0403/lightning_logs"
ENTITY = 'ruihe'


def generate_config_file(directory, config):
    config_path = directory / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"Generated config file at {config_path}")


def sync_to_wandb():
    for gan_dir in Path(LOG_DIR).iterdir():
        if not gan_dir.is_dir():
            continue

        gan = gan_dir.name
        print(f"Processing GAN: {gan}")

        for rate_dir in gan_dir.iterdir():
            if not rate_dir.is_dir():
                continue

            rate = rate_dir.name
            print(f"  Rate: {rate}")

            for model_dir in rate_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model = model_dir.name
                run_name = f"{model}_{gan}_{rate}"

                if not any(model_dir.glob("events.out.tfevents*")):
                    print(f"⚠️ Skipping {run_name}: No TensorBoard logs found")
                    continue

                generate_config_file(model_dir, {
                    "model": model,
                    "gan": gan,
                    "rate": float(rate),
                })

                cmd = (
                    f"wandb sync --sync-tensorboard --include-offline --mark-synced "
                    f"-p {PROJECT} "
                    f"--id '{run_name}' "
                )
                if ENTITY:
                    cmd += f"-e {ENTITY} "
                cmd += f'"{model_dir}"'

                subprocess.run(cmd, shell=True, check=True)
                print(f"✅ Synced {run_name}")


if __name__ == "__main__":
    sync_to_wandb()
