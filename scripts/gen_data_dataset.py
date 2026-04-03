# Generate dataset by name: benchmark_operator, vlasov_maxwell_gkeyll, fusion_real_lhd.

import argparse
try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: run  pip3 install -r requirements.txt   or  python3 -m pip install PyYAML")
from prism_ad.datasets import DATASET_REGISTRY


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="e.g. configs/data_benchmark.yaml")
    p.add_argument("--out_base", default="data")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    name = cfg["dataset_name"]
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name}")
    out_dir = DATASET_REGISTRY[name](args.config, out_base=args.out_base)
    print(f"Generated {name} -> {out_dir}")


if __name__ == "__main__":
    main()
