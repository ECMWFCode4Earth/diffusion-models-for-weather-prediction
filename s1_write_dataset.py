import argparse
import os

from WD.datasets import write_conditional_datasets

parser = argparse.ArgumentParser(
    prog="WeatherDiffCondVanilla",
    description="Vanilla Conditional Diffusion Model",
    epilog="Arg parser for vanilla conditional diffusion model",
)

parser.add_argument(
    "-cf",
    "--config_file_path",
    type=str,
    help="path under which the selected config file is stored.",
)

args = parser.parse_args()

print(args.config_file_path)
assert os.path.isfile(args.config_file_path)

write_conditional_datasets(args.config_file_path)
