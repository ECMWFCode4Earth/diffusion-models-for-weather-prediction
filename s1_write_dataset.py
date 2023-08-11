import hydra
from omegaconf import DictConfig

from WD.datasets import write_conditional_datasets
from WD.utils import generate_uid

@hydra.main(version_base=None, config_path="/data/compoundx/WeatherDiff/config/data", config_name="config")
def main(conf: DictConfig) -> None:
    write_conditional_datasets(conf)

if __name__ == '__main__':
    main()
