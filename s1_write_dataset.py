import hydra
from omegaconf import DictConfig

from WD.datasets import write_conditional_datasets
from WD.utils import generate_uid

@hydra.main(version_base=None, config_path="./config", config_name="data")
def main(conf: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    template_name = hydra_cfg['runtime']['choices']['template']
    write_conditional_datasets(conf, template_name)

if __name__ == '__main__':
    main()
