import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import logging
import annbench

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config_download")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # https://hydra.cc/docs/tutorial/working_directory#original-working-directory
    dataset = annbench.instantiate_dataset(name=cfg.dataset.name,
                                           path=to_absolute_path(cfg.dataset.path))

    log.info(f"Start to download {cfg.dataset.name} on {cfg.dataset.path}")
    dataset.download()


if __name__ == "__main__":
    main()
