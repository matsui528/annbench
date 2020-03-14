import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import logging
import annbench

log = logging.getLogger(__name__)

@hydra.main(config_path="conf/config_download.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    # https://hydra.cc/docs/tutorial/working_directory#original-working-directory
    dataset = annbench.instantiate_dataset(name=cfg.dataset.name,
                                           path=to_absolute_path(cfg.dataset.path))

    log.info("Start to download {} on {}".format(cfg.dataset.name, cfg.dataset.path))
    dataset.download()


if __name__ == "__main__":
    main()
