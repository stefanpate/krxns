import yaml
from pathlib import Path

project_dir = Path(__file__).parent.parent

with open(project_dir / "config.yaml", 'r') as f:
    configs = yaml.safe_load(f)

sim_mats_filepath = Path(configs["FILEPATHS"]['sim_mats'])
cofactors_filepath = Path(configs["FILEPATHS"]['cofactors'])
data_filepath = Path(configs["FILEPATHS"]['data'])
brics_sim_mats_filepath = Path(configs["FILEPATHS"]['brics_sim_mats'])