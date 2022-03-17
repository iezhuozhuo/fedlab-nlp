import sys
from loguru import logger


machine_name = sys.argv[1]
data_name = sys.argv[2]
if machine_name.startswith("pcl"):
    if "124" in machine_name:
        run_prefix = "/raid2"
    else:
        run_prefix = "/ghome"
elif machine_name.startswith("uestc"):
    run_prefix = "/hdd"
elif machine_name.startswith("hit"):
    run_prefix = "/data"
else:
    run_prefix = None
    raise ValueError("please check your machine name")
run_dir = run_prefix + "/zhuo/code/fednlp"
logger.debug(f"run dir in {run_dir}")
sys.path.append(run_dir)

from data.raw_data_loader.SST_2.data_loader import RawDataLoader

data_path = f"{run_prefix}/zhuo/data/glue_data/SST-2/"
h5_file_path = f"{run_prefix}/zhuo/data/fednlp_data/data_files/sst_2_data.h5"

raw_data_loader = RawDataLoader(data_path)
print(f"generate into {h5_file_path}")
raw_data_loader.generate_h5_file(h5_file_path)
