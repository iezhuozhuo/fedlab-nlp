import os
import sys
from loguru import logger

machine_name = sys.argv[1]
if machine_name.startswith("pcl"):
    if "124" in machine_name:
        run_dir = "/raid2/zhuo/code/fednlp"
    else:
        run_dir = "/raid1/zhuo/code/fednlp"
elif machine_name.startswith("uestc"):
    run_dir = "/hdd/zhuo/code/fednlp"
elif machine_name.startswith("hit"):
    run_dir = "/data/zhuo/code/fednlp"
else:
    run_dir = None
    raise ValueError("please check your machine name")
sys.path.append(run_dir)

from globalhost import machine_dict

# run_dir = "/".join(os.path.abspath(sys.argv[0]).split("/")[0:-3])
if not machine_dict.get(machine_name, None):
    logger.critical(f"not that {machine_name}")

for key, value in machine_dict[machine_name].items():
    logger.debug(f"make dir {value}")
    os.makedirs(value, exist_ok=True)