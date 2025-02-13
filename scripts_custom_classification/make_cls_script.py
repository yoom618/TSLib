import os
import math
from omegaconf import OmegaConf
import itertools

def make_combination(config_dict):
    keys, values = zip(*config_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


if __name__ == "__main__":
    script_dir = "scripts_custom_classification"
    data_metainfo = "data_classification.yaml"
    script_path = f"{script_dir}/scripts_all/MambaSL_{{}}.sh"

    dir_setting = {
        "data_dir" : "/data/yoom618/TSLib/dataset",
        "checkpoints": "/data/yoom618/TSLib/checkpoints",
    }

    data_configs = OmegaConf.load(f"{script_dir}/{data_metainfo}")

    gpu_setting = {
        "use_gpu" : True,
        "gpu_type" : "cuda",
        # "gpu" : 0
    }
    # gpu_setting = {
    #     "use_multi_gpu" : True,
    #     "devices" : "1,2"
    # }


    model_id = "TV_{}"
    model = "MambaSingleLayer"
    model_configs = {
        "d_model" : [32, 64, 128, 256, 512, 1024],
        "expand" : [1, 2],
        "d_conv" : [4],
        "tv_dt" : [1, 0],   # 1: True, 0: False
        "tv_B" : [1, 0],    # 1: True, 0: False
        "tv_C" : [1, 0],    # 1: True, 0: False
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,
        "weight_decay" : 0.0001,
        "train_epochs" : 200,
        "patience" : 20,
    }

    os.makedirs(f"{script_dir}/scripts_all", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)
    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        # Fix kernel size of embedding layer into 5% of sequence length
        model_configs["num_kernels"] = [max(1, data_cfg.seq_len // 20)]
        # Set d_state from log2(num_class) to log2(num_class) + 2
        model_configs["d_ff"] = [math.ceil(math.log2(data_cfg.num_class)) + i for i in range(3)]
        # model_configs["d_ff"] = [math.ceil(math.log2(data_cfg.num_class))]

        if data_cfg["dataset"][0].lower() >= "i":
            gpu_setting["gpu"] = 0
        else:
            gpu_setting["gpu"] = 1
        del data_cfg["num_class"], data_cfg["dataset"]

        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]

        model_configs_combination = make_combination(model_configs)
        
        for model_cfg in model_configs_combination:
            script_cfg = gpu_setting
            script_cfg.update(training_configs)
            script_cfg.update(data_cfg)
            script_cfg["model"] = model
            script_cfg["model_id"] = model_id.format(data_key)
            script_cfg.update(model_cfg)

            script = f"python run.py "
            for key, value in script_cfg.items():
                script += f"--{key} {value} "
            script += "\n\n"
            scripts += script
            print(script)

        with open(script_path.format(data_key), "w") as f:
            f.write(scripts)