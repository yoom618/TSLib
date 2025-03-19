import os
import math
from omegaconf import OmegaConf
import itertools

def make_combination(config_dict):
    keys, values = zip(*config_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


if __name__ == "__main__":
    script_dir = "./scripts_custom_classification"
    data_metainfo = "data_classification.yaml"
    script_path = f"{script_dir}/scripts_mamba/MambaSL_{{}}-dff.sh"
    model_id = "{}"
    model = "MambaSingleLayer"

    dir_setting = {
        "data_dir" : "/data/yoom618/TSLib/dataset",
        "checkpoints": "/data/yoom618/TSLib/checkpoints",
    }

    data_configs = OmegaConf.load(f"{script_dir}/{data_metainfo}")

    gpu_setting = {
        "use_gpu" : True,
        "gpu_type" : "cuda",
        "gpu" : 0
    }
    # gpu_setting = {
    #     "use_multi_gpu" : True,
    #     "devices" : "1,2"
    # }


    model_configs = {
        "d_model" : [64, 128, 256, 512, 1024],
        "expand" : [1, 2],
        "d_conv" : [4],
        "tv_dt" : [0, 1],   # 0: False, 1: True
        "tv_B" : [0, 1],    # 0: False, 1: True
        "tv_C" : [0, 1],    # 0: False, 1: True
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

    os.makedirs(f"{script_dir}/scripts_mamba", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)
    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        ## Fix kernel size to 5% of sequence length in Embedding step
        model_configs["num_kernels"] = [max(1, math.ceil(data_cfg.seq_len / 20))]

        # # Fix kernel size of data embedding to 3
        # model_configs["num_kernels"] = [3]
        # model_configs["d_conv"] = [max(1, math.ceil(data_cfg.seq_len / 20))]


        # # Set d_state(=d_ff) from log2(1/p_min) - 1  to  log2(1/p_min) + 1
        # model_configs["d_ff"] = [max(2, math.ceil(math.log2(1/data_cfg.p_min))) + i for i in range(-1, 2)]
        
        # # Set d_state(=d_ff) from log2(num_class) - 1  to  log2(num_class) + 1
        # model_configs["d_ff"] = [max(2, math.ceil(math.log2(data_cfg.num_class))) + i for i in range(-1, 2)]
        
        # Set d_state(=d_ff) from log2(num_class)  to  log2(num_class) * 3
        model_configs["d_ff"] = [math.ceil(math.log2(data_cfg.num_class)) * i for i in range(1,4)]
        

        replace_dict = {}
        if data_cfg["dataset"][0].lower() < "f":
            replace_dict["gpu"] = 0
        elif data_cfg["dataset"][0].lower() < "n":
            replace_dict["gpu"] = 1
        else:
            replace_dict["gpu"] = 2
        if data_cfg["dataset"] == "EigenWorms":
            replace_dict["batch_size"] = 4  # GPU Memory Usage: 9318MiB
            replace_dict["gpu"] = 3
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]

        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]

        model_configs_combination = reversed(make_combination(model_configs))
        
        for model_cfg in model_configs_combination:
            script_cfg = gpu_setting.copy()
            script_cfg.update(data_cfg)
            script_cfg["model"] = model
            script_cfg["model_id"] = model_id.format(data_key)
            script_cfg.update(model_cfg)
            script_cfg.update(training_configs)
            script_cfg.update(replace_dict)

            script = f"python run.py \\\n"
            for key, value in script_cfg.items():
                script += f"  --{key} {value} \\\n"
            script = script[:-3]
            script += "\n\n"
            scripts += script
            print(script)

        with open(script_path.format(data_key), "w") as f:
            f.write(scripts)