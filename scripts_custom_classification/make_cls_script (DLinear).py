# Crossformer script generation
# hyperparameters setting are referenced from below:
# 1) time-series-library


import os
from omegaconf import OmegaConf
import itertools

def make_combination(config_dict):
    keys, values = zip(*config_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


if __name__ == "__main__":
    script_dir = "./scripts_custom_classification"
    data_metainfo = "data_classification.yaml"
    script_path = f"{script_dir}/scripts_baseline/{{}}.sh"
    model_id = "{}"
    model = "DLinear"

    dir_setting = {
        "data_dir" : "/data/yoom618/TSLib/dataset",
        "checkpoints": "/data/yoom618/TSLib/checkpoints",
    }

    data_configs = OmegaConf.load(f"{script_dir}/{data_metainfo}")

    gpu_setting = {
        "use_gpu" : True,
        "gpu_type" : "cuda",
        "gpu" : 1,
    }
    # gpu_setting = {
    #     "use_multi_gpu" : True,
    #     "devices" : "1,2"
    # }


    model_configs = {
        "e_layers" : [3],
        "d_model" : [128],
        "d_ff" : [256],
        "top_k" : [3],
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,
        "train_epochs" : 200,  # increased from 100 to 200 for alignment with MambaSL
        "patience" : 20,  # increased from 10 to 20 for alignment with MambaSL
    }
    
    os.makedirs(f"{script_dir}/scripts_all", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)
    
    scripts = ''
    
    for data_key, data_cfg in data_configs.items():
        model_configs_combination = make_combination(model_configs)

        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]

        
        for model_cfg in model_configs_combination:
            script_cfg = gpu_setting
            script_cfg.update(training_configs)
            script_cfg.update(data_cfg)
            script_cfg["model"] = model
            script_cfg["model_id"] = model_id.format(data_key)
            script_cfg.update(model_cfg)

            script = f"python run.py \n"
            for key, value in script_cfg.items():
                script += f"  --{key} {value} \\ \n"
            script = script[:-3]
            script += "\n\n"
            scripts += script
            print(script)

    with open(script_path.format(model), "w") as f:
        f.write(scripts)