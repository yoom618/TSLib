import math
from omegaconf import OmegaConf
import itertools

def make_combination(config_dict):
    keys, values = zip(*config_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


if __name__ == "__main__":
    script_dir = "scripts_custom_classification"
    script_path = f"{script_dir}/scripts_all/MambaSL_{{}}.sh"

    gpu_setting = {
        "use_gpu" : True,
        # "gpu" : 1,
        "gpu_type" : "cuda"
    }
    # gpu_setting = {
    #     "use_multi_gpu" : True,
    #     "devices" : "1,2"
    # }


    data_configs = OmegaConf.load(f"{script_dir}/data_classification.yaml")

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

    for data_key, data_cfg in data_configs.items():
        scripts = ''

        model_configs["num_kernels"] = [min(1, data_cfg.seq_len // 20)]
        model_configs["d_ff"] = [math.ceil(math.log2(data_cfg.num_class)) + i for i in range(3)]
        # model_configs["d_ff"] = [math.ceil(math.log2(data_cfg.num_class))]

        if data_cfg["dataset"][0].lower() >= "i":
            gpu_setting["gpu"] = 0
        else:
            gpu_setting["gpu"] = 1
        del data_cfg["num_class"], data_cfg["dataset"]
        
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