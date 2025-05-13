import torch
import torch.nn as nn

# from utils.shapelet_util import ModelInfo
from models.Shapelet import ShapeBottleneckModel
from models.FullyConvNet import Model as FullyConvNetwork
# from models.PatchTST import Model as PatchTST
# from models.TimesNet import Model as TimesNet
# from models.Transformer import Model as Transformer
# from models.ResNet import Model as ResNet


dnn_dict = {
    # 'PatchTST': PatchTST,
    'FCN': FullyConvNetwork,
    # 'TimesNet': TimesNet,
    # 'Transformer': Transformer,
    # 'ResNet': ResNet
}


class Model(nn.Module):
    def __init__(
            self,
            configs,
        ):
        super().__init__()

        shapelet_lengths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
        num_shapelet = [configs.num_shapelet] * len(shapelet_lengths)        
        
        self.configs = configs
        self.sbm = ShapeBottleneckModel(
            configs=configs,
            num_shapelet=num_shapelet,
            shapelet_len=shapelet_lengths
        )
        self.deep_model = dnn_dict[configs.dnn_type](configs)
    
    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, gating_value=None):
        sbm_out = self.sbm(x)
        deep_out = self.deep_model(x, x_mark_enc, x_dec, x_mark_dec, mask)

        # Gini Index: compute the gating value 
        p = nn.functional.softmax(sbm_out, dim=-1)
        c = sbm_out.shape[-1]
        gini = p.pow(2).sum(-1, keepdim=True)
        sbm_util = (c * gini - 1)/(c-1)
        if gating_value is not None:
            mask = (sbm_util > gating_value).float()
            sbm_util = torch.ones_like(sbm_util) * mask + sbm_util * (1 - mask)
        deep_util = torch.ones_like(sbm_util) - sbm_util
        output = sbm_util * sbm_out + deep_util * deep_out

        # return output, ModelInfo(d=model_info.d, 
        #                          p=model_info.p,
        #                          eta=sbm_util,
        #                          shapelet_preds=sbm_out,
        #                          dnn_preds=deep_out,
        #                          preds=output,
        #                          loss=self.loss().unsqueeze(0))
        return output

    def loss(self):
        return self.sbm.loss()
    
    def step(self):
        self.sbm.step()