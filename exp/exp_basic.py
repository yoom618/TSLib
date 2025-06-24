import os
import torch
from models import TimesNet, DLinear, FEDformer, \
    LightTS, ETSformer, PatchTST, Crossformer, \
    MambaSimple, \
    MTSMixer, GPT4TS, ModernTCN, TimeMixerPP, InterpretGN
# ### UNUSED 
# from models import Autoformer, Transformer, Nonstationary_Transformer, \
#     Informer, Reformer, Pyraformer, MICN, FiLM, iTransformer, \
#     Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, TemporalFusionTransformer, SCINet, PAttn, TimeXer, WPMixer, 


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            # 'Informer': Informer,
            'LightTS': LightTS,
            # 'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            # 'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            'Crossformer': Crossformer,
            # 'FiLM': FiLM,
            # 'iTransformer': iTransformer,
            # 'Koopa': Koopa,
            # 'TiDE': TiDE,
            # 'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'Mamba': Mamba,
            # 'TimeMixer': TimeMixer,
            # 'TSMixer': TSMixer,
            # 'SegRNN': SegRNN,
            # 'TemporalFusionTransformer': TemporalFusionTransformer,
            # "SCINet": SCINet,
            # 'PAttn': PAttn,
            # 'TimeXer': TimeXer,
            # 'WPMixer': WPMixer,
            'MTSMixer': MTSMixer,
            'GPT4TS': GPT4TS,
            'ModernTCN': ModernTCN,
            'TimeMixerPP': TimeMixerPP,
            'InterpretGN': InterpretGN,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self.args.device
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
