import os
import torch
import util.util as util
from tools import mutils


class BaseModel:
    def name(self):
        return self.__class__.__name__.lower()

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        last_split = opt.checkpoints_dir.split('/')[-1]
        if opt.resume and last_split != 'checkpoints' and (last_split != opt.name or opt.supp_eval):

            self.save_dir = opt.checkpoints_dir
            self.model_save_dir = os.path.join(opt.checkpoints_dir.replace(opt.checkpoints_dir.split('/')[-1], ''),
                                               opt.name, 'weights', mutils.get_formatted_time())
        else:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            self.model_save_dir = os.path.join(opt.checkpoints_dir, opt.name, 'weights', mutils.get_formatted_time())
        self._count = 0

    def set_input(self, input):
        self.input = input

    def forward(self, mode='train'):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def print_optimizer_param(self):
        print(self.optimizers[-1])

    def save(self, label=None):
        epoch = self.epoch
        iterations = self.iterations

        os.makedirs(self.model_save_dir, exist_ok=True)
        if label is None:
            model_name = os.path.join(self.model_save_dir, self.opt.name + '_%03d_%08d.pt' % ((epoch), (iterations)))
        else:
            model_name = os.path.join(self.model_save_dir, self.opt.name + '_' + label + '.pt')

        torch.save(self.state_dict(), model_name)

    def save_eval(self, label=None):
        model_name = os.path.join(self.model_save_dir, label + '.pt')

        torch.save(self.state_dict_eval(), model_name)

    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)
