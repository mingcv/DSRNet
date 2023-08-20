import argparse
import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default=None,
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='errnet_model', help='chooses which model to use.')
        self.parser.add_argument('--dataset', '-d', type=str, default='sirs_dataset',
                                 help='chooses which dataset to use.')
        self.parser.add_argument('--loss', '-l', type=str, default='losses',
                                 help='chooses which loss to use.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
        self.parser.add_argument('--resume_epoch', '-re', type=int, default=None,
                                 help='checkpoint to use. (default: latest')
        self.parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=0')
        self.parser.add_argument('--supp_eval', action='store_true', help='supplementary evaluation')
        self.parser.add_argument('--start_now', action='store_true', help='supplementary evaluation')
        self.parser.add_argument('--testr', action='store_true', help='test for reflections')
        self.parser.add_argument('--select', type=str, default=None)

        # for setting input
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=None,
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for display
        self.parser.add_argument('--no-log', action='store_true', help='disable tf logger?')
        self.parser.add_argument('--no-verbose', action='store_true', help='disable verbose info?')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0,
                                 help='window id of the web display (use 0 to disable visdom)')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')

        self.initialized = True
