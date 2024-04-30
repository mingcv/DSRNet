import os
from os.path import join

import data.sirs_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils

opt = TrainOptions().parse()
print(opt)

opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 1
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 9999
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# modify the following code to
datadir = os.path.join(opt.base_dir)
datadir_syn = join(datadir, 'train/VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'train/real')

train_dataset = datasets.DSRDataset(
    datadir_syn, read_fns('data/VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True)

train_dataset_real = datasets.DSRTestDataset(datadir_real, enable_transforms=True, if_align=opt.if_align)

train_dataset_fusion = datasets.FusionDataset([train_dataset,
                                               train_dataset_real], [0.7, 0.3])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    pin_memory=True, prefetch_factor=32, num_workers=32)

eval_dataset_real = datasets.DSRTestDataset(join(datadir, f'test/real20_{opt.real20_size}'),
                                            fns=read_fns('data/real_test.txt'), if_align=opt.if_align)
eval_dataset_solidobject = datasets.DSRTestDataset(join(datadir, 'test/SIR2/SolidObjectDataset'),
                                                   if_align=opt.if_align)
eval_dataset_postcard = datasets.DSRTestDataset(join(datadir, 'test/SIR2/PostcardDataset'), if_align=opt.if_align)
eval_dataset_wild = datasets.DSRTestDataset(join(datadir, 'test/SIR2/WildSceneDataset'), if_align=opt.if_align)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    pin_memory=True, prefetch_factor=32, num_workers=32)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject, batch_size=1, shuffle=False,
    pin_memory=True, prefetch_factor=32, num_workers=32)
eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard, batch_size=1, shuffle=False,
    pin_memory=True, prefetch_factor=32, num_workers=32)

eval_dataloader_wild = datasets.DataLoader(
    eval_dataset_wild, batch_size=1, shuffle=False,
    pin_memory=True, prefetch_factor=32, num_workers=32)

"""Main Loop"""
engine = Engine(opt)
result_dir = os.path.join(f'./checkpoints/{opt.name}/results',
                          mutils.get_formatted_time())


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)



if opt.resume or opt.debug_eval:
    save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
    os.makedirs(save_dir, exist_ok=True)
    engine.save_model()

    engine.eval(eval_dataloader_real, dataset_name='testdata_real20',
                savedir=save_dir, suffix='real20', max_save_size=10)
    engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject',
                savedir=save_dir, suffix='solidobject', max_save_size=10)
    engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard',
                savedir=save_dir, suffix='postcard', max_save_size=10)
    engine.eval(eval_dataloader_wild, dataset_name='testdata_wild',
                savedir=save_dir, suffix='wild', max_save_size=10)
# define training strategy
set_learning_rate(opt.lr)
while engine.epoch < 50:
    print('random_seed: ', opt.seed)
    engine.train(train_dataloader_fusion)

    if engine.epoch % 1 == 0:
        save_dir = os.path.join(result_dir, '%03d' % engine.epoch)
        os.makedirs(save_dir, exist_ok=True)
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20',
                    savedir=save_dir, suffix='real20', max_save_size=10)
        engine.eval(eval_dataloader_solidobject, dataset_name='testdata_solidobject',
                    savedir=save_dir, suffix='solidobject', max_save_size=10)
        engine.eval(eval_dataloader_postcard, dataset_name='testdata_postcard',
                    savedir=save_dir, suffix='postcard', max_save_size=10)
        engine.eval(eval_dataloader_wild, dataset_name='testdata_wild',
                    savedir=save_dir, suffix='wild', max_save_size=10)
