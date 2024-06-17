import torch
import time
import numpy as np
import os
import pandas as pd
import multiprocessing
import itertools

from models import utils
from trainer_base import TrainerBase

from torch.nn import functional as F


class Inferencer:

    def __init__(self, conf):
        self.start_time = time.time()
        self.conf = conf

        use_cuda = self.conf['env']['cuda'] and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.loader_valid = TrainerBase.init_data_loader(conf=self.conf,
                                                         conf_dataloader=self.conf['dataloader_valid'])

        self.criterion = TrainerBase.init_criterion(self.conf['criterion'], self.device)

        self.model = TrainerBase.init_model(self.conf, self.device)
        self.model.module.load_state_dict(torch.load(self.conf['model']['saved_ckpt']))
        self.model.eval()

        dir_path, fn = os.path.split(self.conf['model']['saved_ckpt'])
        fn, ext = os.path.splitext(fn)

        self.save_dir = os.path.join(dir_path, fn)
        self.num_batches_val = int(len(self.loader_valid))
        self.metric_val = TrainerBase.init_metric(self.conf['env']['task'], self.conf['model']['num_class'])

        self.image_mean = torch.tensor(self.loader_valid.image_loader.image_mean).to(self.device)
        self.image_std = torch.tensor(self.loader_valid.image_loader.image_std).to(self.device)

        self.data_stat = {}

    def inference_classification(self, epoch):
        self.model.eval()

        for batch_idx, (x_in, target) in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)

                output_argmax = torch.argmax(output['vec'], dim=1).detach().cpu().numpy()
                target_argmax = torch.argmax(target.squeeze(), dim=1).detach().cpu().numpy() if self.conf['env']['mode'] == 'valid' else np.zeros_like(output_argmax)
                self.metric_val.update(output_argmax, target_argmax)

        if self.conf['env']['mode'] == 'valid':
            metric_dict = {}
            metric_result = self.metric_val.get_results()
            metric_dict['acc'] = metric_result['acc']
            metric_dict['f1'] = metric_result['f1']

            utils.log_epoch('validation', epoch, metric_dict, False)

            df = pd.DataFrame({'fn': self.loader_valid.image_loader.df['img_path'],
                            'label': self.metric_val.get_pred_flatten()})
            df.to_csv(self.save_dir + '_out.csv', encoding='utf-8-sig', index=False)
            self.metric_val.reset()

    def inference_segmentation(self, epoch):
        self.model.eval()

        for batch_idx, (x_in, target) in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in, img_id = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)

                self.__post_process(x_in, target, output, img_id, batch_idx)

        if self.conf['env']['mode'] == 'valid':
            metrics_out = self.metric_val.get_results()
            c_iou = [metrics_out['Class IoU'][i] for i in range(self.conf['model']['num_class'])]
            m_iou = sum(c_iou) / self.conf['model']['num_class']

            metric_dict = {}
            metric_dict['mIoU'] = m_iou
            for i in range(len(c_iou)):
                metric_dict[f'cIoU_{i}'] = c_iou[i]

            utils.log_epoch('validation', epoch, metric_dict, False)
            self.metric_val.reset()


    def inference_regression(self, epoch):
        self.model.eval()
        metric_dict = {'loss': []}
        batch_losses = 0

        for batch_idx, (x_in, target) in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in, img_id = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)

                # compute loss
                if self.conf['env']['mode'] == 'valid':
                    loss = self.criterion(output['vec'], target)
                    if not torch.isfinite(loss):
                        raise Exception('Loss is NAN. End training.')
                    batch_losses += loss.item()

                self.__post_process(x_in, target, output, img_id, batch_idx)

        if self.conf['env']['mode'] == 'valid':
            loss_mean = batch_losses / self.loader_valid.Loader.__len__()

            metric_dict = {}
            metric_dict['loss'] = loss_mean

            utils.log_epoch('validation', epoch, metric_dict, self.conf['env']['wandb'])
            self.metric_val.reset()

    def __post_process_landmark(self, x_img, target, output, img_id, data_stat):
        if self.conf['env']['draw_results']:

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                x_img = utils.denormalize_img(x_img, self.image_mean, self.image_std).detach().cpu().numpy()
                output_np = output['vec'].detach().cpu().numpy()

                pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(utils.draw_landmark), x_img, output_np, itertools.repeat(self.save_dir), img_id))

    def __post_process_segmentation(self, x_img, target, output, img_id, data_stat):
        # compute metric
        if self.conf['dataloader_valid']['name'] == 'Image2Image':
            output_argmax = torch.argmax(output['seg'], dim=1).cpu()
            self.metric_val.update(target[0][0].cpu().detach().numpy(), output_argmax[0].cpu().detach().numpy())

        if self.conf['env']['draw_results']:
            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                x_img = utils.denormalize_img(x_img, self.image_mean, self.image_std).detach().cpu().numpy()
                output_prob = F.softmax(output['seg'], dim=1).detach().cpu().numpy()

                pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(utils.draw_image), x_img, output_prob, itertools.repeat(self.save_dir), img_id, itertools.repeat(self.conf['model']['num_class'])))

        for idx in range(len(output)):
            data_stat['img_id'] = img_id[idx]
            data_stat['darkcircle_ratio'] = len(torch.where(torch.argmax(output['seg'][idx], dim=0) == 1)[0]) / (output['seg'][idx].shape[-2] * output['seg'][idx].shape[-1])
            data_stat['flush_ratio'] = len(torch.where(torch.argmax(output['seg'][idx], dim=0) == 2)[0]) / (output['seg'][idx].shape[-2] * output['seg'][idx].shape[-1])

    def __post_process(self, x_img, target, output, img_id, batch_idx):
        data_stat = {}

        if self.conf['env']['task'] == 'segmentation':
            self.__post_process_segmentation(x_img, target, output, img_id, data_stat)
        elif self.conf['env']['task'] == 'landmark':
            self.__post_process_landmark(x_img, target, output, img_id, data_stat)

        for key in data_stat.keys():
            if key not in self.data_stat.keys():
                self.data_stat[key] = [data_stat[key]]
            else:
                self.data_stat[key].append(data_stat[key])

        print(f'batch_idx {batch_idx} -> {batch_idx * self.conf["dataloader_valid"]["batch_size"]} images \t Done !!')     # TODO: last batch_idx is invalid

    def inference(self):
        if self.conf['env']['task'] == 'segmentation':
            self.inference_segmentation(0)
        elif self.conf['env']['task'] == 'classification':
            self.inference_classification(0)
        elif self.conf['env']['task'] == 'regression' or 'landmark':
            self.inference_regression(0)

        # save meta data
        df = pd.DataFrame(self.data_stat)
        df.to_csv(self.save_dir + '_out.csv', encoding='utf-8-sig', index=False)
