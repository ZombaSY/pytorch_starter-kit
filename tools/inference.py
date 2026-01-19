import torch
import time
import numpy as np
import os
import pandas as pd
import multiprocessing
import itertools
import logging

from models import utils
from tools import utils_tool
from models.backbones.MobileOne import reparameterize_model
from torch.nn import functional as F


class Inferencer:
    def __init__(self, conf):
        self.conf = conf

        use_cuda = self.conf['env']['cuda'] and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.saved_model_directory = os.path.split(self.conf['model']['saved_ckpt'])[0]
        utils.Logger('log.txt', level=logging.DEBUG if self.conf['env']['debug'] else logging.INFO)


        self.loader_valid = utils_tool.init_data_loader(conf=self.conf,
                                                        conf_dataloader=self.conf['dataloader_valid'])

        self.criterion = utils_tool.init_criterion(self.conf['criterion'], self.device)

        self.model = utils_tool.init_model(self.conf, self.device)
        self.model.module.load_state_dict(torch.load(self.conf['model']['saved_ckpt']), strict=True)

        self.model = reparameterize_model(self.model)
        self.model.eval()

        dir_path, fn = os.path.split(self.conf['model']['saved_ckpt'])
        fn, ext = os.path.splitext(fn)

        self.save_dir = os.path.join(dir_path, fn)
        self.num_batches_val = int(len(self.loader_valid))
        self.metric_val = utils_tool.init_metric(self.conf['env']['task'], self.conf['model']['num_class'])

        self.image_mean = torch.tensor(self.loader_valid.image_loader.image_mean).to(self.device)
        self.image_std = torch.tensor(self.loader_valid.image_loader.image_std).to(self.device)

        self.data_stat = {}
        self.level_converter = utils.LevelConverterTexture()

    def inference_classification(self, epoch):
        self.model.eval()

        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data['input']
                target = data['label']

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

            self.metric_val.reset()

    def inference_segmentation(self, epoch):
        self.model.eval()

        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data['input']
                target = data['label']
                img_id = data['input_path']

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))
                output = self.model(x_in)

                self.__post_process(x_in, target, output, img_id, iteration)

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
        list_score = []
        list_target = []

        for iteration, data in enumerate(self.loader_valid.Loader):
            with torch.no_grad():
                x_in = data['input']
                img_id = data['input_path']
                target = data['label'].to(self.device) if 'label' in data.keys() else None

                x_in = x_in.to(self.device)

                output = self.model(x_in)

                # compute loss
                if self.conf['env']['mode'] == 'valid':
                    loss = self.criterion(output['vec'], target)
                    if not torch.isfinite(loss):
                        raise Exception('Loss is NAN. End training.')
                    batch_losses += loss.item()

                target_item = target.cpu().numpy()
                score_item = output['vec'].detach().cpu().numpy()
                for b in range(output['vec'].shape[0]):
                    list_score.append(score_item[b].tolist())
                    list_target.append(target_item[b].tolist())

                self.__post_process(x_in, target, output, img_id, iteration)

        # correlation = np.corrcoef(np.array(list_score).T[0], np.array(list_target).T[0])[0, 1]

        if self.conf['env']['mode'] == 'valid':
            loss_mean = batch_losses / self.loader_valid.Loader.__len__()

            metric_dict = {}
            metric_dict['loss'] = loss_mean
            # metric_dict['correlation'] = correlation

            utils.log_epoch('validation', epoch, metric_dict, False)
            self.metric_val.reset()

    def __post_process_regression(self, x_img, target, output, img_id):
        for idx in range(x_img.shape[0]):
            utils.append_data_stats(self.data_stat, 'img_id', img_id[idx])
            for i in range(output['vec'].shape[-1]):
                key_name = 'predict_' + str(i).zfill(2)
                utils.append_data_stats(self.data_stat, key_name, output['vec'][idx][i].detach().cpu().item())

                # log recall
                # target_dis = self.conf['env']['target_disease']
                # out1 = self.level_converter.convert_score(output['vec'][idx][i].detach().cpu().item(), target_disease=target_dis, is_reversed=True)
                # out2 = self.level_converter.convert_level(output['vec'][idx][i].detach().cpu().item(), target_disease=target_dis, is_reversed=True)
                # utils.append_data_stats(self.data_stat, target_dis + '_raw', output['vec'][idx][i].detach().cpu().item())
                # utils.append_data_stats(self.data_stat, target_dis + '_score', out1)
                # utils.append_data_stats(self.data_stat, target_dis + '_level', out2)

    def __post_process_landmark(self, x_img, target, output, img_id):
        if self.conf['env']['draw_results']:

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            with multiprocessing.Pool(4) as pools:
                x_img = utils.denormalize_img(x_img, self.image_mean, self.image_std).detach().cpu().numpy()
                output_np = output['vec'].detach().cpu().numpy()
                pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(utils.draw_landmark), x_img, output_np, itertools.repeat(self.save_dir), img_id))

    def __post_process_segmentation(self, x_img, target, output, img_id):
        # compute metric
        if self.conf['dataloader_valid']['name'] == 'Image2Image' and self.conf['env']['mode'] == 'valid':
            output_argmax = torch.argmax(output['seg'], dim=1).cpu()
            for b in range(output['seg'].shape[0]):
                self.metric_val.update(target[b][0].cpu().detach().numpy(), output_argmax[b].cpu().detach().numpy())

        if self.conf['env']['draw_results']:
            with multiprocessing.Pool(4) as pools:
                x_img = utils.denormalize_img(x_img, self.image_mean, self.image_std).detach().cpu().numpy()
                output_prob = F.softmax(output['seg'], dim=1).detach().cpu().numpy()
                pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(utils.draw_image), x_img, output_prob, itertools.repeat(self.save_dir), img_id, itertools.repeat(self.conf['model']['num_class'])))

        for idx in range(x_img.shape[0]):
            output_argmax = torch.argmax(output['seg'][idx], dim=0)
            utils.append_data_stats(self.data_stat, 'img_id', img_id[idx])
            for i in range(self.conf['model']['num_class']):
                key_name = 'segmap_ratio_' + str(i).zfill(2)
                utils.append_data_stats(self.data_stat, key_name, len(torch.where(output_argmax == i)[0]) / (output['seg'][idx].shape[-2] * output['seg'][idx].shape[-1]))

    def __post_process(self, x_img, target, output, img_id, iteration):
        if self.conf['env']['task'] == 'segmentation':
            self.__post_process_segmentation(x_img, target, output, img_id)
        elif self.conf['env']['task'] == 'landmark':
            self.__post_process_landmark(x_img, target, output, img_id)
        elif self.conf['env']['task'] == 'regression':
            self.__post_process_regression(x_img, target, output, img_id)

        print(f'iteration {iteration} -> {(iteration + 1) * self.conf["dataloader_valid"]["batch_size"]} images done !!')     # TODO: last iteration is invalid

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
