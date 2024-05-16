import torch
import time
import numpy as np
import os
import pandas as pd
import multiprocessing
import itertools
import cv2

from models import utils
from trainer_base import TrainerBase

from torch.nn import functional as F


class Inferencer:

    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.loader_val = TrainerBase.init_data_loader(args=self.args,
                                                       x_path=self.args.valid_x_path,
                                                       y_path=self.args.valid_y_path,
                                                       csv_path=self.args.valid_csv_path)
        self.criterion = TrainerBase.init_criterion(self.args, self.device)

        self.model = TrainerBase.init_model(self.args, self.device)
        self.model.module.load_state_dict(torch.load(args.model_path))
        self.model.eval()

        dir_path, fn = os.path.split(self.args.model_path)
        fn, ext = os.path.splitext(fn)

        self.save_dir = os.path.join(dir_path, fn)
        self.num_batches_val = int(len(self.loader_val))
        self.metric_val = TrainerBase.init_metric(self.args.task, self.args.num_class)

        self.image_mean = torch.tensor(self.loader_val.image_loader.image_mean).to(self.device)
        self.image_std = torch.tensor(self.loader_val.image_loader.image_std).to(self.device)

    def inference_classification(self, epoch):
        self.model.eval()

        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, _ = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)

                output_argmax = torch.argmax(output['class'], dim=1).detach().cpu().numpy()
                target_argmax = torch.argmax(target.squeeze(), dim=1).detach().cpu().numpy() if self.args.mode == 'inference' else np.zeros_like(output_argmax)
                self.metric_val.update(output_argmax, target_argmax)

        if self.args.mode == 'inference':
            metric_list_mean = {}
            metric_result = self.metric_val.get_results()
            metric_list_mean['acc'] = metric_result['acc']
            metric_list_mean['f1'] = metric_result['f1']

            for key in metric_list_mean.keys():
                metric_list_mean[key] = np.mean(metric_list_mean[key])

            for key in metric_list_mean.keys():
                log_str = f'validation {key}: {metric_list_mean[key]}'
                print(f'{utils.Colors.LIGHT_GREEN} {log_str} {utils.Colors.END}')

        df = pd.DataFrame({'fn': self.loader_val.image_loader.df['img_path'],
                           'label': self.metric_val.get_pred_flatten()})
        df.to_csv(self.save_dir + '_out.csv', encoding='utf-8-sig', index=False)
        self.metric_val.reset()

    def inference_segmentation(self, epoch):
        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, img_id = x_in
                target, _ = target

                x_img = x_in
                x_in = x_in.to(self.device)
                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                target = target.long().to(self.device)
                path, fn = os.path.split(img_id[0])
                img_id, ext = os.path.splitext(fn)
                output, _ = self.model(x_in)
                self.post_process(x_img, target, output, img_id, draw_results=self.args.draw_results)

                print(f'batch {batch_idx} -> {img_id} \t Done !!')

        # get metrics, computed on 'self.post_process'
        metrics_out = self.metric.get_results()
        cIoU = [metrics_out['Class IoU'][i] for i in range(self.args.num_class)]
        mIoU = sum(cIoU) / self.args.num_class
        print('Val mIoU: {}'.format(mIoU))
        for i in range(self.args.num_class):
            print(f'Val Class {i} IoU: {cIoU[i]}')

    def inference_regression(self, epoch):
        self.model.eval()
        post_out = {}
        metric_list_mean = {'loss': []}
        batch_losses = 0

        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, img_id = x_in
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)

                # compute loss
                loss = self.criterion(output['class'], target)
                if not torch.isfinite(loss):
                    raise Exception('Loss is NAN. End training.')

                batch_losses += loss.item()

                self.__post_process(x_in, target, output, img_id, post_out, batch_idx)

        loss_mean = batch_losses / self.loader_val.Loader.__len__()
        metric_list_mean['loss'] = loss_mean
        for key in metric_list_mean.keys():
            log_str = f'validation {key}: {metric_list_mean[key]}'
            print(f'{utils.Colors.LIGHT_GREEN} {epoch} epoch / {log_str} {utils.Colors.END}')

        self.metric_val.reset()

    def __post_process_landmark(self, x_img, target, output, img_id, post_out):
        if self.args.draw_results:

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                x_img = utils.denormalize_img(x_img, self.image_mean, self.image_std)
                x_img_np = x_img.detach().cpu().numpy()
                output_np = output['class'].detach().cpu().numpy()

                pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(utils.draw_landmark), x_img_np, output_np, itertools.repeat(self.save_dir), img_id))

        return post_out

    def __post_process_segmentation(self, x_img, target, output, img_id, post_out):
        # compute metric
        if self.args.dataloader == 'Image2Image':
            output_argmax = torch.argmax(output['seg'], dim=1).cpu()
            self.metric.update(target[0][0].cpu().detach().numpy(), output_argmax[0].cpu().detach().numpy())

        if self.args.draw_results:

            with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pools:
                x_img = utils.denormalize_img(x_img, self.image_mean, self.image_std)
                output_prob = F.softmax(output['seg'], dim=1)

                pools.map(utils.multiprocessing_wrapper, zip(itertools.repeat(utils.draw_image), x_img, output_prob, itertools.repeat(self.save_dir), img_id), itertools.repeat(self.args.num_class))

        return post_out

    def __post_process(self, x_img, target, output, img_id, post_out, batch_idx):
        post_out_tmp = {}
        post_out_tmp['img_id'] = img_id

        if self.args.task == 'segmentation':
            self.__post_process_segmentation(x_img, target, output, img_id, post_out_tmp)
        elif self.args.task == 'landmark':
            self.__post_process_landmark(x_img, target, output, img_id, post_out_tmp)

        for key in post_out_tmp.keys():
            if key not in post_out.keys():
                post_out[key] = [post_out_tmp[key]]
            else:
                post_out[key].append(post_out_tmp[key])

        print(f'batch_idx {batch_idx} -> {batch_idx * self.args.batch_size} images \t Done !!')     # TODO: last batch_idx is invalid

        return post_out

    def inference(self):
        if self.args.task == 'segmentation':
            self.inference_segmentation(0)
        elif self.args.task == 'classification':
            self.inference_classification(0)
        elif self.args.task == 'regression' or 'landmark':
            self.inference_regression(0)
