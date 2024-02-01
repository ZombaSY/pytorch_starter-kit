import torch
import time
import numpy as np
import os
import cv2

from models import metrics
from models import utils
from models import dataloader as dataloader_hub
from trainer_base import TrainerBase

from torch.nn import functional as F


class Inferencer:

    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.loader_val = self.init_data_loader(batch_size=1,
                                                mode='validation',
                                                dataloader_name=self.args.dataloader,
                                                x_path=self.args.val_x_path,
                                                y_path=self.args.val_y_path,
                                                csv_path=self.args.val_csv_path)
        
        self.model = TrainerBase.init_model(self.args.model_name, self.device, self.args)
        self.model.load_state_dict(torch.load(args.model_path))
        self.model.eval()

        dir_path, fn = os.path.split(self.args.model_path)
        fn, ext = os.path.splitext(fn)

        self.img_save_dir = os.path.join(dir_path, fn)
        self.num_batches_val = int(len(self.loader_val))
        self.metric = self.init_metric(self.args.task, self.args.num_class)

        self.image_mean = self.loader_val.image_loader.image_mean
        self.image_std = self.loader_val.image_loader.image_std

    def start_inference_classification(self):
        for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
            with torch.no_grad():
                x_in, img_id = x_in
                target, _ = target

                x_img = x_in
                x_in = x_in.to(self.device)
                target = target.to(self.device)  # (shape: (batch_size, img_h, img_w))

                output, _ = self.model(x_in)

                self.post_process(output, target, x_img, img_id, draw_results=self.args.draw_results)

                print(f'batch {batch_idx} -> {img_id} \t Done !!')

        metrics_out = self.metric.get_results()
        mean_kappa_score = metrics_out['Mean Kappa Score']
        kappa_scores = metrics_out['Class Kappa Score']
        mean_acc_score = metrics_out['Mean Accuracy']
        acc_scores = metrics_out['Class Accuracy']

        print(f'Mean Kappa Score : {mean_kappa_score} \n'
              f'Mean Accuracy : {mean_acc_score}')

        for i in range(self.args.num_class):
            print(f'\t Val Class Kappa Score {i} : {kappa_scores[i]}')
            print(f'\t Val Class Accuracy {i} : {acc_scores[i]}')

    def start_inference_segmentation(self):
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

    def __post_process_segmentation(self, x_img, target, output, img_id, post_out, draw_results):
        # compute metric
        if self.args.dataloader == 'Image2Image':
            output_argmax = torch.argmax(output['seg'], dim=1).cpu()
            self.metric.update(target[0][0].cpu().detach().numpy(), output_argmax[0].cpu().detach().numpy())
            
        if draw_results:
            output_prob = F.softmax(output['seg'][0], dim=0)
            utils.draw_image(x_img, output_prob, self.img_save_dir, img_id, self.args.num_class)

        return post_out
    
    def __post_process(self, x_img, target, output, img_id, post_out, batch_idx, draw_results=False):
        post_out_tmp = {}
        post_out_tmp['img_id'] = img_id

        if draw_results:
            x_img = utils.denormalize_img(x_img, self.image_mean, self.image_std)

        if self.args.task == 'regression':
            self.__post_process_segmentation(x_img, target, output, img_id, post_out_tmp, draw_results)

        for key in post_out_tmp.keys():
            if key not in post_out.keys():
                post_out[key] = [post_out_tmp[key]]
            else:
                post_out[key].append(post_out_tmp[key])

        print(f'batch_idx {batch_idx} -> {img_id} \t Done !!')

        return post_out

    def init_data_loader(self,
                         batch_size,
                         mode,
                         dataloader_name,
                         x_path=None,
                         y_path=None,
                         csv_path=None):

        if dataloader_name == 'Image2Image':
            loader = dataloader_hub.Image2ImageDataLoader(x_path=x_path,
                                                          y_path=y_path,
                                                          batch_size=batch_size,
                                                          num_workers=self.args.worker,
                                                          pin_memory=self.args.pin_memory,
                                                          mode=mode,
                                                          args=self.args)
        elif self.args.dataloader == 'Image2Vector':
            loader = dataloader_hub.Image2VectorDataLoader(csv_path=csv_path,
                                                           batch_size=batch_size,
                                                           num_workers=self.args.worker,
                                                           pin_memory=self.args.pin_memory,
                                                           mode=mode,
                                                           args=self.args)
        else:
            raise Exception('No dataloader named', dataloader_name)

        return loader

    def init_metric(self, task_name, num_class):
        if task_name == 'segmentation':
            metric = metrics.StreamSegMetrics_segmentation(num_class)
        elif task_name == 'classification':
            metric = metrics.StreamSegMetrics_classification(num_class)
        else:
            raise Exception('No task named', task_name)

        return metric

    def inference(self):
        if self.args.task == 'segmentation':
            self.start_inference_segmentation()
