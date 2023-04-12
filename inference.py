import torch
import time
import numpy as np
import os
import pandas as pd
import cv2

from models import metrics
from models import utils
from models import dataloader as dataloader_hub
from models import model_implements

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

        self.model = self.init_model(self.args.model_name)
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
                self.post_process(output, target, x_img, img_id, draw_results=self.args.draw_results)

                print(f'batch {batch_idx} -> {img_id} \t Done !!')

        # get metrics, computed on 'self.post_process'
        metrics_out = self.metric.get_results()
        cIoU = [metrics_out['Class IoU'][i] for i in range(self.args.num_class)]
        mIoU = sum(cIoU) / self.args.num_class
        print('Val mIoU: {}'.format(mIoU))
        for i in range(self.args.num_class):
            print(f'Val Class {i} IoU: {cIoU[i]}')

    def post_process(self, output, target, x_img, img_id, draw_results=False):
        # reconstruct original image
        x_img = x_img.squeeze(0).data.cpu().numpy()
        x_img = np.transpose(x_img, (1, 2, 0))
        x_img = x_img * np.array(self.image_std)
        x_img = x_img + np.array(self.image_mean)
        x_img = x_img * 255.0
        x_img = x_img.astype(np.uint8)

        if self.args.task == 'segmentation':
            output_prob = F.softmax(output[0], dim=0)

            output_argmax = torch.argmax(output, dim=1).cpu()
            self.metric.update(target.squeeze(1).cpu().detach().numpy(), output_argmax.numpy())

            if draw_results:
                output_grey = (output_prob.cpu().detach().numpy() * 255).astype(np.uint8)

                # draw heatmap
                output_heatmap_overlay = []
                for i in range(1, self.args.num_class):
                    output_grey_tmp = output_grey[i]
                    output_heatmap = utils.grey_to_heatmap(output_grey_tmp)
                    output_grey_tmp = np.repeat(output_grey_tmp[:, :, None] / 255, 3, 2)
                    output_heatmap_overlay.append((x_img * (1 - output_grey_tmp)) + (output_heatmap * output_grey_tmp))
                output_heatmap_overlay = np.array(output_heatmap_overlay).astype(np.uint8)

                if not os.path.exists(self.img_save_dir):
                    os.mkdir(self.img_save_dir)

                utils.cv2_imwrite(os.path.join(self.img_save_dir, img_id) + '.png', x_img)
                for i in range(1, self.args.num_class):
                    utils.cv2_imwrite(os.path.join(self.img_save_dir, img_id) + f'_zargmax_class_{i}.png', output_grey[i])

        if self.args.task == 'classification':
            output_argmax = torch.argmax(output, dim=1).cpu().numpy()

            self.metric.update(output_argmax.round(0), target.cpu().numpy().round(0))

            if draw_results:
                if not os.path.exists(self.img_save_dir):
                    os.mkdir(self.img_save_dir)

                utils.cv2_imwrite(os.path.join(self.img_save_dir, img_id) + '.png', x_img)
                cv2.putText(x_img, output_argmax[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        return

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

    def init_model(self, model_name):
        if model_name == 'Unet':
            model = model_implements.Unet(n_channels=self.args.input_channel, n_classes=self.args.num_class).to(self.device)
        elif model_name == 'Swin':
            model = model_implements.Swin(num_classes=self.args.num_class,
                                          in_channel=self.args.input_channel).to(self.device)
        elif model_name == 'ResNet18_multihead':
            model = model_implements.ResNet18_multihead(num_classes=self.args.num_class, sub_classes=1).to(self.device)
        else:
            raise Exception('No model named', model_name)

        return torch.nn.DataParallel(model)

    def init_metric(self, task_name, num_class):
        if task_name == 'segmentation':
            metric = metrics.StreamSegMetrics_segmentation(num_class)
        elif task_name == 'classification':
            metric = metrics.StreamSegMetrics_classification(num_class)
        else:
            raise Exception('No task named', task_name)

        return metric