import cv2
import torch
import time
import numpy as np
import os
import pandas as pd

from models import metrics
from models import utils
from models import dataloader as dataloader_hub
from models import model_implements

from torch.nn import functional as F
from PIL import Image


class Inferencer:

    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.loader_form = self.__init_data_loader(self.args.val_x_path,
                                                   self.args.val_y_path,
                                                   batch_size=1,
                                                   mode='validation')

        self.loader_val = self.loader_form.Loader

        self.model = self.__init_model(self.args.model_name)
        self.model.load_state_dict(torch.load(args.model_path))
        self.model.eval()

        dir_path, fn = os.path.split(self.args.model_path)
        fn, ext = os.path.splitext(fn)
        save_dir = dir_path + '/' + fn + '/'

        self.dir_path = dir_path
        self.model_fn = fn
        self.img_save_dir = save_dir
        self.num_batches_val = int(len(self.loader_val))
        self.metric = self._init_metric(self.args.inference_mode, self.args.num_class)

        self.image_mean = self.loader_form.image_loader.image_mean
        self.image_std = self.loader_form.image_loader.image_std
        self.fn_list = []

    def start_inference_classification(self):
        img_id_list = []
        level_list = []
        label_list = []

        for batch_idx, (img, target) in enumerate(self.loader_val):
            with torch.no_grad():
                x_in, img_id = img
                target, _ = target

                x_in = x_in.to(self.device)
                target = target[0].long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)
                output_argmax = torch.argmax(output[0], dim=0).cpu()

                output_argmax_np = output_argmax.numpy()
                target_np = target.cpu().detach().numpy()
                self.metric.update(target_np[None, :], output_argmax_np[None, :])

                path, fn = os.path.split(img_id[0])
                img_id, ext = os.path.splitext(fn)

                if batch_idx % 300 == 0:
                    print(f'{batch_idx} batch {img_id} \t Done !!')

                level_list.append(output_argmax_np)
                label_list.append(target_np)
                img_id_list.append(img_id)

        metrics = self.metric.get_results()
        mean_kappa_score = metrics['Mean Kappa Score']
        kappa_scores = metrics['Class Kappa Score']
        mean_acc_score = metrics['Mean Accuracy']
        acc_scores = metrics['Class Accuracy']

        print(f'Mean Kappa Score : {mean_kappa_score} \n'
              f'Mean Accuracy : {mean_acc_score}')

        for i in range(self.args.num_class):
            print(f'\t \t Val Class Kappa Score {i} : {kappa_scores[i]}')
            print(f'\t \t Val Class Accuracy {i} : {acc_scores[i]}')

        df = pd.DataFrame({'file_name': img_id_list,
                           'level': level_list,
                           'target': label_list,
                           })
        df.to_csv(self.dir_path + '/' + self.model_fn + '_score.csv', encoding='utf-8-sig', index=False)

    def start_inference_segmentation(self):
        img_id_list = []

        for batch_idx, (img, target) in enumerate(self.loader_val):
            with torch.no_grad():
                x_in, img_id = img
                target, _ = target

                x_in = x_in.to(self.device)
                target = target.long().to(self.device)  # (shape: (batch_size, img_h, img_w))

                output = self.model(x_in)

                self.post_process(output, target, x_in, img_id, batch_idx, draw_results=False)

                img_id_list.append(img_id)

        metrics = self.metric.get_results()
        cIoU = [metrics['Class IoU'][i] for i in range(self.args.num_class)]
        mIoU = sum(cIoU) / self.args.num_class

        print('Val mIoU: {}'.format(mIoU))
        for i in range(self.args.num_class):
            print(f'Val Class {i} IoU: {cIoU[i]}')

        df = pd.DataFrame({'file_name': img_id_list,
                           # 'level': level_list,
                           # 'target': label_list,
                           })
        df.to_csv(self.dir_path + '/' + self.model_fn + '_score.csv', encoding='utf-8-sig', index=False)

    def post_process(self, output, target, x_img, img_id, batch_idx, draw_results=False):

        if self.args.criterion == 'CE':
            output_argmax = torch.argmax(output, dim=1).cpu()
            self.metric.update(target.squeeze(1).cpu().detach().numpy(), output_argmax.numpy())

            if draw_results:
                # reconstruct original image
                x_img = x_img.squeeze(0).data.cpu().numpy()
                x_img = np.transpose(x_img, (1, 2, 0))
                x_img = x_img * np.array(self.image_std)
                x_img = x_img + np.array(self.image_mean)
                x_img = x_img * 255.0
                x_img = x_img.astype(np.uint8)

                output_prob = F.softmax(output[0], dim=0)
                # output_prob = F.sigmoid(output[0, 1, :, :])
                output_grey = (output_prob.cpu().detach().numpy() * 255).astype(np.uint8)

                # draw heatmap
                output_heatmap_overlay = []
                for i in range(1, self.args.num_class):
                    output_grey_tmp = output_grey[i]
                    output_heatmap = utils.grey_to_heatmap(output_grey_tmp)
                    output_grey_tmp = np.repeat(output_grey_tmp[:, :, None] / 255, 3, 2)
                    output_heatmap_overlay.append((x_img * (1 - output_grey_tmp)) + (output_heatmap * output_grey_tmp))
                output_heatmap_overlay = np.array(output_heatmap_overlay)

                path, fn = os.path.split(img_id[0])
                img_id, ext = os.path.splitext(fn)
                dir_path, fn = os.path.split(self.args.model_path)
                fn, ext = os.path.splitext(fn)
                save_dir = dir_path + '/' + fn + '/'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                Image.fromarray(x_img).save(save_dir + img_id + '.png', quality=100)
                for i in range(1, self.args.num_class):
                    Image.fromarray(output_grey[i]).save(save_dir + img_id + f'_zargmax_class_{i}.png', quality=100)
                    Image.fromarray(output_heatmap_overlay[i - 1].astype(np.uint8)).save(save_dir + img_id + f'_heatmap_overlay_class_{i}.png', quality=100)

        metric_result = {}
        # metric_result = utils.metrics_np(output_argmax[None, :], target.squeeze(0).detach().cpu().numpy(), b_auc=True)
        # keys = [metric_result.keys()]

        print(f'batch {batch_idx} -> {img_id} \t Done !!')
        return metric_result

    def __init_model(self, model_name):
        if model_name == 'Unet':
            model = model_implements.Unet(n_channels=self.args.input_channel, n_classes=self.args.num_class).to(
                self.device)
        elif model_name == 'Swin':
            model = model_implements.Swin(num_classes=self.args.num_class,
                                          in_channel=self.args.input_channel).to(self.device)

        else:
            raise Exception('No model named', model_name)

        return torch.nn.DataParallel(model)

    def __init_data_loader(self,
                           x_path,
                           y_path,
                           batch_size,
                           mode):

        if self.args.dataloader == 'Image2Image':
            loader = dataloader_hub.Image2ImageDataLoader(x_path=x_path,
                                                          y_path=y_path,
                                                          batch_size=batch_size,
                                                          num_workers=self.args.worker,
                                                          pin_memory=self.args.pin_memory,
                                                          mode=mode,
                                                          args=self.args)
        else:
            raise Exception('No dataloader named', self.args.dataloader)

        return loader

    def _init_metric(self, mode, num_class):
        if mode == 'segmentation':
            metric = metrics.StreamSegMetrics_segmentation(num_class)
        elif mode == 'classification':
            metric = metrics.StreamSegMetrics_classification(num_class)
        else:
            raise Exception('No mode named', mode)

        return metric
