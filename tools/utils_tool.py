import torch
import math

from models import dataloader
from models import lr_scheduler
from models import model_implements
from models import losses as loss_hub
from models import utils
from models import metrics


def init_data_loader(conf, conf_dataloader):
    loader = getattr(dataloader, conf_dataloader['name'])(conf, conf_dataloader)

    return loader


def set_scheduler(conf, conf_dataloader, optimizer, data_loader):
    scheduler = None
    steps_per_epoch = math.ceil((data_loader.__len__() / conf_dataloader['batch_size']))

    if hasattr(conf, 'scheduler'):
        if conf['scheduler']['name'] == 'WarmupCosine':
            scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=optimizer,
                                                            warmup_steps=steps_per_epoch * conf['scheduler']['warmup_epoch'],
                                                            t_total=conf['env']['epoch'] * steps_per_epoch,
                                                            cycles=conf['env']['epoch'] / conf['scheduler']['cycles'],
                                                            last_epoch=-1)
        elif conf['scheduler']['name'] == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf['scheduler']['cycles'], eta_min=conf['scheduler']['lr_min'])
        elif conf['scheduler']['name'] == 'Constant':
            scheduler = lr_scheduler.ConstantLRSchedule(optimizer, last_epoch=-1)
        elif conf['scheduler']['name'] == 'WarmupConstant':
            scheduler = lr_scheduler.WarmupConstantSchedule(optimizer, warmup_steps=steps_per_epoch * conf['scheduler']['warmup_epoch'])
        else:
            utils.Logger().info(f"{utils.Colors.LIGHT_PURPLE}No scheduler found --> {conf['scheduler']['name']}{utils.Colors.END}")
    else:
        pass

    return scheduler


def init_model(conf, device):
    model = getattr(model_implements, conf['model']['name'])(conf['model']).to(device)

    return torch.nn.DataParallel(model)


def init_criterion(conf_criterion, device):
    criterion = getattr(loss_hub, conf_criterion['name'])(conf_criterion).to(device)

    return criterion


def init_optimizer(conf_optimizer, model):
    optimizer = None

    if conf_optimizer['name'] == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=conf_optimizer['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=conf_optimizer['weight_decay'])
    elif conf_optimizer['name'] == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=conf_optimizer['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=conf_optimizer['weight_decay'])
    return optimizer


def init_metric(task_name, num_class):
    if 'segmentation' in task_name:
        metric = metrics.StreamSegMetrics_segmentation(num_class)
    elif task_name in ['classification', 'regression', 'landmark', 'regression-ssl', 'landmark-ssl']:
        metric = metrics.StreamSegMetrics_classification(num_class)
    else:
        raise Exception('No task named', task_name)

    return metric


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
