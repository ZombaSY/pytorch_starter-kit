input_size=[448, 448]   # (height, width)
crop_size=448
data_cache=True

conf=dict(
    env=dict(
        debug=False,
        CUDA_VISIBLE_DEVICES='3',
        mode='train',
        cuda=True,
        wandb=True,
        saved_model_directory='model_ckpt',
        project_name='wandb_project',
        task='regression',
        train_fold=1,
        epoch=10000,
        early_stop_epoch=300,
    ),

    model=dict(
        name='SegmentatorRegressor',
        num_class=1,
        in_channel=3,
        base_c=16,
        normalization='InstanceNorm1d',
        activation='ReLU',
        dropblock=True,
        use_ema=False,
        saved_ckpt='',

        # segmentation model
        model=dict(
            name='UNet_light_dsv',
            num_class=2,
            in_channel=3,
            base_c=16,
            saved_ckpt='pretrained/UNet_light_dsv-folds_0-mIoU_0.8425-Epoch_215.pt',
        ),
    ),

    dataloader_train=dict(
        name='Image2Vector',
        mode='train',
        data_path_folds=['/path/to/train1.csv',
                         '/path/to/train2.csv',
                         '/path/to/train3.csv'],
        label_cols=['score-pseudo'],
        data_cache=data_cache,
        weighted_sampler=False,
        batch_size=128,
        input_size=input_size,
        workers=8,

        augmentations=dict(
            transform_blur=0.5,
            transform_clahe=0.5,
            transform_cutmix=0.0,
            transform_mixup=0.5,
            transform_coarse_dropout=0.5,
            transform_fancyPCA=0.5,
            transform_fog=0.2,
            transform_g_noise=0.5,
            transform_jitter=0.5,
            transform_hflip=0.0,
            transform_vflip=0.0,
            transform_jpeg=0.5,
            transform_perspective=0.1,
            transform_rand_resize=0.0,
            transform_rand_crop=crop_size,
            transform_resize=input_size,
            transform_rotate=0.0,
            transform_rain=0.2,
        )
    ),
    dataloader_valid=dict(
        name='Image2Vector',
        mode='valid',
        data_path_folds=['/path/to/valid1.csv',
                         '/path/to/valid2.csv',
                         '/path/to/valid3.csv'],
        label_cols=['score-pseudo'],
        data_cache=data_cache,
        weighted_sampler=False,
        batch_size=128,
        input_size=input_size,
        workers=8,
    ),

    criterion=dict(
        name='MSELoss'
    ),
    optimizer=dict(
        name='AdamW',
        lr=0.00001,
        lr_min=0.000001,
        weight_decay=0.005,

        scheduler=dict(
            name='WarmupCosine',
            cycles=20,  # unit: epoch
            warmup_epoch=10
        ),
    ),
)