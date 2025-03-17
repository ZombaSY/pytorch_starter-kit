input_size=[128, 128]   # (height, width)
crop_size=128

conf=dict(
    env=dict(
        debug=True,
        CUDA_VISIBLE_DEVICES='0',
        mode='train',
        cuda=True,
        wandb=True,
        saved_model_directory='model_ckpt',
        project_name='wandb_project',
        task='landmark',
        train_fold=1,
        epoch=10000,
        early_stop_epoch=300,
    ),

    model=dict(
        name='Mobileone_s0_landmark',
        num_class=48,
        in_channel=3,
        normalization='BatchNorm1d',
        activation='ReLU',
        dropblock=True,
        use_ema=False,
        input_size=[crop_size, crop_size],
        saved_ckpt='pretrained/ImageNet/mobileone_s0_unfused.pth.tar',
    ),

    dataloader_train=dict(
        name='Image2Landmark',
        mode='train',
        data_path='/path/to/train.csv',
        # data_path_folds=['/path/to/train-fold_0.csv',
        #                  '/path/to/train-fold_1.csv',
        #                  '/path/to/train-fold_2.csv',
        #                  '/path/to/train-fold_3.csv',
        #                  '/path/to/train-fold_4.csv'],
        data_cache=True,
        weighted_sampler=False,
        batch_size=64,
        input_size=input_size,
        workers=16,

        augmentations=dict(
            transform_blur=0.5,
            transform_clahe=0.5,
            transform_cutmix=0.0,
            transform_coarse_dropout=0.0, # error on keypoint augmentation!
            transform_fancyPCA=0.5,
            transform_fog=0.3,
            transform_g_noise=0.5,
            transform_jitter=0.5,
            transform_hflip=0.0,  # use `transform_landmark_hflip` instead.
            transform_vflip=0.0,
            transform_jpeg=0.5,
            transform_perspective=0.2,
            transform_rand_resize=0.0,
            transform_rand_crop=crop_size,
            transform_resize=input_size,
            transform_rain=0.3,
            transform_rotate=0.2,
            transform_landmark_hflip=0.5,
        )
    ),

    dataloader_valid=dict(
        name='Image2Landmark',
        mode='valid',
        data_path='/path/to/valid.csv',
        # data_path_folds=['/path/to/valid-fold_0.csv',
        #                  '/path/to/valid-fold_1.csv',
        #                  '/path/to/valid-fold_2.csv',
        #                  '/path/to/valid-fold_3.csv',
        #                  '/path/to/valid-fold_4.csv'],
        data_cache=True,
        weighted_sampler=False,
        batch_size=64,
        input_size=input_size,
        workers=16,
    ),

    criterion=dict(
        name='WingLoss',
        omega=0.1,
        epsilon=2
    ),

    optimizer=dict(
        name='AdamW',
        lr=5e-3,
        lr_min=5e-5,
        weight_decay=5e-3,
    ),

    scheduler=dict(
        name='WarmupCosine',
        cycles=20,  # unit: epoch
        warmup_epoch=10
    ),
)