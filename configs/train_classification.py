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
    project_name='wandb_sample',
    task='classification',

    train_fold=2,
    epoch=10000,
    early_stop_epoch=300,
  ),

  model=dict(
    name='Mobileone_s0_classification',
    num_class=1000,
    in_channel=3,
    normalization='BatchNorm1d',
    activation='ReLU',
    dropblock=True,
    use_ema=False,
    input_size=[crop_size, crop_size],
    saved_ckpt='pretrained/ImageNet/mobileone_s0_unfused.pth.tar',
  ),

  dataloader_train=dict(
    name='Image2Vector',
    mode='train',
    data_path='/path/to/train.csv',
    # data_path_folds=['/path/to/train-fold_0.csv',
    #                  '/path/to/train-fold_1.csv',
    #                  '/path/to/train-fold_2.csv',
    #                  '/path/to/train-fold_3.csv',
    #                  '/path/to/train-fold_4.csv'],
    label_cols=['col1', 'col2', 'col3'],
    data_cache=True,
    weighted_sampler=False,
    batch_size=32,
    input_size=input_size,
    workers=8,

    augmentations=dict(
      transform_blur=0.3,
      transform_clahe=0.1,
      transform_cutmix=0.0,
      transform_coarse_dropout=0.5,
      transform_fancyPCA=0.1,
      transform_fog=0.05,
      transform_g_noise=0.3,
      transform_jitter=0.1,
      transform_hflip=0.5,
      transform_vflip=0.0,
      transform_jpeg=0.3,
      transform_mixup=0.5,
      transform_perspective=0.1,
      transform_rand_resize=0.0,
      transform_rand_crop=crop_size,
      transform_rain=0.05,
      transform_rotate=0.3,
    )
  ),

  dataloader_valid=dict(
    name='Image2Vector',
    mode='valid',
    data_path='/path/to/valid.csv',
    # data_path_folds=['/path/to/valid-fold_0.csv',
    #                  '/path/to/valid-fold_1.csv',
    #                  '/path/to/valid-fold_2.csv',
    #                  '/path/to/valid-fold_3.csv',
    #                  '/path/to/valid-fold_4.csv'],
    label_cols=['col1', 'col2', 'col3'],
    data_cache=True,
    weighted_sampler=False,
    batch_size=32,
    input_size=input_size,
    workers=8,
  ),

  criterion=dict(
    name='LabelSmoothingCrossEntropyLoss'
  ),

  optimizer=dict(
    name='AdamW',
    lr=1e-3,
    lr_min=1e-5,
    weight_decay=5e-3,

    scheduler=dict(
      name='WarmupCosine',
      cycles=50,
      warmup_epoch=20),
  ),
)