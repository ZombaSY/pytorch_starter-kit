conf=dict(
  env=dict(
    debug=True,
    CUDA_VISIBLE_DEVICES='0',
    mode='train',
    cuda=True,
    wandb=True,
    saved_model_directory='model_ckpt',
    project_name='wandb_project',
    task='regression',

    train_fold=2,
    epoch=10000,
    early_stop_epoch=300,
  ),

  model=dict(
    name='Mobileone_s0_regression',
    num_class=1,
    in_channel=3,
    # base_c=16,
    normalization='InstanceNorm1d',
    activation='ReLU',
    dropblock=True,
  ),

  dataloader_train=dict(
    name='Image2Vector',
    mode='train',
    data_path='/path/to/train.csv',
    label_cols=['col1', 'col2', 'col3'],
    data_cache=True,
    weighted_sampler=False,
    batch_size=32,
    input_size=[448, 448],  # (height, width)
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
      transform_rand_crop=448,
      transform_rain=0.2,
      transform_rotate=0.3,
    )
  ),

  dataloader_valid=dict(
    name='Image2Vector',
    mode='valid',
    data_path='/path/to/train.csv',
    label_cols=['col1', 'col2', 'col3'],
    data_cache=True,
    weighted_sampler=False,
    batch_size=32,
    input_size=[448, 448],  # (height, width)
    workers=8,
  ),

  criterion=dict(
    name='MSELoss'
  ),

  optimizer=dict(
    name='AdamW',
    lr=0.001,
    lr_min=0.0001,
    weight_decay=0.005,

    scheduler=dict(
      name='WarmupCosine',
      cycles=50,
      warmup_epoch=20),
  ),
)