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
    name='Mobileone_s0_landmark',
    num_class=212,
    in_channel=3,
    # base_c=16,
    normalization='InstanceNorm1d',
    activation='ReLU',
    dropblock=True,
  ),

  dataloader_train=dict(
    name='Image2Landmark',
    mode='train',
    data_path='/path/to/train.csv',
    data_cache=False,
    weighted_sampler=False,
    batch_size=32,
    input_size=[336, 336],  # (height, width)
    workers=8,

    augmentations=dict(
      transform_blur=0.5,
      transform_clahe=0.5,
      transform_cutmix=0.0,
      transform_coarse_dropout=0.5,
      transform_fancyPCA=0.5,
      transform_fog=0.2,
      transform_g_noise=0.5,
      transform_jitter=0.5,
      transform_hflip=0.0,
      transform_vflip=0.0,
      transform_jpeg=0.5,
      transform_perspective=0.0,
      transform_rand_resize=0.0,
      transform_rand_crop=336,
      transform_rain=0.2,
      transform_rotate=0.3,
      transform_landmark_rotate=0.5,
      transform_landmark_hflip=0.5,
    )
  ),

  dataloader_valid=dict(
    name='Image2Landmark',
    mode='valid',
    data_path='/path/to/valid.csv',
    data_cache=False,
    weighted_sampler=False,
    batch_size=32,
    input_size=[336, 336],  # (height, width)
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