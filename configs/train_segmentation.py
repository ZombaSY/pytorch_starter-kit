conf=dict(
  env=dict(
    debug=True,
    CUDA_VISIBLE_DEVICES='0',
    mode='train',
    cuda=True,
    wandb=True,
    saved_model_directory='model_ckpt',
    project_name='wandb_project',
    task='segmentation',

    train_fold=2,
    epoch=10000,
    early_stop_epoch=300,
  ),

  model=dict(
    name='Swin_T_semanticSegmentation',
    num_class=20,
    in_channel=3,
    saved_ckpt='',
  ),

  dataloader_train=dict(
    name='Image2Image',
    mode='train',
    data_path='/path/to/train.csv',
    data_cache=True,
    batch_size=32,
    input_size=[512, 512],  # (height, width)
    workers=8,

    augmentations=dict(
      transform_blur=0.3,
      transform_clahe=0.1,
      transform_cutmix=0.8,
      transform_coarse_dropout=0.2,
      transform_fancyPCA=0.1,
      transform_fog=0.01,
      transform_g_noise=0.1,
      transform_jitter=0.01,
      transform_hflip=0.5,
      transform_vflip=0.0,
      transform_jpeg=0.5,
      transform_perspective=0.1,
      transform_rand_resize=0.7,
      transform_rand_crop=256,
      transform_rain=0.01,
    )
  ),

  dataloader_valid=dict(
    name='Image2Image',
    mode='valid',
    data_path='/path/to/valid.csv',
    data_cache=True,
    batch_size=32,
    input_size=[512, 512],  # (height, width)
    workers=8,
  ),

  criterion=dict(
    name='CrossEntropyLoss'
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