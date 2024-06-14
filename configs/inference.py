conf=dict(
  env=dict(
    debug=True,
    CUDA_VISIBLE_DEVICES='0',
    mode='valid',
    cuda=True,
    task='regression',

    draw_results=False,
  ),

  model=dict(
    name='ConvNextV2_l_regression',
    num_class=1,
    in_channel=3,
    normalization='InstanceNorm1d',
    activation='ReLU',
    dropblock=True,
    saved_ckpt='/path/to/model.pt',
  ),

  dataloader_valid=dict(
    name='Image2Vector',
    mode='valid',
    data_path='/path/to/data.csv',
    data_cache=False,
    batch_size=32,
    input_size=[512, 512],  # (height, width)
    workers=8,
    weighted_sampler=False,
    label_cols=['col1', 'col2', 'col3'],
  ),

  criterion=dict(
    name='MSELoss'
  ),
)