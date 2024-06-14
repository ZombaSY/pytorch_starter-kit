conf=dict(
  env=dict(
    debug=True,
    CUDA_VISIBLE_DEVICES='0',
    mode='export',
    cuda=True,
    task='regression',
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

  export=dict(
    target='onnx',
    input_shape=[1, 3, 336, 336],
    opset_version=12
  )
)