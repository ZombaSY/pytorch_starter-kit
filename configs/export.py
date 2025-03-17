crop_size=128

conf=dict(
    env=dict(
        debug=True,
        CUDA_VISIBLE_DEVICES='0',
        mode='export',
        cuda=True,
        task='classification',
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

    export=dict(
        target='onnx',
        input_shape=[1, 3, crop_size, crop_size],
        opset_version=12
    )
)