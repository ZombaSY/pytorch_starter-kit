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
        name='Mobileone_s0_regression',
        num_class=1000,
        in_channel=3,
        normalization='InstanceNorm1d',
        activation='ReLU',
        dropblock=True,
        use_ema=False,
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