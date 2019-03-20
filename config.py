config = {
    # raw input data for each TF family
    'publication_data': {
        'bHLH': 'raw_data/Combined_Max_Myc_Mad_Mad_r_log_normalized.xlsx',
        'ETS': 'raw_data/Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log_normalized.xlsx',
        'E2F': 'raw_data/Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log_normalized.xlsx',
        'RUNX': 'raw_data/Combined_Runx1_10nM_50nM_Runx2_10nM_50nM_log_normalized.xlsx',
    },

    # TF names for each family # TODO reps, different concentration
    'tf_names': {
        'bHLH': ['Myc', 'Mad', 'Max'],
        'ETS': ['Ets1_100nM', 'Elk1_100nM', 'Gabpa_100nM'],
        'E2F': ['E2f1_250nM', 'E2f3_250nM', 'E2f4_500nM'],
        'RUNX': ['Runx1_10nM', 'Runx2_10nM']
    },

    # training config
    'training': {
        'gpu_id': 0,

        'fully_connected': {
            'n_folds': 5,
            'epochs': 100,
            'batch_size': 500,
            'n_hid': 20,
        },

        'conv': {
            'n_folds': 5,
            'epochs': 100,
            'batch_size': 500,
            'filters': [
                # n_filter, filter_width, dilation
                (100, 8, 1),
                (50, 8, 2),
            ],
        },
    },

    'training_one_model': {
        'gpu_id': 0,

        'fully_connected': {
            'n_folds': 5,
            'epochs': 100,
            'batch_size': 500,
            'n_hid': 20,
        },

        'conv': {
            'n_folds': 5,
            'epochs': 200,  # increase epoch
            'batch_size': 500,
            'filters': [  # TODO can try increasing model complexity since we're sharing weights now
                # n_filter, filter_width, dilation
                (100, 8, 1),
                (50, 8, 2),
            ],
        },
    },

}
