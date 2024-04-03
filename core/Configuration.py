class Config:

    # Hampel
    HAMPEL_WINDOW_SIZE = 3
    HAMPEL_GAUSSIAN_DISTR_SCALE_FACTOR = 1.4826
    HAMPEL_N_SIGMAS = 1

    # Butterworth low pass filter
    BUTTERWORTH_LOW_PASS_FILTER_ORDER = 2
    BUTTERWORTH_LOW_PASS_FILTER_FS = 100.0
    BUTTERWORTH_LOW_PASS_FILTER_CUTOFF = 15

    # Threshold for the probability of joint detected. Default: 75
    THRESHOLD_PROBABILITY_OF_JOINT_DETECTED = 75.0

    @staticmethod
    def get_dataset_diving_analysis_parameters(dataset):
        dataset_diving_analysis_parameters_dict = {}

        if dataset and dataset.dataset_diving_analysis_parameters:

            # Hampel
            dataset_diving_analysis_parameters_dict['hampel_window_size'] = \
                dataset.dataset_diving_analysis_parameters.hampel_window_size
            dataset_diving_analysis_parameters_dict['hampel_gaussian_distr_scale_factor'] = \
                dataset.dataset_diving_analysis_parameters.hampel_gaussian_distr_scale_factor
            dataset_diving_analysis_parameters_dict['hampel_n_sigmas'] = \
                dataset.dataset_diving_analysis_parameters.hampel_n_sigmas

            # Butterworth low pass filter
            dataset_diving_analysis_parameters_dict['butterworth_low_pass_filter_order'] = \
                dataset.dataset_diving_analysis_parameters.butterworth_low_pass_filter_order
            dataset_diving_analysis_parameters_dict['butterworth_low_pass_filter_fs'] = \
                dataset.dataset_diving_analysis_parameters.butterworth_low_pass_filter_fs
            dataset_diving_analysis_parameters_dict['butterworth_low_pass_filter_cutoff'] = \
                dataset.dataset_diving_analysis_parameters.butterworth_low_pass_filter_cutoff

            # Threshold for the probability of joint detected
            dataset_diving_analysis_parameters_dict['threshold_probability_of_joint_detected'] = \
                dataset.dataset_diving_analysis_parameters.threshold_probability_of_joint_detected
        else:
            # Hampel
            dataset_diving_analysis_parameters_dict['hampel_window_size'] = \
                Config.HAMPEL_WINDOW_SIZE
            dataset_diving_analysis_parameters_dict['hampel_gaussian_distr_scale_factor'] = \
                Config.HAMPEL_GAUSSIAN_DISTR_SCALE_FACTOR
            dataset_diving_analysis_parameters_dict['hampel_n_sigmas'] = \
                Config.HAMPEL_N_SIGMAS

            # Butterworth low pass filter
            dataset_diving_analysis_parameters_dict['butterworth_low_pass_filter_order'] = \
                Config.BUTTERWORTH_LOW_PASS_FILTER_ORDER
            dataset_diving_analysis_parameters_dict['butterworth_low_pass_filter_fs'] = \
                Config.BUTTERWORTH_LOW_PASS_FILTER_FS
            dataset_diving_analysis_parameters_dict['butterworth_low_pass_filter_cutoff'] = \
                Config.BUTTERWORTH_LOW_PASS_FILTER_CUTOFF

            # Threshold for the probability of joint detected
            dataset_diving_analysis_parameters_dict['threshold_probability_of_joint_detected'] = \
                Config.THRESHOLD_PROBABILITY_OF_JOINT_DETECTED

        return dataset_diving_analysis_parameters_dict
