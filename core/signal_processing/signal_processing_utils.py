from core.Configuration import Config
from core.signal_processing.butterworth_low_pass_filter import get_data_with_butterworth_low_pass_filter
from core.signal_processing.outliers_removal import hampel_filter_forloop


def get_enhanced_data_list(video, data):
    dataset_diving_analysis_parameters_dict = Config.get_dataset_diving_analysis_parameters(video.dataset)

    # Hampel algorithm (outliers removal)
    data_without_outliers, detected_outliers_indices = hampel_filter_forloop(
        data,
        window_size=dataset_diving_analysis_parameters_dict['hampel_window_size'],
        k=dataset_diving_analysis_parameters_dict['hampel_gaussian_distr_scale_factor'],
        n_sigmas=dataset_diving_analysis_parameters_dict['hampel_n_sigmas']
    )

    # Butterworth low-pass filter
    data_without_outliers_filtered = get_data_with_butterworth_low_pass_filter(
        data_without_outliers,
        order=dataset_diving_analysis_parameters_dict['butterworth_low_pass_filter_order'],
        fs=dataset_diving_analysis_parameters_dict['butterworth_low_pass_filter_fs'],
        cutoff=dataset_diving_analysis_parameters_dict['butterworth_low_pass_filter_cutoff']
    )

    return data_without_outliers_filtered
