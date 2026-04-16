import pandas as pd


def generate_window_table_names(months_list, window_sizes, default_start):
    """Generate window table metadata for each window size and end month.

    Returns a dict keyed by window_size with list of specs.
    """
    specs_by_size = {}
    for window_size in window_sizes:
        size_specs = []
        for end_month in months_list:
            start_month = end_month - pd.offsets.DateOffset(months=window_size - 1)
            if start_month < default_start:
                continue

            start_ym = start_month.strftime('%y%m')
            end_ym = end_month.strftime('%y%m')
            size_specs.append(
                {
                    'table_name': f"data_window.cus_feature_{window_size}m_{start_ym}_{end_ym}",
                    'short_name': f"cus_feature_{window_size}m_{start_ym}_{end_ym}",
                    'start_date': start_month.strftime('%Y-%m-01'),
                    'end_date': (end_month + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d'),
                    'start_ym': start_ym,
                    'end_ym': end_ym,
                    'window_size': window_size,
                }
            )
        specs_by_size[window_size] = size_specs
    return specs_by_size
