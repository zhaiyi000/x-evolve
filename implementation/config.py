import os


if 'funsearch_config_type' not in os.environ:
    raise Exception('need config type')


config_type = os.environ['funsearch_config_type']
if config_type not in ['bin_packing', 'cap_set']:
    raise Exception('wrong type')


if config_type == 'bin_packing':
    evaluate_function_c_l1 = 0.1
    evaluate_function_c_1 = 100
    evaluate_function_temperature = 0.1
    evaluate_function_mask_half = False

    sample_iterator_temperature = 1
    sample_iterator_no_update_cnt = 3
elif config_type == 'cap_set':
    evaluate_function_c_l1 = 0.
    evaluate_function_c_1 = 1
    evaluate_function_temperature = 10
    evaluate_function_mask_half = True

    sample_iterator_temperature = 100
    sample_iterator_no_update_cnt = 10
else:
    raise Exception('wrong type')
