### For multiple exmaples per patient with pcodes without notes, dim intercept and all examples for pre_trained 
import os
import _pickle as pickle

import numpy as np
from sklearn.utils import shuffle

from Preprocessing.parse import parse_admission, parse_diagnoses, parse_procedures, parse_lab
from Preprocessing.parse import calibrate_patient_by_admission, calibrate_patient_by_lab, calibrate_patient_by_procedures
from Preprocessing.encoding import encode_code, encode_pcode, encode_lab
from Preprocessing.build_dataset_2 import split_patients, code_matrix
from Preprocessing.build_dataset_2 import build_code_xy, build_single_lab_xy, build_heart_failure_y
from Preprocessing.auxiliary import generate_code_levels, generate_patient_code_adjacent, generate_code_code_adjacent, co_occur


if __name__ == '__main__':
    data_path = 'data'
    raw_path = os.path.join(data_path, 'mimic3', 'raw')
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        print('please put the CSV files in `data/mimic3/raw`')
        exit()
    single_patient_admission, patient_admission = parse_admission(raw_path)
    single_admission_codes, admission_codes = parse_diagnoses(raw_path, single_patient_admission, patient_admission)
    calibrate_patient_by_admission(patient_admission, admission_codes)
    calibrate_patient_by_admission(single_patient_admission, single_admission_codes)
    
    single_admission_items, admission_items = parse_lab(single_patient_admission, patient_admission)
    calibrate_patient_by_lab(single_patient_admission, single_admission_codes, single_admission_items)
    calibrate_patient_by_lab(patient_admission, admission_codes, admission_items)
    
    admission_pcodes = parse_procedures(raw_path, patient_admission)
    calibrate_patient_by_procedures(patient_admission, admission_codes, admission_items, admission_pcodes)
    
    print('There are %d valid patients with single admission' % len(single_patient_admission))   ### 26092
    print('There are %d valid patients with multiple admissions' % len(patient_admission))       ### 4077
    
    max_admission_num = 0           ### 14
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
    max_code_num_in_a_visit = 0
    for admission_id, codes in admission_codes.items():
        if len(codes) > max_code_num_in_a_visit:
            max_code_num_in_a_visit = len(codes)
    print(max_admission_num)
    
    ### Encoding
    single_admission_codes_encoded, admission_codes_encoded, code_map = encode_code(single_admission_codes, admission_codes)
    single_admission_items_encoded, admission_items_encoded, item_map = encode_lab(single_admission_items, admission_items)
    admission_pcodes_encoded, pcode_map = encode_pcode(admission_pcodes)
    
    code_num, pcode_num, item_num = len(code_map), len(pcode_map), len(item_map)     ### 6398, 1310, 682
    print(code_num, pcode_num, item_num)
    
    
    ### Split patients
    train_pids, valid_pids, test_pids = split_patients(
        patient_admission,
        admission_codes, code_map)
    

    
    
    ### Split codes into training, valid, testing
    #codes_matrix, labs_matrix, n_examples = code_matrix(np.array(list(patient_admission.keys())), patient_admission, admission_codes_encoded, 
    #                                       admission_pcodes_encoded, admission_items_encoded, max_admission_num, code_num, pcode_num, item_num)
    train_matrix, train_proc_matrix, train_lab_matrix, train_visit_lens, n_train = code_matrix(train_pids, patient_admission, admission_codes_encoded, admission_pcodes_encoded, admission_items_encoded, 
                                        max_admission_num, code_num, pcode_num, item_num)
    valid_matrix, valid_proc_matrix, valid_lab_matrix, valid_visit_lens, n_valid = code_matrix(valid_pids, patient_admission, admission_codes_encoded, admission_pcodes_encoded, admission_items_encoded, 
                                        max_admission_num, code_num, pcode_num, item_num)
    test_matrix, test_proc_matrix, test_lab_matrix, test_visit_lens, n_test = code_matrix(test_pids, patient_admission, admission_codes_encoded, admission_pcodes_encoded, admission_items_encoded, 
                                        max_admission_num, code_num, pcode_num, item_num)
    print(n_train, n_valid, n_test)        #5258+309+569 = 6136
    
    
    ### Visit lens broken 
    #code_x, code_y, code_lab_x = build_code_xy(codes_matrix, labs_matrix, n_examples, max_admission_num, code_num, pcode_num, item_num)
    train_codes_x, train_codes_y, train_proc_x, train_lab_x = build_code_xy(train_matrix, train_proc_matrix, train_lab_matrix, 
                                                                            n_train, max_admission_num, code_num, pcode_num, item_num)
    valid_codes_x, valid_codes_y, valid_proc_x, valid_lab_x = build_code_xy(valid_matrix, valid_proc_matrix, valid_lab_matrix,
                                                                            n_valid, max_admission_num, code_num, pcode_num, item_num)
    test_codes_x, test_codes_y, test_proc_x, test_lab_x = build_code_xy(test_matrix, test_proc_matrix, test_lab_matrix, 
                                                                        n_test, max_admission_num, code_num, pcode_num, item_num)
    
    
    ### Dataset for pre-trained Lab -> Diagnosis
    n_pre_trained_examples = len(single_admission_codes_encoded) + sum([len(admission) for p, admission in patient_admission.items()])
    print(n_pre_trained_examples)       ### 36305
    lab_x, lab_y = build_single_lab_xy(single_patient_admission, single_admission_items_encoded, single_admission_codes_encoded, 
                                       patient_admission, admission_codes_encoded, admission_items_encoded, 
                                       n_pre_trained_examples, code_num, item_num)
    
    lab_x, lab_y = shuffle(lab_x, lab_y)
    train_single_lab_x, train_single_lab_y = lab_x[:int(n_pre_trained_examples*0.85)], lab_y[:int(n_pre_trained_examples*0.85)]
    valid_single_lab_x, valid_single_lab_y = lab_x[int(n_pre_trained_examples*0.85):-int(n_pre_trained_examples*0.1)], lab_y[int(n_pre_trained_examples*0.85):-int(n_pre_trained_examples*0.1)]
    test_single_lab_x, test_single_lab_y = lab_x[-int(n_pre_trained_examples*0.1):], lab_y[-int(n_pre_trained_examples*0.1):]
    ######
    
    
    
    #code_x, code_y, code_lab_x = shuffle(code_x, code_y, code_lab_x)
    #train_codes_x, train_codes_y, train_lab_x = code_x[:int(n_examples*0.8)], code_y[:int(n_examples*0.8)], code_lab_x[:int(n_examples*0.8)]
    #valid_codes_x, valid_codes_y, valid_lab_x = code_x[int(n_examples*0.8):-int(n_examples*0.15)], code_y[int(n_examples*0.8):-int(n_examples*0.15)], code_lab_x[int(n_examples*0.8):-int(n_examples*0.15)]
    #test_codes_x, test_codes_y, test_lab_x = code_x[-int(n_examples*0.15):], code_y[-int(n_examples*0.15):], code_lab_x[-int(n_examples*0.15):]
    train_codes_x, train_codes_y, train_lab_x, train_proc_x, train_visit_lens = shuffle(train_codes_x, train_codes_y, train_lab_x, train_proc_x, train_visit_lens)
    valid_codes_x, valid_codes_y, valid_lab_x, valid_proc_x, valid_visit_lens = shuffle(valid_codes_x, valid_codes_y, valid_lab_x, valid_proc_x, valid_visit_lens)
    test_codes_x, test_codes_y, test_lab_x, test_proc_x, test_visit_lens = shuffle(test_codes_x, test_codes_y, test_lab_x, test_proc_x, test_visit_lens)
    print(train_codes_x.shape, train_codes_y.shape, train_lab_x.shape, train_proc_x.shape, train_visit_lens.shape)
    print(valid_codes_x.shape, valid_codes_y.shape, valid_lab_x.shape, valid_proc_x.shape, valid_visit_lens.shape)
    print(test_codes_x.shape, test_codes_y.shape, test_lab_x.shape, test_proc_x.shape, test_visit_lens.shape)
    
    
    train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
    valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
    test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)

    
    code_levels = generate_code_levels(data_path, code_map)
    patient_code_adj = generate_patient_code_adjacent(code_x=train_codes_x, code_num=code_num)
    code_code_adj_t = generate_code_code_adjacent(code_level_matrix=code_levels, code_num=code_num)
    co_occur_matrix = co_occur(train_pids, patient_admission, admission_codes_encoded, code_num)
    code_code_adj = code_code_adj_t * co_occur_matrix
    # patient_code_adj = patient_code_adj / np.sum(patient_code_adj, axis=-1, keepdims=True)
    # code_code_adj = code_code_adj / np.sum(code_code_adj, axis=-1, keepdims=True)
    # code_code_adj[np.isnan(code_code_adj)] = 0
    # code_code_adj[np.isinf(code_code_adj)] = 0
    

    train_codes_data = (train_codes_x, train_codes_y, train_lab_x, train_proc_x, train_visit_lens)
    valid_codes_data = (valid_codes_x, valid_codes_y, valid_lab_x, valid_proc_x, valid_visit_lens)
    test_codes_data = (test_codes_x, test_codes_y, test_lab_x, test_proc_x, test_visit_lens)
    train_labs_data = (train_single_lab_x, train_single_lab_y)
    valid_labs_data = (valid_single_lab_x, valid_single_lab_y)
    test_labs_data = (test_single_lab_x, test_single_lab_y)
    
    

    
    mimic3_path = os.path.join('data', 'mimic3')
    encoded_path = os.path.join(mimic3_path, 'encoded')
    if not os.path.exists(encoded_path):
        os.makedirs(encoded_path)
    
    
    print('saving encoded data ...')
    pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
    
    print('saving standard data ...')
    standard_path = os.path.join(mimic3_path, 'standard')
    if not os.path.exists(standard_path):
        os.makedirs(standard_path)
    
    pickle.dump({
        'train_codes_data': train_codes_data,
        'valid_codes_data': valid_codes_data,
        'test_codes_data': test_codes_data
    }, open(os.path.join(standard_path, 'codes_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_labs_data': train_labs_data,
        'valid_labs_data': valid_labs_data,
        'test_labs_data': test_labs_data
    }, open(os.path.join(standard_path, 'labs_dataset.pkl'), 'wb'))
    pickle.dump({
        'train_hf_y': train_hf_y,
        'valid_hf_y': valid_hf_y,
        'test_hf_y': test_hf_y
    }, open(os.path.join(standard_path, 'heart_failure.pkl'), 'wb'))
    pickle.dump({
        'code_levels': code_levels,
        'patient_code_adj': patient_code_adj,
        'code_code_adj': code_code_adj
    }, open(os.path.join(standard_path, 'auxiliary.pkl'), 'wb'))
    