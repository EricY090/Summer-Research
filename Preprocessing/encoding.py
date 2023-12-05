import re

from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))


def encode_code(single_admission_codes: dict, admission_codes: dict) -> tuple[dict, dict]:
    print('encoding diagnosis code ...')
    
    ### Map each ICD9 code to a number
    code_map = dict()
    for i, (admission_id, codes) in enumerate(admission_codes.items()):
        for code in codes:
            if code not in code_map:
                code_map[code] = len(code_map) + 1
                
    for j, (single_admission_id, single_codes) in enumerate(single_admission_codes.items()):
        for code in single_codes:
            if code not in code_map:
                code_map[code] = len(code_map) + 1
    

    ### {Admission: [mapped number for ICD9 codes]}
    admission_codes_encoded = {
        admission_id: [code_map[code] for code in codes]
        for admission_id, codes in admission_codes.items()
    }
    single_admission_codes_encoded = {
        admission_id: [code_map[code] for code in codes]
        for admission_id, codes in single_admission_codes.items()
    }
    return single_admission_codes_encoded, admission_codes_encoded, code_map


def encode_pcode(admission_pcodes: dict) -> tuple[dict, dict]:
    print('encoding procedure codes ...')
    
    ### Map each ICD9 code to a number
    pcode_map = dict()
    for i, (admission_id, codes) in enumerate(admission_pcodes.items()):
        for code in codes:
            if code not in pcode_map:
                pcode_map[code] = len(pcode_map) + 1

    ### {Admission: [mapped number for ICD9 codes]}
    admission_pcodes_encoded = {
        admission_id: [pcode_map[code] for code in codes]
        for admission_id, codes in admission_pcodes.items()
    }
    return admission_pcodes_encoded, pcode_map


def encode_lab(single_admission_items: dict, admission_items: dict) -> tuple[dict, dict]:
    print('encoding lab ...')
    
    ### Map each itemid to a number
    ### {Patient: [mapped number for itemid]}
    item_map = dict()
    single_admission_items_encoded = dict()
    admission_items_encoded = dict()
    for i, (admission, items) in enumerate(single_admission_items.items()):
        if admission not in single_admission_items_encoded:
            single_admission_items_encoded[admission] = []
        else: print("Impossible")
        for itemid, l in items.items():
            if itemid not in item_map:
                item_map[itemid] = len(item_map) + 1
            if l[1] == "abnormal":
                single_admission_items_encoded[admission].append(item_map[itemid])
    for i, (admission, items) in enumerate(admission_items.items()):
        if admission not in admission_items_encoded:
            admission_items_encoded[admission] = []
        else: print("Impossible")
        for item in items:
            if item[0] not in item_map:
                item_map[item[0]] = len(item_map) + 1
            if item[1] == "abnormal":
                admission_items_encoded[admission].append(item_map[item[0]])
    
    return single_admission_items_encoded, admission_items_encoded, item_map


def extract_word(text: str) -> list:
    """Extract words from a text
    @param: text, str
    @param: max_len, the maximum length of text we want to extract, default None
    @return: list, words list in the text
    """
    # replace non-word-character with space
    text = re.sub(r'[^A-Za-z_]', ' ', text.strip().lower())
    # tokenize text using NLTK
    words = word_tokenize(text)
    clean_words = []
    for word in words:
        if word not in stopwords_set:
            word = ps.stem(word).lower()
            if word not in stopwords_set:
                clean_words.append(word)
    return clean_words


def encode_note_train(patient_note: dict, pids: np.ndarray, max_note_len=None) -> tuple[dict, dict]:
    print('encoding train notes ...')
    dictionary = dict()
    patient_note_encoded = dict()
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        words = extract_word(patient_note[pid])
        note_encoded = []
        for word in words:
            if word not in dictionary:
                wid = len(dictionary) + 1
                dictionary[word] = wid
            else:
                wid = dictionary[word]
            note_encoded.append(wid)
        if max_note_len is not None:
            note_encoded = note_encoded[:max_note_len]
        patient_note_encoded[pid] = note_encoded
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return patient_note_encoded, dictionary


def encode_note_test(patient_note: dict, pids: np.ndarray, dictionary: dict, max_note_len=None) -> dict:
    print('encoding valid/test notes ...')
    patient_note_encoded = dict()
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        words = extract_word(patient_note[pid])
        note_encoded = []
        for word in words:
            if word in dictionary:
                note_encoded.append(dictionary[word])
        if len(note_encoded) == 0:
            note_encoded.append(0)
        if max_note_len is not None:
            note_encoded = note_encoded[:max_note_len]
        patient_note_encoded[pid] = note_encoded
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return patient_note_encoded
