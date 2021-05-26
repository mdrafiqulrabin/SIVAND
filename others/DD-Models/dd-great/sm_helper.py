import os
import json
from datetime import datetime

#######################################

root_path = "/scratch/rabin/deployment/root-simplify/sm-great/"
vocabulary_path = root_path + "vocab.txt"
config_path = root_path + "config.yml"

vm_json_root_path = "/scratch/rabin/deployment/root-simplify/data_selection/vm_rnn_transformer/"
vm_json_paths = {
    "rnn_buggy": vm_json_root_path + "buggy_correct_prediction_rnn_samefile.txt",
    "rnn_nonbuggy": vm_json_root_path + "nonbuggy_correct_prediction_rnn_samefile.txt",
    "transformer_buggy": vm_json_root_path + "buggy_correct_prediction_transformer_samefile.txt",
    "transformer_nonbuggy": vm_json_root_path + "nonbuggy_correct_prediction_transformer_samefile.txt"
}
vm_json_path = None

vm_model_paths = {
    "rnn": "/scratch/rabin/models/great/vm/rnn/checkpoints/",
    "transformer": "/scratch/rabin/models/great/vm/transformer/checkpoints/"
}
vm_model_path = None

g_buggy_types = ["buggy", "nonbuggy"]

data_path = root_path + "sm_data/tmp/great/"
eval_tmp = data_path + "eval/tmp.txt"

dd_file = root_path + "sm_data/dd_data/{}"

# <modify here>
g_buggy_type = g_buggy_types[0]
g_dd_count = 1000

#######################################

sample_keys = ["has_bug", "source_tokens", "error_location", "repair_targets", "repair_candidates"]
marker_keys = ["error_location", "repair_targets", "repair_candidates"]

g_sample = {}
g_marker = {}

#######################################


def set_model_json_from_configuration(m_type):
    global vm_json_path, vm_model_path
    vm_json_path = vm_json_paths["{}_{}".format(m_type, g_buggy_type)]
    vm_model_path = vm_model_paths[m_type]


def get_current_time():
    return str(datetime.now())


def get_eval_txt_files():
    txt_files = []
    try:
        vm_data_path = "/scratch/rabin/data/vm/great/"
        eval_path = vm_data_path + "eval/"
        txt_files = [eval_path + f for f in os.listdir(eval_path) if '.txt' in f]
    except:
        pass
    return txt_files


def get_eval_tmp_json():
    try:
        with open(eval_tmp, 'r') as f:
            for l in f:
                if l.strip():
                    return json.loads(l)
    except:
        return ''


def set_eval_tmp_json(sample):
    try:
        with open(eval_tmp, 'w') as f:
            json.dump(sample, f)
    except:
        pass


def check_eval_tmp(line, sample=None):
    try:
        if not sample:
            sample = json.loads(line)
        if g_buggy_type == g_buggy_types[0]:  # buggy
            assert len(sample["repair_targets"]) > 0
        else:
            assert len(sample["repair_targets"]) == 0
        sample["repair_candidates"] = [t for t in sample["repair_candidates"] if isinstance(t, int)]
        js_org = sample.copy()
        set_eval_tmp_json(js_org.copy())
        js_dup = get_eval_tmp_json()
        if len(js_dup) > 0 and js_dup == js_org:
            global g_sample, g_marker
            g_sample, g_marker = get_marker_tokens(js_dup.copy())
            return g_sample
    except:
        pass
    return None


def get_marker_tokens(sample):
    sample["marker_tokens"] = list(sample["source_tokens"])
    sample["error_location"] = [sample["error_location"]]
    marker_tokens = {k: [] for k in marker_keys}
    for k in marker_keys:
        for i, p in enumerate(sample[k]):
            t = "<{}_{}>".format(k, i) + sample["marker_tokens"][p] + "<\\{}_{}>".format(k, i)
            marker_tokens[k].append(t)
            sample["marker_tokens"][p] = t
    return sample, marker_tokens


def update_marker_positions(deltas):
    sample = g_sample.copy()
    sample["source_tokens"] = list(deltas)
    sample["marker_tokens"] = list(deltas)
    for k in marker_keys[::-1]:
        sample[k] = []
        for i, t in enumerate(g_marker[k]):
            if t in deltas:
                p = deltas.index(t)
                sample[k].append(p)
                t = t.replace("<{}_{}>".format(k, i), '').replace("<\\{}_{}>".format(k, i), '')
                sample["source_tokens"][p] = t
                deltas[p] = t
            else:
                return ""
    sample["error_location"] = sample["error_location"][0]
    return sample


def get_pretty_sample(sample):
    sample.pop("edges")
    sample.pop("marker_tokens")
    # sample = json.dumps(sample, indent=4, separators=(',', ': '))
    sample = json.dumps(sample)
    return sample


def filter_keys(sample):
    all_keys = list(sample.keys())
    for k in all_keys:
        if k not in sample_keys:
            sample.pop(k)
    return sample


def get_dd_json_data(time, n_pass, result, sample):
    pred, loss, accuracy = result[0], result[1], result[2]
    loc_pred, rep_pred, tar_pred = pred[0][0], pred[1][0], pred[2][0]
    sample = filter_keys(sample.copy())
    error_location_pred = [p for i, p in enumerate(loc_pred) if i in [sample["error_location"]]]
    repair_targets_pred = [p for i, p in enumerate(rep_pred) if i in sample["repair_targets"]]
    repair_candidates_pred = [p for i, p in enumerate(rep_pred) if i in sample["repair_candidates"]]
    j_data = []
    data1 = {"result": {
        "time": time, "n_pass": n_pass, "n_token": len(sample["source_tokens"]),
        "loss": loss, "accuracy": accuracy
    }}
    data2 = {"sample": {
        "has_bug": sample["has_bug"],
        "source_tokens": sample["source_tokens"]
    }}
    data3 = {"position": {
        "error_location": sample["error_location"],
        "repair_targets": sample["repair_targets"],
        "repair_candidates": sample["repair_candidates"]
    }}
    data4 = {"prediction": {
        "error_location": error_location_pred[0],
        "repair_targets": repair_targets_pred,
        "repair_candidates": repair_candidates_pred,
        "target_probs": tar_pred
    }}
    for j in [data1, data2, data3, data4]:
        j_data += [json.dumps(j)]
    return j_data


def save_simplified_code(all_methods, output_file):
    open(output_file, 'w').close()
    with open(output_file, 'a') as f:
        for jCode in all_methods:
            print(jCode)
            f.write(jCode)
            f.write("\n")
