import sys

sys.path.append('.')
sys.path.insert(0, "..")

import yaml
import tensorflow as tf
from checkpoint_tracker import Tracker
from data import data_loader, vocabulary
from meta_model import VarMisuseModel
import sm_helper as hp

###############################################################

g_model = None
g_data = None
g_all_data = []
g_cnt_dict = {}

###############################################################


def evaluate_single_data(data, model):
    losses, accs, preds, attns = [], [], [], []  # get_metrics()
    for batch in data.batcher(mode='eval'):
        tokens, edges, error_loc, repair_targets, repair_candidates = batch
        token_mask = tf.clip_by_value(tf.reduce_sum(tokens, -1), 0, 1)
        pointer_preds, attns = model(tokens, token_mask, edges, training=False)
        batch_loss, batch_acc, batch_pred = model.get_loss(pointer_preds, token_mask, error_loc, repair_targets,
                                                           repair_candidates)
        losses, accs, preds = batch_loss, batch_acc, batch_pred
        break  # single sample only
    accs = [a.numpy().tolist() for a in accs]
    losses = [l.numpy().tolist() for l in losses]
    preds = [p.numpy().tolist() for p in preds]
    attns = [a.numpy().tolist() for a in attns]
    return [attns, preds, losses, accs]


###############################################################


if __name__ == '__main__':
    config = yaml.safe_load(open(hp.config_path))
    print("Configuration:", config)

    hp.set_model_json_from_configuration(config["model"]["configuration"])
    if hp.vm_model_path is None:
        raise ValueError("Must provide a path to pre-trained models when running final evaluation")

    data = data_loader.DataLoader(hp.data_path, config["data"], vocabulary.Vocabulary(hp.vocabulary_path))
    model = VarMisuseModel(config['model'], data.vocabulary.vocab_dim)
    model.run_dummy_input()
    tracker = Tracker(model, hp.vm_model_path)
    tracker.restore_single_ckpt()

    # g_data/g_model for DD
    g_data, g_model = data, model

    # apply_dd_eval(g_data, g_model)
    js_count, dd_count = 0, 0
    with open(hp.vm_json_path) as file:
        for line in file:
            if not line.strip(): continue
            js_count += 1
            try:
                print("\nStart: {}\n".format(js_count))
                g_all_data.clear()
                g_cnt_pass = [0, 0, 0]

                sample = hp.check_eval_tmp(line.strip())
                assert sample["results"]["model"] == config["model"]["configuration"]
                results = evaluate_single_data(data, model)

                if hp.g_buggy_type == hp.g_buggy_types[0]:  # buggy
                    assert results[-1][0] == 0 and results[-1][-1] == 1.0
                else:  # nonbuggy
                    assert results[-1][0] == 1.0 and results[-1][1] == 0 and results[-1][2] == 0

                # original sample
                g_all_data.append("\nOriginal sample:\n")
                g_all_data.append(hp.get_pretty_sample(sample.copy()))

                # save filename
                save_name = "{}___L{}.txt".format(sample["txt_file"], sample["js_count"])

                # attentions of tokens
                sample_tokens = sample["source_tokens"]
                g_all_data.append("\n\nAll source tokens:\n")
                g_all_data.append(str(sample_tokens))
                sample_attns = results[0][0]
                g_all_data.append("\n\nAll attention probs:\n")
                g_all_data.append(str(sample_attns))

                # top-k index
                attn_idx = sorted(range(len(sample_attns)), key=lambda i: sample_attns[i], reverse=True)[:10]
                topk_tokens = [sample_tokens[i] for i in attn_idx]
                g_all_data.append("\n\nTop-k source tokens:\n")
                g_all_data.append(str(topk_tokens))
                topk_attns = [sample_attns[i] for i in attn_idx]
                g_all_data.append("\n\nTop-k attention probs:\n")
                g_all_data.append(str(topk_attns))

                # print/save all simplified code
                dd_count += 1
                output_file = hp.dd_file.format(save_name)
                hp.save_simplified_code(g_all_data, output_file)
                print("\nDone: {}-{}\n".format(js_count, dd_count))

                if dd_count >= hp.g_dd_count:
                    quit()
            except Exception as e:
                print("\nError: {}\n{}".format(js_count, str(e)))
