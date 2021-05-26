from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from config import Config
from dd_model import Model
import sm_helper as hp

###############################################################

g_model = None
g_all_data = []
g_cnt_dict = {}

###############################################################

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)
    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--release', action='store_true',
                        help='release the loaded model for a smaller model size.')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=239)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)

    g_model = Model(config)
    print('Created model')

    assert g_model is not None

    # read file
    file_list = hp.get_file_list()
    for idx, java_file in enumerate(file_list):
        print("\nStart [{}]: {}\n".format(idx + 1, java_file))
        g_all_data.clear()

        try:
            # method_name and method_body
            g_all_data.append("\npath = {}".format(java_file))
            method_name, method_body = hp.load_method(java_file)
            assert (len(method_name) > 0) and (len(method_body) > 0)
            g_cnt_dict[method_name] = g_cnt_dict.get(method_name, 0) + 1
            g_all_data.append("method_name = {}".format(method_name))
            g_all_data.append("method_body = {}".format(method_body))
            hp.store_method(hp.g_simp_file, method_body)

            # check if prediction is correct
            predict, _, _ = hp.prediction_with_c2s(g_model, hp.g_root_path, hp.g_simp_file)
            assert method_name == predict

            # get path-context and attention
            attn_tokens, topk_attns, topk_paths = hp.get_attention(g_model, method_name, hp.g_root_path, hp.g_simp_file)
            for i in range(len(topk_paths)):
                g_all_data.append("\ntopk path-contexts for subtoken-{}: {}".format(i + 1, attn_tokens[i]))
                topk_terminal = []
                for j in range(len(topk_paths[i])):
                    short_path_context = topk_paths[i][j].strip().split(',')
                    full_path_context = hp.get_full_path_context(short_path_context)
                    topk_terminal.append([short_path_context[0], short_path_context[-1]])
                    g_all_data.append("[{}] {}".format(f"{topk_attns[i][j]:.4f}", full_path_context))
                g_all_data.append(
                    "\ntopk terminals for subtoken-{}: {}\n{}".format(i + 1, attn_tokens[i], topk_terminal))

            # print/save path-context and attention
            save_name = "L{}_{}_{}.txt".format(str(idx + 1), method_name, g_cnt_dict[method_name])
            output_file = "attn_data/{}".format(save_name)
            hp.save_simplified_code(g_all_data, output_file)
            print("\nDone [{}]: {}\n".format(idx + 1, java_file))
        except:
            print("\nError [{}]: {}\n".format(idx + 1, java_file))

    g_model.close_session()
