from config import Config
from dd_tensorflow_model import Code2VecModel
import sm_helper as hp

###############################################################

g_model = None
g_all_data = []
g_cnt_dict = {}

###############################################################

if __name__ == '__main__':
    config = Config(set_defaults=True, load_from_args=True, verify=True)
    g_model = Code2VecModel(config)
    print('Done creating code2vec model')
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
            predict, _, _ = hp.prediction_with_c2v(g_model, hp.g_root_path, hp.g_simp_file)
            assert method_name == predict

            # get path-context and attention
            topk_attn, topk_path = hp.get_attention(g_model, hp.g_root_path, hp.g_simp_file)
            topk_terminal = []
            g_all_data.append("\ntopk path-contexts:")
            for i in range(len(topk_path)):
                path_context = topk_path[i].strip().split(',')
                topk_terminal.append([path_context[0], path_context[-1]])
                g_all_data.append("[{}] {}".format(f"{topk_attn[i]:.4f}", topk_path[i]))
            g_all_data.append("\ntopk terminals:\n{}".format(topk_terminal))

            # print/save path-context and attention
            save_name = "L{}_{}_{}.txt".format(str(idx + 1), method_name, g_cnt_dict[method_name])
            output_file = "attn_data/{}".format(save_name)
            hp.save_simplified_code(g_all_data, output_file)
            print("\nDone [{}]: {}\n".format(idx + 1, java_file))
        except:
            print("\nError [{}]: {}\n".format(idx + 1, java_file))

    g_model.close_session()
