from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from config import Config
from dd_model import Model
import DD
import sm_helper as hp

###############################################################

g_model = None
g_original_method_name = None
g_predicted_method_name = None
g_cnt_pass = [0, 0, 0]
g_all_data = []
g_cnt_dict = {}

###############################################################


def deltas_to_code(d):
    s = "".join([c[1] for c in d])
    return s


class MyDD(DD.DD):
    def __init__(self):
        DD.DD.__init__(self)

    def _test(self, deltas):
        if not deltas:
            return self.PASS

        try:
            g_cnt_pass[0] = g_cnt_pass[0] + 1
            code = deltas_to_code(deltas[:])
            hp.store_method(hp.g_simp_file, code)
            if hp.is_parsable(code, g_original_method_name, hp.g_simp_file):
                g_cnt_pass[1] = g_cnt_pass[1] + 1
                predict, score, loss = hp.prediction_with_c2s(g_model, hp.g_root_path, hp.g_simp_file)
                time = hp.get_current_time()
                print('time = {}, predict = {}, score = {}, loss = {}'.format(time, predict, score, loss))
                if predict == g_predicted_method_name:
                    g_cnt_pass[2] = g_cnt_pass[2] + 1
                    j_data = hp.get_json_data(time, score, loss, code, deltas[:], g_cnt_pass)
                    g_all_data.append("{}".format(j_data))
                    return self.FAIL
        except:
            pass

        return self.PASS


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
        g_cnt_pass = [0, 0, 0]

        try:
            # method_name and method_body
            g_all_data.append("\npath = {}".format(java_file))
            method_name, method_body = hp.load_method(java_file)
            assert (len(method_name) > 0) and (len(method_body) > 0)
            g_cnt_dict[method_name] = g_cnt_dict.get(method_name, 0) + 1
            g_all_data.append("method_name = {}".format(method_name))
            g_all_data.append("method_body = {}".format(method_body))
            hp.store_method(hp.g_simp_file, method_body)

            # set predicted method_name as global method_name
            g_original_method_name = method_name
            predict, score, loss = hp.prediction_with_c2s(g_model, hp.g_root_path, hp.g_simp_file)
            g_predicted_method_name = predict
            g_all_data.append("predict, score, loss = {}, {}, {}".format(predict, score, loss))

            assert g_original_method_name == g_predicted_method_name

            # create deltas by char/token
            deltas = []
            if hp.g_deltas_type == "token":
                deltas = hp.get_token_deltas(method_body)
            else:
                deltas = hp.get_char_deltas(method_body)

            try:
                # run ddmin
                mydd = MyDD()
                print("Simplifying failure-inducing input...")
                g_all_data.append("\nTrace of simplified code(s):")
                c = mydd.ddmin(deltas)  # Invoke DDMIN
                print("The 1-minimal failure-inducing input is", c)
                print("Removing any element will make the failure go away.")
                javaCode = deltas_to_code(c)
                g_all_data.append("\nMinimal simplified code:\n{}".format(javaCode))
            except Exception as e:
                g_all_data.append("\nException:\n{}".format(str(e)))

            # print/save all simplified code
            save_name = "L{}_{}_{}.txt".format(str(idx + 1), method_name, g_cnt_dict[method_name])
            output_file = "dd_data/{}".format(save_name)
            hp.save_simplified_code(g_all_data, output_file)
            print("\nDone [{}]: {}\n".format(idx + 1, java_file))
        except:
            print("\nError [{}]: {}\n".format(idx + 1, java_file))

    g_model.close_session()
