import DD
import helper as hp

###############################################################

g_model = None
g_original_method_name = None
g_predicted_method_name = None
g_all_data = []
g_cnt_dict = {}
g_cnt_pass = [0, 0, 0]

###############################################################


class MyDD(DD.DD):
    def __init__(self):
        DD.DD.__init__(self)

    def _test(self, _deltas):
        if not _deltas:
            return self.PASS

        try:
            g_cnt_pass[0] = g_cnt_pass[0] + 1
            _code = hp.deltas_to_code(_deltas)
            hp.store_method(hp.g_simp_file, _code)
            if hp.is_parsable(_code):
                g_cnt_pass[1] = g_cnt_pass[1] + 1
                _predict, _score, _loss = hp.prediction_with_M(g_model, hp.g_simp_file)
                _time = hp.get_current_time()
                print('time = {}, predict = {}, score = {}, loss = {}'.format(_time, _predict, _score, _loss))
                if _predict == g_predicted_method_name:
                    g_cnt_pass[2] = g_cnt_pass[2] + 1
                    _data = hp.get_json_data(_time, _score, _loss, _code, _deltas[:], g_cnt_pass)
                    g_all_data.append("{}".format(_data))
                    return self.FAIL
        except Exception:
            pass

        return self.PASS


if __name__ == '__main__':
    g_model = hp.load_model_M()
    assert g_model is not None

    # do for each file
    file_list = hp.get_file_list()
    for idx, file_path in enumerate(file_list):
        print("\nStart [{}]: {}\n".format(idx + 1, file_path))
        g_all_data.clear()
        g_cnt_pass = [0, 0, 0]

        try:
            # get method_name and method_body
            g_all_data.append("\npath = {}".format(file_path))
            method_name, method_body = hp.load_method(file_path)
            assert (len(method_name) > 0) and (len(method_body) > 0)
            g_cnt_dict[method_name] = g_cnt_dict.get(method_name, 0) + 1
            g_all_data.append("method_name = {}".format(method_name))
            g_all_data.append("method_body = {}".format(method_body))
            hp.store_method(hp.g_simp_file, method_body)

            # check predicted method_name
            g_original_method_name = method_name
            predict, score, loss = hp.prediction_with_M(g_model, hp.g_simp_file)
            g_predicted_method_name = predict
            g_all_data.append("predict, score, loss = {}, {}, {}".format(predict, score, loss))
            assert g_original_method_name == g_predicted_method_name

            # create deltas by char/token
            deltas = []
            if hp.g_deltas_type == "token":
                deltas = hp.get_token_deltas(method_body)
            else:
                deltas = hp.get_char_deltas(method_body)

            # run ddmin to simplify program
            try:
                mydd = MyDD()
                print("Simplifying prediction-preserving input...")
                g_all_data.append("\nTrace of simplified code(s):")
                c = mydd.ddmin(deltas)  # Invoke DDMIN
                print("The 1-minimal prediction-preserving input is", c)
                print("Removing any element will make the prediction go away.")
                program = hp.deltas_to_code(c)
                g_all_data.append("\nMinimal simplified code:\n{}".format(program))
            except Exception as e:
                g_all_data.append("\nException:\n{}".format(str(e)))

            # save all simplified traces
            save_name = "L{}_{}_{}.txt".format(str(idx + 1), method_name, g_cnt_dict[method_name])
            output_file = "dd_data/{}".format(save_name)
            hp.save_simplified_code(g_all_data, output_file)
            print("\nDone [{}]: {}\n".format(idx + 1, file_path))
        except:
            print("\nError [{}]: {}\n".format(idx + 1, file_path))

    g_model.close_session()
