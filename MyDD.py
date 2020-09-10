import DD
import helper as hp
import pandas as pd

g_model = None
g_original_method_name = None
g_predicted_method_name = None
g_all_data = []
g_cnt_dict = {}


def deltas_to_code(d):
    s = "".join([c[1] for c in d])
    # s = str(s.replace('"', '\\"'))
    return s


class MyDD(DD.DD):
    def __init__(self):
        DD.DD.__init__(self)

    def _test(self, deltas):
        if not deltas:
            return self.PASS

        src = deltas_to_code(deltas)
        if hp.is_parsable(src):
            predicted_method_name = hp.prediction_with_hc33(g_model, src)
            if predicted_method_name == g_predicted_method_name:
                g_all_data.append(src)
                return self.FAIL

        return self.PASS


if __name__ == '__main__':
    # TODO: modify here
    model_file = 'SVM_HC33.model'
    test_file = '/scratch/rabin/deployment/root-dd/data_selection/test.result'

    # load model
    g_model = hp.load_model_hc33(model_file)
    assert g_model is not None

    # read file
    df = pd.read_csv(test_file)
    file_list = df["path"].tolist()
    for idx, java_file in enumerate(file_list):
        print("\nStart: {}\n".format(java_file))
        g_all_data.clear()

        # method_name and method_body
        g_all_data.append("\npath = {}".format(java_file))
        method_name, method_body = hp.load_method(java_file)
        method_body = hp.single_line_body(method_body)
        assert (len(method_name) > 0) and (len(method_body) > 0)
        g_all_data.append("method_name = {}".format(method_name))
        g_all_data.append("method_body = {}".format(method_body))

        # set predicted method_name as global method_name
        g_original_method_name = method_name
        g_predicted_method_name = hp.prediction_with_hc33(g_model, method_body)
        g_all_data.append("predicted_method_name = {}".format(g_predicted_method_name))

        # create deltas by char
        data = list(method_body)  # ['a',...,'z']
        deltas = list(zip(range(len(data)), data))  # [('a',0), ..., ('z',n)]

        # run ddmin
        mydd = MyDD()
        print("Simplifying failure-inducing input...")
        g_all_data.append("\nTrace of simplified code(s):")
        c = mydd.ddmin(deltas)  # Invoke DDMIN
        print("The 1-minimal failure-inducing input is", c)
        print("Removing any element will make the failure go away.")
        javaCode = deltas_to_code(c)
        g_all_data.append("\nMinimal simplified code:\n{}".format(javaCode))

        # print/save all simplified code
        g_cnt_dict[method_name] = g_cnt_dict.get(method_name, 0) + 1
        output_file = "data/{}_{}.dd".format(method_name, g_cnt_dict[method_name])
        hp.save_simplified_code(g_all_data, output_file)

        print("\nDone: {}\n".format(java_file))
