import DD
import helper as hp

g_model = None
g_original_method_name = None
g_predicted_method_name = None
g_all_methods = []


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
        predicted_method_name = hp.prediction_with_hc33(g_model, src)
        if predicted_method_name == g_predicted_method_name:
            g_all_methods.append(src)
            return self.FAIL
        return self.PASS


if __name__ == '__main__':
    # TODO: modify here
    model_file = 'SVM_HC33.model'
    java_file = 'data/test.java'

    # load model
    g_model = hp.load_model_hc33(model_file)
    assert g_model is not None

    # read file
    method_name, method_body = hp.load_method(java_file)
    method_body = str(method_body.replace('"', '\"').replace('\r', '').replace('\n', ''))
    assert len(method_name.strip()) > 0

    # set global method_name
    g_original_method_name = method_name
    g_predicted_method_name = hp.prediction_with_hc33(g_model, method_body)
    g_all_methods.append(method_body)

    # create deltas
    data = list(method_body)  # ['a',...,'z']
    deltas = list(zip(range(len(data)), data))  # [('a',0), ..., ('z',n)]

    # run ddmin
    mydd = MyDD()
    print("Simplifying failure-inducing input...")
    c = mydd.ddmin(deltas)  # Invoke DDMIN
    print("The 1-minimal failure-inducing input is", c)
    print("Removing any element will make the failure go away.")
    javaCode = deltas_to_code(c)
    g_all_methods.append(javaCode)

    # save all simplified code
    output_file = java_file.replace(".java", ".txt")
    hp.save_simplified_code(g_all_methods, output_file)
