import ast
import subprocess
import _pickle as pk

# For handcrafted
HC33_TARGETS = ["equals", "main", "setUp", "onCreate", "toString", "run", "hashCode", "init", "execute", "get", "close"]
LABELS2INDEX = {m: i for i, m in enumerate(HC33_TARGETS)}
INDEX2LABELS = {i: m for i, m in enumerate(HC33_TARGETS)}


def load_method(java_file):
    try:
        cmd = ['java', '-jar', 'models/LoadJavaMethod.jar', java_file]
        # subprocess.call(cmd)
        content = subprocess.check_output(cmd, encoding="utf-8")
        content = ' '.join(str(content).split())
        content = ast.literal_eval(str(content))
        return content[0], content[1]
    except:
        return "", ""


def load_model_hc33(mod):
    try:
        with open("models/HC33/" + mod, 'rb') as f:
            clf = pk.load(f)
            return clf
    except:
        return None


def prediction_with_hc33(g_model, src):
    if len(src.strip()) < 1:
        return ""
    try:
        cmd = ['java', '-jar', 'models/HC33/Features_HC33.jar', src]
        y_test = subprocess.check_output(cmd, encoding="utf-8")  # 0,...,1
        y_test = [int(y) for y in y_test.split(',')]  # [0,...,1]
        assert sum(y_test) > 0
        y_pred = g_model.predict([y_test])
        y_pred = INDEX2LABELS[y_pred[0]]
        return y_pred
    except:
        return ""


def save_simplified_code(all_methods, output_file):
    open(output_file, 'w').close()
    with open(output_file, 'a') as f:
        for jCode in all_methods:
            print(jCode)
            f.write(jCode + "\n")
