import json
import re
import statistics
import subprocess
from datetime import datetime

import javalang
import pandas as pd

###############################################################
# TODO: all (if)

g_root_path = "/scratch/rabin/deployment/root-simplify/sm-code2seq"

g_c2s_model = "/scratch/rabin/models/code2seq/main/java-large/saved_model_iter52"

g_files_root = "/scratch/rabin/deployment/root-simplify/data_selection"
g_test_files = {
    "correct_samefile": g_files_root + "/mn_c2x/c2x_jl_test_correct_prediction_samefile.txt"
}

g_test_loc = "correct_samefile"
g_test_file = g_test_files[g_test_loc]

g_c2s_test = g_root_path + "/data/sm/sm.test.c2s"
g_db_name = "sm"

JAR_LOAD_JAVA_METHOD = g_files_root + "/LoadJavaMethod.jar"
JAR_C2S_JAVA_EXTRACTOR = g_root_path + "/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar"

###############################################################
# TODO: DD (if)

g_deltas_types = ["token", "char"]
g_deltas_type = g_deltas_types[0]

g_simp_file = "data/tmp/sm_test.java"

###############################################################
# TODO: attn (if)

g_topk_attn = 200  # topk attentions

###############################################################


def get_file_list():
    file_list = []
    try:
        df = pd.read_csv(g_test_file)
        file_list = df["path"].tolist()[:1000]
    except:
        pass
    return file_list


def get_subtoken_name(name):
    name = '|'.join(re.sub(r"([A-Z])", r" \1", name).split())
    return name.lower()


def get_flat_name(name):
    name = name.split('|')
    if len(name) < 2:
        return ''.join(name)
    else:
        return name[0] + ''.join([x.capitalize() for x in name[1:]])


def fix_internal_delimiter(name):  # a|bb|ccc
    name = name.split('|')
    name = [name[0]] + [n.title() for n in name[1:]]
    return "".join(name)  # aBbCcc


def get_current_time():
    return str(datetime.now())


def is_parsable(src, original_method_name, java_file):
    try:
        tree = javalang.parse.parse("class Test { " + src + " }")
        assert tree is not None
        method_name, _ = load_method(java_file)
        assert method_name == original_method_name
    except:
        return False
    return True


def load_method(java_file):
    try:
        cmd = ['java', '-jar', JAR_LOAD_JAVA_METHOD, java_file]
        contents = subprocess.check_output(cmd, encoding="utf-8", close_fds=True)
        contents = contents.split()
        method_name = contents[0]
        method_body = " ".join(contents[1:])
        return method_name, method_body
    except:
        return "", ""


def store_method(dd_file, method_body):
    with open(dd_file, "w") as f:
        f.write(method_body + "\n")


def prediction_with_c2s(model, root_path, java_file):
    try:
        # preprocess data
        open(g_c2s_test, 'w').close()
        cmd = ['/bin/sh', g_root_path + '/preprocess_test.sh',
               root_path, java_file, g_db_name]  # source --> /bin/sh
        subprocess.call(cmd, close_fds=True)
        c2s_after = open(g_c2s_test, 'r').read()
        assert len(c2s_after.strip()) > 0
        # evaluate model
        result = model.evaluate()
        assert result is not None and len(result) > 0
        [predict, score, loss] = result
        return get_flat_name(predict), statistics.mean(score), float(loss)
    except:
        return ""


def save_simplified_code(all_methods, output_file):
    open(output_file, 'w').close()
    with open(output_file, 'a') as f:
        for jCode in all_methods:
            print(jCode)
            f.write(jCode + "\n")
        f.write("\n")


def get_json_data(time, score, loss, code, tokens=None, n_pass=None):
    score, loss = str(round(float(score), 4)), str(round(float(loss), 4))
    data = {'time': time, 'score': score, 'loss': loss, 'code': code}
    if tokens:
        data['n_tokens'] = len(tokens)
    if n_pass:
        data['n_pass'] = n_pass
    j_data = json.dumps(data)
    return j_data


###############################################################


def get_char_deltas(program):
    data = list(program)  # ['a',...,'z']
    deltas = list(zip(range(len(data)), data))  # [('a',0), ..., ('z',n)]
    return deltas


def get_token_deltas(program):
    token, tokens = "", []
    for c in program:
        if not c.isalpha():
            tokens.append(token)
            tokens.append(c)
            token = ""
        else:
            token = token + c
    tokens.append(token)
    tokens = [token for token in tokens if len(token) != 0]
    deltas = list(zip(range(len(tokens)), tokens))
    return deltas


def get_substr_deltas(program, n):
    substrs = [program[i: i + n] for i in range(0, len(program), n)]
    deltas = list(zip(range(len(substrs)), substrs))
    return deltas


###############################################################


c2s_ast_map = {'ArAc': 'ArrayAccessExpr', 'ArBr': 'ArrayBracketPair', 'ArCr': 'ArrayCreationExpr',
               'ArCrLvl': 'ArrayCreationLevel', 'ArIn': 'ArrayInitializerExpr', 'ArTy': 'ArrayType',
               'Asrt': 'AssertStmt', 'AsAn': 'AssignExpr:and', 'As': 'AssignExpr:assign', 'AsLS': 'AssignExpr:lShift',
               'AsMi': 'AssignExpr:minus', 'AsOr': 'AssignExpr:or', 'AsP': 'AssignExpr:plus', 'AsRe': 'AssignExpr:rem',
               'AsRSS': 'AssignExpr:rSignedShift', 'AsRUS': 'AssignExpr:rUnsignedShift', 'AsSl': 'AssignExpr:slash',
               'AsSt': 'AssignExpr:star', 'AsX': 'AssignExpr:xor', 'And': 'BinaryExpr:and',
               'BinAnd': 'BinaryExpr:binAnd', 'BinOr': 'BinaryExpr:binOr', 'Div': 'BinaryExpr:divide',
               'Eq': 'BinaryExpr:equals', 'Gt': 'BinaryExpr:greater', 'Geq': 'BinaryExpr:greaterEquals',
               'Ls': 'BinaryExpr:less', 'Leq': 'BinaryExpr:lessEquals', 'LS': 'BinaryExpr:lShift',
               'Minus': 'BinaryExpr:minus', 'Neq': 'BinaryExpr:notEquals', 'Or': 'BinaryExpr:or',
               'Plus': 'BinaryExpr:plus', 'Mod': 'BinaryExpr:remainder', 'RSS': 'BinaryExpr:rSignedShift',
               'RUS': 'BinaryExpr:rUnsignedShift', 'Mul': 'BinaryExpr:times', 'Xor': 'BinaryExpr:xor',
               'Bk': 'BlockStmt', 'BoolEx': 'BooleanLiteralExpr', 'Cast': 'CastExpr', 'Catch': 'CatchClause',
               'CharEx': 'CharLiteralExpr', 'ClsEx': 'ClassExpr', 'ClsD': 'ClassOrInterfaceDeclaration',
               'Cls': 'ClassOrInterfaceType', 'Cond': 'ConditionalExpr', 'Ctor': 'ConstructorDeclaration',
               'Do': 'DoStmt', 'Dbl': 'DoubleLiteralExpr', 'Emp': 'EmptyMemberDeclaration', 'Enc': 'EnclosedExpr',
               'ExpCtor': 'ExplicitConstructorInvocationStmt', 'Ex': 'ExpressionStmt', 'Fld': 'FieldAccessExpr',
               'FldDec': 'FieldDeclaration', 'Foreach': 'ForeachStmt', 'For': 'ForStmt', 'If': 'IfStmt',
               'Init': 'InitializerDeclaration', 'InstanceOf': 'InstanceOfExpr', 'IntEx': 'IntegerLiteralExpr',
               'IntMinEx': 'IntegerLiteralMinValueExpr', 'Labeled': 'LabeledStmt', 'Lambda': 'LambdaExpr',
               'LongEx': 'LongLiteralExpr', 'MarkerExpr': 'MarkerAnnotationExpr', 'Mvp': 'MemberValuePair',
               'Cal': 'MethodCallExpr', 'Mth': 'MethodDeclaration', 'MethRef': 'MethodReferenceExpr', 'Nm': 'NameExpr',
               'NormEx': 'NormalAnnotationExpr', 'Null': 'NullLiteralExpr', 'ObjEx': 'ObjectCreationExpr',
               'Prm': 'Parameter', 'Prim': 'PrimitiveType', 'Qua': 'QualifiedNameExpr', 'Ret': 'ReturnStmt',
               'SMEx': 'SingleMemberAnnotationExpr', 'StrEx': 'StringLiteralExpr', 'SupEx': 'SuperExpr',
               'SwiEnt': 'SwitchEntryStmt', 'Switch': 'SwitchStmt', 'Sync': 'SynchronizedStmt', 'This': 'ThisExpr',
               'Thro': 'ThrowStmt', 'Try': 'TryStmt', 'TypeDec': 'TypeDeclarationStmt', 'Type': 'TypeExpr',
               'TypePar': 'TypeParameter', 'Inverse': 'UnaryExpr:inverse', 'Neg': 'UnaryExpr:negative',
               'Not': 'UnaryExpr:not', 'PosDec': 'UnaryExpr:posDecrement', 'PosInc': 'UnaryExpr:posIncrement',
               'Pos': 'UnaryExpr:positive', 'PreDec': 'UnaryExpr:preDecrement', 'PreInc': 'UnaryExpr:preIncrement',
               'Unio': 'UnionType', 'VDE': 'VariableDeclarationExpr', 'VD': 'VariableDeclarator',
               'VDID': 'VariableDeclaratorId', 'Void': 'VoidType', 'While': 'WhileStmt', 'Wild': 'WildcardType'}


def get_full_path_context(short_path_context):
    full_ast_path = ''
    for short_ast_node in short_path_context[1].split('|'):
        node_parts = re.split(r'(\d+)', short_ast_node)
        node_parts = [n for n in node_parts]
        node_parts[0] = c2s_ast_map[node_parts[0]]
        full_ast_path += ''.join(node_parts) + '|'
    return "{},{},{}".format(short_path_context[0], full_ast_path[:-1], short_path_context[-1])


def get_attention(model, method_name, root_path, java_file):
    try:
        # preprocess data
        open(g_c2s_test, 'w').close()
        cmd = ['/bin/sh', 'preprocess_test.sh',
               root_path, java_file, g_db_name]  # source --> /bin/sh
        subprocess.call(cmd, close_fds=True)
        c2s_after = open(g_c2s_test, 'r').read()
        assert len(c2s_after.strip()) > 0

        # set topk for attention
        ast_paths = c2s_after.strip().split()[1:]
        l_topk_attn = len(ast_paths) if len(ast_paths) < g_topk_attn else g_topk_attn

        # get topk path-context
        attentions = model.get_attention()
        assert attentions is not None
        attn_tokens = get_subtoken_name(method_name).split('|')
        topk_attns, topk_paths = [], []
        for attn_idx in range(len(attn_tokens)):
            sub_attn = attentions[attn_idx][:l_topk_attn]
            topk_idx = sorted(range(len(sub_attn)), key=sub_attn.__getitem__, reverse=True)[:l_topk_attn]
            topk_path = [ast_paths[i] for i in topk_idx]
            topk_paths.append(topk_path)
            topk_attn = [sub_attn[i] for i in topk_idx]
            topk_attns.append(topk_attn)
        return attn_tokens, topk_attns, topk_paths
    except:
        return ""
