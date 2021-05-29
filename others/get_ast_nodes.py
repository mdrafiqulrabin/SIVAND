import subprocess
from pathlib import Path

JAR_JAVA_AST_DATA = "GetAstNodes/target/jar/GetAstNodes.jar"
INNER_DELIMITER = " ___INNER___ "
OUTER_DELIMITER = " ___OUTER___ "


def get_ast_data(str_code):
    cmd = ['java', '-jar', JAR_JAVA_AST_DATA, str_code]
    content = subprocess.check_output(cmd, encoding="utf-8", close_fds=True)
    [all_terminals, all_classes] = content.strip().split(OUTER_DELIMITER)
    all_terminals = all_terminals.split(INNER_DELIMITER)
    all_classes = all_classes.split(INNER_DELIMITER)
    return all_terminals, all_classes


if __name__ == '__main__':
    program = Path('sample_input.java').read_text()
    print("program:\n{}".format(program))
    ast_terminals, ast_classes = get_ast_data(program)
    print("ast_terminals = {}".format(ast_terminals))
    print("ast_classes = {}".format(ast_classes))
    ast_nodes = list(set(ast_terminals + ast_classes))
    print("all_nodes = {}".format(ast_nodes))
