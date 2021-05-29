import subprocess
from pathlib import Path

JAR_LOAD_JAVA_METHOD = "LoadJavaMethod/target/jar/LoadJavaMethod.jar"


def load_method(file_path):
    cmd = ['java', '-jar', JAR_LOAD_JAVA_METHOD, file_path]
    contents = subprocess.check_output(cmd, encoding="utf-8", close_fds=True)
    contents = contents.split()
    name, body = contents[0], " ".join(contents[1:])
    return name, body


if __name__ == '__main__':
    input_path = 'sample_input.java'
    program = Path(input_path).read_text()
    print("program:\n{}".format(program))
    method_name, single_line = load_method(input_path)
    print("method_name = {}".format(method_name))
    print("single_line = {}".format(single_line))
