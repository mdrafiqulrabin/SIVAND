import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

/**
 * Input: <file_path>.java
 * Output: "method_name" "single_line"
 */
public class Main {

    public static void main(String[] args) {
        File inputFile = new File(args[0]);
        CompilationUnit root = getParseUnit(inputFile);
        List<String> contents = loadJavaMethod(root);
        System.out.print(String.join(" ", contents));
    }

    static List<String> loadJavaMethod(CompilationUnit root) {
        List<String> contents = new ArrayList<>();
        if (root != null) {
            try {
                MethodDeclaration methodDec = root.findAll(MethodDeclaration.class).get(0);
                String method_name = methodDec.getName().toString();
                contents.add(method_name);
                String single_line = methodDec.toString()
                        .replaceAll("\\n", " ")
                        .replaceAll("\\r", " ");
                contents.add(single_line);
            } catch (Exception ignore){}
        }
        return contents;
    }

    static CompilationUnit getParseUnit(File javaFile) {
        CompilationUnit root = null;
        try {
            String txtCode = new String(Files.readAllBytes(javaFile.toPath()));
            // add tmp class to parse
            if(!txtCode.startsWith("class")) txtCode = "class T { \n" + txtCode + "\n}";
            // remove comments
            StaticJavaParser.getConfiguration().setAttributeComments(false);
            // parse code
            root = StaticJavaParser.parse(txtCode);
        } catch (Exception ignore) {}
        return root;
    }

}