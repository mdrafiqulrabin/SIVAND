import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

/**
 * Input: .../file.java
 * Output: "method_name" "method_body"
 */
public class Main {

    public static void main(String[] args) {
        File inputFile = new File(args[0]);
        //System.out.println(inputFile);
        CompilationUnit root = getParseUnit(inputFile);
        List<String> contents = loadJavaMethod(root);
        System.out.print(String.join(" ", contents));
    }

    static List<String> loadJavaMethod(CompilationUnit root) {
        List<String> contents = new ArrayList<String>();
        if (root != null) {
            try {
                MethodDeclaration methodDec = root.findAll(MethodDeclaration.class).get(0);
                String method_name = methodDec.getName().toString();
                contents.add(method_name);
//                methodDec.setName("f"); // keep original name
                String method_body = methodDec.toString()
                        .replaceAll("\\n", " ")
                        .replaceAll("\\r", " ");
                contents.add(method_body);
            } catch (Exception ignore){}
        }
        return contents;
    }

    static CompilationUnit getParseUnit(File javaFile) {
        CompilationUnit root = null;
        try {
            String txtCode = new String(Files.readAllBytes(javaFile.toPath()));
            if(!txtCode.startsWith("class")) txtCode = "class T { \n" + txtCode + "\n}"; // add tmp class to parse
            StaticJavaParser.getConfiguration().setAttributeComments(false); // remove comments
            root = StaticJavaParser.parse(txtCode);
        } catch (Exception ignore) {}
        return root;
    }

}