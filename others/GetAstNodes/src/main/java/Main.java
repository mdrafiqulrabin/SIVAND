import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.TreeVisitor;

import java.util.ArrayList;
import java.util.List;

/**
 * Input: string_program
 * Output: ast_nodes
 */
public class Main {
    static final String INNER_DELIMITER = " ___INNER___ ";
    static final String OUTER_DELIMITER = " ___OUTER___ ";

    public static void main(String[] args) {
        String strCode = args[0];
        CompilationUnit cu = getParseUnit(strCode);
        List<String> allTerminals = getTerminalNodes(cu);
        List<String> allClasses = getNodeClasses(cu);
        String astData = String.join(INNER_DELIMITER, allTerminals)
                + OUTER_DELIMITER
                + String.join(INNER_DELIMITER, allClasses);
        System.out.print(astData);
    }

    public static List<String> getTerminalNodes (CompilationUnit cu) {
        List<String> allTerminals = new ArrayList<>();
        MethodDeclaration md = cu.findAll(MethodDeclaration.class).get(0);
        md.setName("METHOD_NAME"); // skip actual method name
        new TreeVisitor() {
            @Override
            public void process(Node node) {
                if (node.getChildNodes().size() == 0) {
                    allTerminals.add(node.toString());
                }
            }
        }.visitPreOrder(md);
        return allTerminals;
    }

    public static List<String> getNodeClasses(CompilationUnit cu) {
        List<String> allClasses = new ArrayList<>();
        MethodDeclaration md = cu.findAll(MethodDeclaration.class).get(0);
        new TreeVisitor() {
            @Override
            public void process(Node node) {
                String clsStr = node.getClass().toString();
                // clsStr = clsStr.replace("class com.github.javaparser.ast.", "");
                String lastName = clsStr.substring(clsStr.lastIndexOf('.') + 1);
                allClasses.add(lastName);
            }
        }.visitPreOrder(md);
        return allClasses;
    }

    public static CompilationUnit getParseUnit(String txtCode) {
        CompilationUnit cu = null;
        try {
            // add tmp class to parse
            if(!txtCode.startsWith("class")) txtCode = "class T { \n" + txtCode + "\n}";
            // remove comments
            StaticJavaParser.getConfiguration().setAttributeComments(false);
            // parse code
            cu = StaticJavaParser.parse(txtCode);
        } catch (Exception ignore) {}
        return cu;
    }

}
