import org.lemurproject.galago.core.retrieval.Results;
import org.lemurproject.galago.core.retrieval.Retrieval;
import org.lemurproject.galago.core.retrieval.RetrievalFactory;
import org.lemurproject.galago.core.retrieval.query.Node;
import org.lemurproject.galago.core.retrieval.query.StructuredQuery;
import org.lemurproject.galago.utility.Parameters;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Util {
    public static final String indexPath = "C:\\Users\\Alan\\Projects School\\646\\HW2\\index";

    public static Parameters getInitParams_T1() {
        //- Define what index to query
        Parameters queryParams = Parameters.create();
        queryParams.set("index", indexPath);
        //- Set how many docs to return
        queryParams.set("requested", 1000);
        //- Do verbose output
        queryParams.set("verbose", true);
        return queryParams;
    }

    public static Parameters getInitParams_T2() {
        //- Define what index to query
        Parameters queryParams = Parameters.create();
        queryParams.set("index", indexPath);
        //- Set how many docs to return
        queryParams.set("requested", 1000);
        //- Do verbose output
        queryParams.set("verbose", true);
        queryParams.set("mu", 1000);
        queryParams.set("scorer", "dirichlet");
        queryParams.set("defaultTextPart", "postings.krovetz");
        return queryParams;
    }

    public static void run(Parameters queryParams, String outputFile, boolean rm) {
        try {
            Retrieval ret = RetrievalFactory.create(queryParams);

            Map<String, Node> queries = parseQueries(ret, queryParams, rm);
            System.out.println(queries.get(queries.keySet().toArray()[0]).toString());
            Map<String, Results> resultsMap = new HashMap<>();

            for (String qid : queries.keySet()) {
                Results results = ret.executeQuery(queries.get(qid), queryParams);
                resultsMap.put(qid, results);
            }

            writeToFile(outputFile, resultsMap);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void writeToFile(String fileName, Map<String, Results> resultsMap) throws IOException {
        String fullFileName = "galago-output/" + fileName + ".glg";
        File file = new File(fullFileName);
        file.createNewFile();
        PrintStream ps = new PrintStream(fullFileName);
        for (String qid : resultsMap.keySet()) {
            resultsMap.get(qid).printToTrecrun(ps, qid, "galago");
        }
        ps.flush();
        ps.close();
    }

    private static Map<String, Node> parseQueries(Retrieval ret, Parameters queryParams, boolean rm) throws Exception {
        FileInputStream fis = new FileInputStream("query.titles.tsv");
        Scanner sc = new Scanner(fis);

        Map<String, Node> result = new HashMap<>();
        while (sc.hasNextLine()) {
            String line = sc.nextLine();
            int separatorIdx = line.indexOf('\t');
            String qid = line.substring(0, separatorIdx);
            String queryText = line.substring(separatorIdx+1);

            if (rm) {
                queryText = "#rm(" + queryText + ")";
            }

            Node q = StructuredQuery.parse(queryText);
            Node transq = ret.transformQuery(q, queryParams);
            result.put(qid, transq);
        }
        return result;
    }
}
