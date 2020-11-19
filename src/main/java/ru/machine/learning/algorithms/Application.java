package ru.machine.learning.algorithms;

import com.google.common.collect.Streams;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.SparkSession;
import tech.tablesaw.api.Table;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import static java.util.Optional.ofNullable;
import static java.util.stream.Collectors.groupingBy;

public class Application {

    private static final String TRAIN = Thread.currentThread().getContextClassLoader().getResource("train.csv").getPath();
    private static final String TEST = Thread.currentThread().getContextClassLoader().getResource("test.csv").getPath();
    private static final String LABELS = Thread.currentThread().getContextClassLoader().getResource("test_labels.csv").getPath();

    public static void main(String[] args) throws IOException {
        var testTable = Table.read().file("test.csv");
        var testLabels = Table.read().file("test_labels.csv");
        var trainTable = Table.read().file("train.csv");

        testLabels.removeColumns("PassengerId");
        var uniqueSex = testTable.column("Sex").unique().asList();
        var uniqueEmbarked = testTable.column("Embarked").unique().asList();

        /*testTable = Table.create(testTable.stream()
            .map(r -> {
                r.setInt("Sex", uniqueSex.indexOf(r.getString("Sex")));
                return r;
            }).map(r -> {
            r.setInt("Embarked", uniqueEmbarked.indexOf(r.getString("Embarked")));
            return r;
        }));*/


        var session = SparkSession.builder()
            .config("spark.master", "local")
            .master("local[*]")
            .config("job.local.dir", "file:/Users/ilya/IdeaProjects/k-nearest-neighbors/scr/main/resources")
            .appName("knn")
            .getOrCreate();

        var encoder = new StringIndexer()
            .setInputCols(new String[]{"Sex", "Embarked"})
            .setOutputCols(new String[]{"Sex_cat", "Embarked_cat"});

        var test = session.read()
            .format("csv")
            .option("header", true)
            .load(TEST);
        var labels = session.read()
            .format("csv")
            .option("header", true)
            .load(LABELS)
            .drop("PassengerId");

        test = encoder.fit(test)
            .transform(test)
            .drop("SibSp", "Parch", "Sex", "Embarked");

        var train = session.read()
            .format("csv")
            .option("header", true)
            .load(TRAIN);
        train = encoder.fit(train)
            .transform(train)
            .drop("SibSp", "Parch", "Sex", "Embarked");

        var start = System.currentTimeMillis();
        var result = new Knn(session, 5).fit(train).predict(test);
        var end = System.currentTimeMillis();

        var lb = labels.select("Survived").collectAsList();

        session.stop();

        var merge =
            Streams.zip(lb.stream(), result.entrySet().stream(), Map::entry)
                .map(e -> Map.entry(Integer.parseInt(e.getKey().getString(0)), Integer.valueOf((String) e.getValue().getValue())))
                .collect(groupingBy(e -> {
                    if (e.getKey().equals(e.getValue()) && e.getKey().equals(0)) {
                        return "tn";
                    } else if (e.getKey().equals(e.getValue()) && e.getKey().equals(1)) {
                        return "tp";
                    } else if (!e.getKey().equals(e.getValue()) && e.getKey().equals(0)) {
                        return "fp";
                    } else {
                        return "fn";
                    }
                }));
        System.out.printf("Compute in %f.2 seconds\n", (end - start) / 1000f);
        printMetrics(merge);
    }

    private static void printMetrics(Map<String, List<Entry<Integer, Integer>>> merge) {
        var truePositive = ofNullable(merge.get("tp")).orElse(List.of()).size();
        var trueNegative = ofNullable(merge.get("tn")).orElse(List.of()).size();
        var falsePositive = ofNullable(merge.get("fp")).orElse(List.of()).size();
        var falseNegative = ofNullable(merge.get("fn")).orElse(List.of()).size();

        var accuracy = (float) (trueNegative + truePositive) /
            (truePositive + trueNegative + falseNegative + falsePositive);
        var precision = (float) truePositive /
            (truePositive + falsePositive);
        var recall = (float) truePositive /
            (truePositive + falseNegative);
        var f1 = 2 * (recall * precision) / (recall + precision);

        System.out.printf(
            """
                Accuracy = %f.4
                Precision = %f.4
                Recall = %f.4
                F1 = %f.4
                """, accuracy, precision, recall, f1);
    }
}
