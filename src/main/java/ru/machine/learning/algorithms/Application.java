package ru.machine.learning.algorithms;

import io.vavr.collection.HashMap;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Traversable;
import ru.machine.learning.algorithms.bayes.GaussianNaiveBayes;
import ru.machine.learning.algorithms.utils.TrainTestSplit;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import java.io.IOException;

public class Application {

    private static final String DATA = Thread.currentThread().getContextClassLoader().getResource("train.csv").getPath();

    public static void main(String[] args) throws IOException {
        var table = Table.read().file(DATA);
        System.out.printf("Init table shape: %s\n\n", table.shape());
        table.removeColumns("Cabin", "Name", "PassengerId", "Ticket", "SibSp", "Parch");
        var cleanTable = table.dropRowsWithMissingValues();

        var mapEmbarkedToDouble = HashMap.of(
            "C", 0.0,
            "Q", 1.0,
            "S", 2.0);
        var mapSexToDouble = HashMap.of(
            "male", 1.0,
            "female", 0.0);

        var mappedSex = cleanTable.stringColumn("Sex")
            .map(s -> mapSexToDouble.getOrElse(s, 1.0), s -> DoubleColumn.create(s, cleanTable.rowCount()));
        var mappedEmbarked = cleanTable.stringColumn("Embarked")
            .map(s -> mapEmbarkedToDouble.getOrElse(s, 1.0), s -> DoubleColumn.create(s, cleanTable.rowCount()));

        cleanTable.replaceColumn("Sex", mappedSex);
        cleanTable.replaceColumn("Embarked", mappedEmbarked);

        var splitted = TrainTestSplit.split(cleanTable, "Survived", 0.8);

        System.out.printf("Full table shape: %s\n\n", cleanTable.shape());

        System.out.printf("Train labels size: %s\n", splitted._3.size());
        System.out.printf("Class 0 size in test: %s\n", splitted._3.filter(i -> i == 0).size());
        System.out.printf("Class 1 size in test: %s\n\n", splitted._3.filter(i -> i == 1).size());

        System.out.printf("Test labels size: %s\n", splitted._4.size());
        System.out.printf("Class 0 size in test: %s\n", splitted._4.filter(i -> i == 0).size());
        System.out.printf("Class 1 size in test: %s\n\n", splitted._4.filter(i -> i == 1).size());

        for (var n : List.range(1, 2)) {
            System.out.printf("\n\nN = %d\n\n", n);
            var start = System.nanoTime();
            /*var predicted = new Knn(n)
                .fit(splitted._1, splitted._3)
                .predict(splitted._2);*/
            var predicted = new GaussianNaiveBayes()
                .fit(splitted._1, splitted._3)
                .predict(splitted._2);
            var end = System.nanoTime();
            var testLabels = List.ofAll(splitted._4.asList());

            var merge =
                testLabels.zip(predicted)
                    .groupBy(t -> {
                        if (t._1.equals(t._2()) && t._1.equals(0)) {
                            return "tn";
                        } else if (t._1.equals(t._2) && t._1.equals(1)) {
                            return "tp";
                        } else if (!t._1.equals(t._2) && t._1.equals(0)) {
                            return "fp";
                        } else {
                            return "fn";
                        }
                    })
                    .mapValues(Traversable::size);

            System.out.printf("Computed in %f.2 seconds\n", (end - start) / 1000000000f);

            printMetrics(merge);
        }
    }

    private static void printMetrics(Map<String, Integer> merge) {
        var truePositive = merge.getOrElse("tp", 0);
        var trueNegative = merge.getOrElse("tn", 0);
        var falsePositive = merge.getOrElse("fp", 0);
        var falseNegative = merge.getOrElse("fn", 0);

        var accuracy = (float) (trueNegative + truePositive) /
            (truePositive + trueNegative + falseNegative + falsePositive);
        var precision = (float) truePositive /
            (truePositive + falsePositive);
        var recall = (float) truePositive /
            (truePositive + falseNegative);
        var f1 = 2 * (recall * precision) / (recall + precision);

        System.out.printf(
            """
                True Positive = %d
                True Negative = %d
                False Positive = %d
                False Negative = %d
                                
                Accuracy = %f.4
                Precision = %f.4
                Recall = %f.4
                F1 = %f.4
                """,
            truePositive, trueNegative, falsePositive, falseNegative,
            accuracy, precision, recall, f1);
    }
}
