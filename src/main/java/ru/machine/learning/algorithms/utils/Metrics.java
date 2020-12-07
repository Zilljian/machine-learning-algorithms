package ru.machine.learning.algorithms.utils;

import io.vavr.Tuple4;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import java.util.function.DoublePredicate;
import java.util.function.Supplier;

public class Metrics {

    public static void printClassificationMetrics(Map<String, Integer> merge) {
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

    public static void describeData(Tuple4<Table, Table, DoubleColumn, DoubleColumn> splitted) {
        var classes = List.ofAll(splitted._3.unique().asList());

        System.out.printf("Train labels size: %s\n", splitted._3.size());
        classes.forEach(
            c -> {
                DoublePredicate predicate = d -> d == c;
                System.out.printf("Class %.0f size in test: %s\n", c, splitted._3.filter(predicate).size());
            }
        );

        System.out.printf("\nTest labels size: %s\n", splitted._4.size());
        classes.forEach(
            c -> {
                DoublePredicate predicate = d -> d == c;
                System.out.printf("Class %.0f size in test: %s\n", c, splitted._4.filter(predicate).size());
            }
        );
    }

    public static <T> T withTime(Supplier<T> func) {
        var start = System.nanoTime();
        var result = func.get();
        var end = System.nanoTime();
        System.out.printf("Computed in %f.2 seconds\n", (end - start) / 1000000000f);
        return result;
    }
}
