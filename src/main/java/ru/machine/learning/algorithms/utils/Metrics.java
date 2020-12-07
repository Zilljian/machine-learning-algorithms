package ru.machine.learning.algorithms.utils;

import io.vavr.Tuple4;
import io.vavr.collection.List;
import io.vavr.collection.Seq;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.DoublePredicate;
import java.util.function.Supplier;

public class Metrics {

    public static void printBinaryClassificationMetrics(Seq<Double> real, Seq<Double> predicted) {
        var truePositive = new AtomicInteger(0);
        var trueNegative = new AtomicInteger(0);
        var falsePositive = new AtomicInteger(0);
        var falseNegative = new AtomicInteger(0);

        var classes = real.toSortedSet().toList();
        var c0 = classes.get(0);
        var c1 = classes.get(1);

        real.zip(predicted)
            .forEach(t -> {
                if (t._1.equals(t._2()) && t._1.equals(c0)) {
                    trueNegative.incrementAndGet();
                } else if (t._1.equals(t._2) && t._1.equals(c1)) {
                    truePositive.incrementAndGet();
                } else if (!t._1.equals(t._2) && t._1.equals(c0)) {
                    falsePositive.incrementAndGet();
                } else {
                    falseNegative.incrementAndGet();
                }
            });

        var accuracy = (float) (trueNegative.get() + truePositive.get()) /
            (truePositive.get() + trueNegative.get() + falseNegative.get() + falsePositive.get());
        var precision = (float) truePositive.get() /
            (truePositive.get() + falsePositive.get());
        var recall = (float) truePositive.get() /
            (truePositive.get() + falseNegative.get());
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
            truePositive.get(), trueNegative.get(), falsePositive.get(), falseNegative.get(),
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
