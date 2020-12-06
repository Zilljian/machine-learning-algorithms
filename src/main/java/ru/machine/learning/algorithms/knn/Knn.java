package ru.machine.learning.algorithms.knn;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.collection.LinearSeq;
import io.vavr.collection.List;
import io.vavr.collection.List.Nil;
import io.vavr.collection.Traversable;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.Table;

import javax.annotation.Nonnull;

import static java.util.Comparator.comparing;

public class Knn {

    private final int K;

    private List<Tuple2<List<Double>, Integer>> trainRowsToTarget;

    public Knn(int k) {
        this.K = k;
    }

    public Knn fit(@Nonnull Table train, @Nonnull IntColumn trainTarget) {
        this.trainRowsToTarget = toList(train)
            .zipWithIndex()
            .map(t -> t.map2(trainTarget::get));
        return this;
    }

    public List<Integer> predict(Table test) {
        var testRows = toList(test);
        return testRows
            .map(e -> Tuple.of(e, trainRowsToTarget))
            .map(t -> t._2.map(train -> {
                var label = train._2;
                var distance = calculateDistance(t._1, train._1);
                return Tuple.of(label, distance);
            }))
            .map(this::resolveLabel);
    }

    private Integer resolveLabel(LinearSeq<Tuple2<Integer, Double>> neighbors) {
        return neighbors
            .sorted(comparing(Tuple2::_2))
            .take(K)
            .groupBy(Tuple2::_1)
            .mapValues(Traversable::size)
            .maxBy(comparing(Tuple2::_2))
            .getOrNull()._1;
    }

    private double calculateDistance(List<Double> testRow, List<Double> trainRow) {
        return Math.sqrt(
            testRow.zip(trainRow)
                .map(t -> Math.pow(t._1 - t._2, 2))
                .reduce(Double::sum)
        );
    }

    private List<List<Double>> toList(Table table) {
        List<List<Double>> result = Nil.instance();
        for (var e : table) {
            result = result.prepend(List.ofAll(e.columnNames()).map(e::getNumber));
        }
        return result;
    }
}
