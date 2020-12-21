package ru.machine.learning.algorithms.model.knn;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import io.vavr.collection.Traversable;
import lombok.RequiredArgsConstructor;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import ru.machine.learning.algorithms.model.Model;
import ru.machine.learning.algorithms.model.Util;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import javax.annotation.Nonnull;

import static java.util.Comparator.comparing;

@RequiredArgsConstructor
public class Knn implements Model {

    private final int k;
    private DistanceMeasure metric = new EuclideanDistance();

    private List<Tuple2<List<Double>, Double>> trainRowsToTarget;

    public Knn() {
        this.k = 5;
    }

    public Knn(int k, DistanceMeasure metric) {
        this.k = 5;
        this.metric = metric;
    }

    public static Knn withParams(@Nonnull Map<String, ?> params) {
        var k = (int) params.get("k")
            .getOrElseThrow(() -> new RuntimeException("Param 'k' is expected, but doesn't exist"));
        var metric = (Metric) params.get("metric")
            .getOrElseThrow(() -> new RuntimeException("Param 'metric' is expected, but doesn't exist"));
        return new Knn(k, metric.get());
    }

    public Knn withMetric(Metric metric) {
        this.metric = metric.get();
        return this;
    }

    @Override
    public Knn fit(@Nonnull Table train, @Nonnull DoubleColumn trainTarget) {
        this.trainRowsToTarget = Util.toList(train)
            .zipWithIndex()
            .map(t -> t.map2(trainTarget::get));
        return this;
    }

    @Override
    public Seq<Double> predict(@Nonnull Table test) {
        var testRows = Util.toList(test);
        return testRows
            .map(r -> trainRowsToTarget.map(train -> {
                var label = train._2;
                var distance = calculateDistance(r, train._1);
                return Tuple.of(label, distance);
            }))
            .map(this::resolveLabel);
    }

    private Double resolveLabel(List<Tuple2<Double, Double>> neighbors) {
        return neighbors
            .sorted(comparing(Tuple2::_2))
            .take(k)
            .groupBy(Tuple2::_1)
            .mapValues(Traversable::size)
            .maxBy(comparing(Tuple2::_2))
            .getOrNull()._1;
    }

    private double calculateDistance(List<Double> testRow, List<Double> trainRow) {
        return metric.compute(Util.listToArray(testRow), Util.listToArray(trainRow));
    }
}
