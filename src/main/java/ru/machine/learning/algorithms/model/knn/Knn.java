package ru.machine.learning.algorithms.model.knn;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.collection.LinearSeq;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Traversable;
import lombok.RequiredArgsConstructor;
import org.apache.commons.math3.ml.distance.CanberraDistance;
import org.apache.commons.math3.ml.distance.ChebyshevDistance;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.ml.distance.ManhattanDistance;
import ru.machine.learning.algorithms.model.Model;
import ru.machine.learning.algorithms.model.Util;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import javax.annotation.Nonnull;

import static java.util.Comparator.comparing;

@RequiredArgsConstructor
public class Knn implements Model {

    enum Metric {
        Euclidean,
        Manhattan,
        Canberra,
        Chebyshev
    }

    private final int k;
    private final DistanceMeasure metric;

    private List<Tuple2<List<Double>, Double>> trainRowsToTarget;

    public Knn(int k) {
        this.k = k;
        this.metric = new EuclideanDistance();
    }

    public Knn() {
        this.k = 5;
        this.metric = new EuclideanDistance();
    }

    public static Knn withParams(@Nonnull Map<String, ?> params) {
        var k = (int) params.get("k")
            .getOrElseThrow(() -> new RuntimeException("Param 'k' is expected, but doesn't exist"));
        var metricName = (Metric) params.get("metric")
            .getOrElseThrow(() -> new RuntimeException("Param 'metric' is expected, but doesn't exist"));
        var metric = switch (metricName) {
            case Euclidean -> new EuclideanDistance();
            case Manhattan -> new ManhattanDistance();
            case Canberra -> new CanberraDistance();
            case Chebyshev -> new ChebyshevDistance();
        };
        return new Knn(k, metric);
    }

    @Override
    public Knn fit(@Nonnull Table train, @Nonnull DoubleColumn trainTarget) {
        this.trainRowsToTarget = Util.toList(train)
            .zipWithIndex()
            .map(t -> t.map2(trainTarget::get));
        return this;
    }

    @Override
    public List<Double> predict(@Nonnull Table test) {
        var testRows = Util.toList(test);
        return testRows
            .map(e -> Tuple.of(e, trainRowsToTarget))
            .map(t -> t._2.map(train -> {
                var label = train._2;
                var distance = calculateDistance(t._1, train._1);
                return Tuple.of(label, distance);
            }))
            .map(this::resolveLabel);
    }

    private Double resolveLabel(LinearSeq<Tuple2<Double, Double>> neighbors) {
        return neighbors
            .sorted(comparing(Tuple2::_2))
            .take(k)
            .groupBy(Tuple2::_1)
            .mapValues(Traversable::size)
            .maxBy(comparing(Tuple2::_2))
            .getOrNull()._1;
    }

    private double calculateDistance(List<Double> testRow, List<Double> trainRow) {
        var testRowArray = new double[testRow.size()];
        var trainRowArray = new double[trainRow.size()];
        testRow.zipWithIndex()
            .forEach(t -> testRowArray[t._2] = t._1);
        trainRow.zipWithIndex()
            .forEach(t -> trainRowArray[t._2] = t._1);
        return metric.compute(testRowArray, trainRowArray);
    }
}
