package ru.machine.learning.algorithms.bayes;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.Tuple3;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import io.vavr.collection.Set;
import io.vavr.collection.Stream;
import io.vavr.collection.Traversable;
import ru.machine.learning.algorithms.ProbaModel;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;

import java.util.function.Function;
import javax.annotation.Nonnull;

import static java.util.Comparator.comparing;

public class GaussianNaiveBayes implements ProbaModel {

    private Map<String, List<Double>> trainColNameToValues;
    private List<Integer> trainTargetLabels;

    @Override
    public GaussianNaiveBayes fit(@Nonnull Table train, @Nonnull IntColumn trainTarget) {
        this.trainTargetLabels = List.ofAll(trainTarget.asList());
        this.trainColNameToValues = List.ofAll(train.columns())
            .map(c -> ((DoubleColumn) c).asList())
            .map(List::ofAll)
            .zip(List.ofAll(train.columnNames()))
            .toMap(Tuple2::_2, Tuple2::_1);
        return this;
    }

    @Override
    public Seq<Integer> predict(@Nonnull Table test) {
        return predictProba(test)
            .map(Tuple2::_1);
    }

    @Override
    public Seq<Tuple2<Integer, Double>> predictProba(@Nonnull Table test) {
        var sd = standardDeviationByColumn();
        var m = meanByColumn();
        var targetLabels = targetLabelsProbabilities().keySet();
        Function<Row, Tuple2<Integer, Double>> toLabel = r -> internalPredictOnRow(r, sd, m, targetLabels);
        return Stream.ofAll(test.stream())
            .map(toLabel);
    }

    private Tuple2<Integer, Double> internalPredictOnRow(Row r,
                                                         Map<String, Map<Integer, Double>> mean,
                                                         Map<String, Map<Integer, Double>> sd,
                                                         Set<Integer> targetLabels) {
        return List.ofAll(r.columnNames())
            .map(n -> Tuple.of(r.getDouble(n), sd.get(n).get(), mean.get(n).get()))
            .flatMap(t3 -> targetLabels
                .map(tl -> Tuple.of(tl, t3
                         .map2(v -> v.get(tl).get())
                         .map3(v -> v.get(tl).get())
                     )
                )
            )
            .map(t2 -> t2
                .map2(t3 ->
                          (
                              1d / (Math.sqrt(2 * Math.PI) * t3._2)
                          )
                              * Math.exp(
                              -(
                                  (t3._1 - Math.pow(t3._3, 2))
                                      / (2 * Math.pow(t3._2, 2))
                              )
                          )
                )
            )
            .maxBy(comparing(Tuple2::_2))
            .get();
    }

    private Map<Integer, Double> targetLabelsProbabilities() {
        return trainTargetLabels
            .groupBy(Function.identity())
            .mapValues(Traversable::size)
            .mapValues(v -> (double) v / trainTargetLabels.size());
    }

    private Map<String, Map<Integer, Double>> meanByColumn() {
        return trainColNameToValues
            .mapValues(l -> l.zip(trainTargetLabels))
            .mapValues(t -> t.groupBy(Tuple2::_2)
                .mapValues(l -> l.map(Tuple2::_1))
            )
            .mapValues(m -> m
                .toMap(Tuple2::_1, t -> {
                    var n = countTargetLabel(t._1);
                    return 1d / n * t._2.reduce(Double::sum);
                })
            );
    }

    private Map<String, Map<Integer, Double>> standardDeviationByColumn() {
        var mean = meanByColumn();
        return trainColNameToValues
            .mapValues(l -> l.zip(trainTargetLabels))
            .mapValues(t -> t.groupBy(Tuple2::_2)
                .mapValues(l -> l.map(Tuple2::_1))
            )
            .toMap(
                Tuple2::_1,
                t -> {
                    var labelToValues = t._2;
                    var labelToMean = mean.get(t._1).get();
                    return labelToValues
                        .map(t2 -> Tuple.of(t2._1, t2._2, labelToMean.get(t2._1).get()));
                }
            )
            .mapValues(l -> l
                .toMap(
                    Tuple3::_1,
                    t -> {
                        var n = countTargetLabel(t._1);
                        return Math.sqrt(
                            1d / n * t._2
                                .map(x -> x - Math.pow(t._3, 2))
                                .reduce(Double::sum)
                        );
                    }
                )
            );
    }

    private Integer countTargetLabel(Integer label) {
        return trainTargetLabels
            .filter(t -> t.equals(label))
            .size();
    }
}
