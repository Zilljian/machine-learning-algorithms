package ru.machine.learning.algorithms.model.bayes;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.Tuple3;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import io.vavr.collection.Stream;
import io.vavr.collection.Traversable;
import ru.machine.learning.algorithms.model.ProbaModel;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;

import java.util.function.Function;
import javax.annotation.Nonnull;

import static java.util.Comparator.comparing;

public class GaussianNaiveBayes implements ProbaModel {

    private Map<String, List<Double>> trainColNameToValues;
    private List<Double> trainTargetLabels;
    private final Map<Double, Double> priorLabelsProbabilities;

    public GaussianNaiveBayes() {
        this.priorLabelsProbabilities = null;
    }

    public GaussianNaiveBayes(Map<Double, Double> priorLabelsProbabilities) {
        this.priorLabelsProbabilities = priorLabelsProbabilities;
    }

    public static GaussianNaiveBayes withParams(@Nonnull Map<String, ?> params) {
        return new GaussianNaiveBayes((Map<Double, Double>) params.get("priors")
            .getOrElseThrow(() -> new RuntimeException("Param 'priors' is expected, but doesn't exist"))
        );
    }

    @Override
    public GaussianNaiveBayes fit(@Nonnull Table train, @Nonnull DoubleColumn trainTarget) {
        this.trainTargetLabels = List.ofAll(trainTarget.asList());
        this.trainColNameToValues = listOnColumn(train);
        return this;
    }

    @Override
    public Seq<Double> predict(@Nonnull Table test) {
        return predictProba(test)
            .map(Tuple2::_1);
    }

    @Override
    public Seq<Tuple2<Double, Double>> predictProba(@Nonnull Table test) {
        var sd = standardDeviationByColumn();
        var mean = meanByColumn();
        Function<Row, Tuple2<Double, Double>> toLabel = r -> internalPredictOnRow(r, mean, sd);
        return Stream.ofAll(test.stream())
            .map(toLabel);
    }

    private Tuple2<Double, Double> internalPredictOnRow(Row r,
                                                        Map<String, Map<Double, Double>> mean,
                                                        Map<String, Map<Double, Double>> sd) {
        var labelsProba = priorLabelsProbabilities();
        return List.ofAll(r.columnNames())
            .map(n -> Tuple.of(r.getNumber(n), sd.get(n).get(), mean.get(n).get()))
            .flatMap(t3 -> labelsProba.keySet()
                .map(tl -> Tuple.of(tl, t3
                         .map2(v -> v.get(tl).get())
                         .map3(v -> v.get(tl).get())
                     )
                )
            )
            .map(t2 -> t2
                .map2(t3 ->
                          (
                              1d / Math.sqrt(2 * Math.PI * Math.pow(t3._2, 2))
                          )
                              * Math.exp(
                              -(
                                  (Math.pow(t3._1 - t3._3, 2))
                                      / (2 * Math.pow(t3._2, 2))
                              )
                          )
                )
            )
            .groupBy(Tuple2::_1)
            .mapValues(
                l -> {
                    var labelProba = labelsProba.get(l.get(0)._1).get();
                    return l
                        .map(Tuple2::_2)
                        .reduce((d1, d2) -> d1 * d2) * labelProba;
                }
            )
            .maxBy(comparing(Tuple2::_2))
            .get();
    }

    private Map<String, List<Double>> listOnColumn(Table train) {
        return List.ofAll(train.columns())
            .map(c -> List.ofAll(((DoubleColumn) c).asList()))
            .map(List::ofAll)
            .zip(List.ofAll(train.columnNames()))
            .toMap(Tuple2::_2, Tuple2::_1);
    }

    private Map<Double, Double> priorLabelsProbabilities() {
        return priorLabelsProbabilities != null ?
               priorLabelsProbabilities :
               trainTargetLabels
                   .groupBy(Function.identity())
                   .mapValues(Traversable::size)
                   .mapValues(v -> (double) v / trainTargetLabels.size());
    }

    private Map<String, Map<Double, Double>> meanByColumn() {
        return trainColNameToValues
            .mapValues(l -> l.zip(trainTargetLabels))
            .mapValues(t -> t.groupBy(Tuple2::_2)
                .mapValues(l -> l.map(Tuple2::_1))
            )
            .mapValues(m -> m
                .toMap(
                    Tuple2::_1,
                    t -> {
                        var n = countTargetLabel(t._1);
                        return 1d / n * t._2.reduce(Double::sum);
                    })
            );
    }

    private Map<String, Map<Double, Double>> standardDeviationByColumn() {
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
                                .map(x -> Math.pow(x - t._3, 2))
                                .reduce(Double::sum)
                        );
                    }
                )
            );
    }

    private Integer countTargetLabel(Double label) {
        return trainTargetLabels
            .filter(t -> t.equals(label))
            .size();
    }
}
