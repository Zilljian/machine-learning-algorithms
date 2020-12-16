package ru.machine.learning.algorithms.model.tree.metric;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.Tuple3;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Traversable;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.util.function.Function;

import static java.util.Comparator.comparing;
import static java.util.Comparator.comparingDouble;
import static java.util.function.Predicate.not;

@Setter
@Accessors(chain = true)
public class GiniIndex implements SplitMetric {

    private Map<String, Boolean> categoricalCols;
    private int minEntries = 10;

    @Override
    public Tuple3<Double, Double, String> findBestSplitCandidate(Map<String, List<Double>> colValues, List<Double> target) {
        var splitCandidates = colValues
            .map(t2 -> categoricalCols.getOrElse(t2._1, false) ?
                       choose(t2._2, target).append(t2._1) :
                       chooseWithContinuous(t2._2, target).append(t2._1)
            );
        return splitCandidates.minBy(comparingDouble(Tuple3::_2)).get();
    }

    private Tuple2<Double, Double> choose(List<Double> values, List<Double> target) {
        double n = values.size();
        var valueToTarget = values
            .zip(target);
        var minGiniIndexSplit = valueToTarget
            .groupBy(Tuple2::_1) // group by unique feature values
            .mapValues(v -> v.map(Tuple2::_2))
            .map(t -> t.append(valueToTarget.filter(not(t2 -> t2._1.equals(t._1))).map(Tuple2::_2))) // zip with all other value
            .filter(t3 -> t3._2.size() > minEntries && t3._3.size() > minEntries)
            .map(computeWeightedSumOfSplit(n))
            .minBy(Tuple2::_2)
            .getOrNull(); // calculate entropy for each unique feature value

        // if cannot be splitted
        if (minGiniIndexSplit == null) {
            return Tuple.of(1d, 1d);
        }

        var splitValue = minGiniIndexSplit._1;
        var minGiniIndex = minGiniIndexSplit._2;
        return Tuple.of(splitValue, minGiniIndex);
    }

    private Tuple2<Double, Double> chooseWithContinuous(List<Double> values, List<Double> target) {
        double n = values.size();
        var splitsOfValues = splitOnEachValue(values, target);

        var minGiniIndexSplit = splitsOfValues
            .map(selectThreshold())
            .filter(t3 -> t3._2.size() > minEntries && t3._3.size() > minEntries)
            .map(computeWeightedSumOfSplit(n))
            .minBy(Tuple2::_2)
            .getOrNull();

        // if cannot be splitted
        if (minGiniIndexSplit == null) {
            return Tuple.of(1d, 1d);
        }

        var splitValue = minGiniIndexSplit._1;
        var minGiniIndex = minGiniIndexSplit._2;
        return Tuple.of(splitValue, minGiniIndex);
    }

    private List<Tuple2<List<Tuple2<Double, Double>>, List<Tuple2<Double, Double>>>> splitOnEachValue(List<Double> values, List<Double> target) {
        var splits =
            List.<Tuple2<List<Tuple2<Double, Double>>, List<Tuple2<Double, Double>>>>empty();
        var sortedValues = values
            .zip(target)// zip with target
            .sorted(comparing(Tuple2::_1));

        for (var i : List.range(1, sortedValues.size())) {
            splits = splits.append(sortedValues.splitAt(i));
        }
        return splits;
    }

    private Function<
        Tuple3<Double, List<Double>, List<Double>>,
        Tuple2<Double, Double>> computeWeightedSumOfSplit(double n) {
        return t3 -> {
            var leftSize = t3._2.size();
            var leftPart = t3._2
                .groupBy(Function.identity())
                .mapValues(Traversable::size);
            var leftPartIndex = computeGiniIndex(leftPart, leftSize);
            var rightSize = t3._3.size();
            var rightPart = t3._3
                .groupBy(Function.identity())
                .mapValues(Traversable::size);
            var rightPartIndex = computeGiniIndex(rightPart, rightSize);
            var weightedSum = leftPartIndex * (leftSize / n) + rightPartIndex * (rightSize / n);
            return Tuple.of(t3._1, weightedSum);
        };
    }

    private Double computeGiniIndex(Map<Double, Integer> map, double n) {
        return 1d - map
            .mapValues(x -> x / n)
            .values()
            .reduce(Double::sum);
    }

    private Function<
        Tuple2<List<Tuple2<Double, Double>>, List<Tuple2<Double, Double>>>,
        Tuple3<Double, List<Double>, List<Double>>> selectThreshold() {
        return t -> {
            var splitValue = (t._1.map(Tuple2::_1).max().get() + t._2.map(Tuple2::_1).min().get()) / 2;
            return Tuple.of(splitValue, t._1.map(Tuple2::_2), t._2.map(Tuple2::_2));
        };
    }
}
