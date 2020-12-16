package ru.machine.learning.algorithms.model.tree;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.Tuple3;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import io.vavr.collection.Traversable;
import lombok.RequiredArgsConstructor;
import ru.machine.learning.algorithms.model.Model;
import ru.machine.learning.algorithms.model.Util;
import ru.machine.learning.algorithms.model.tree.Condition.Operation;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import java.util.function.Function;
import java.util.function.Predicate;
import javax.annotation.Nonnull;

import static java.util.Comparator.comparing;
import static java.util.Comparator.comparingDouble;
import static java.util.function.Predicate.not;

@RequiredArgsConstructor
public class DecisionTreeClassifier implements Model {

    private Node root;
    private final int N;
    private int depth = 0;
    private List<String> categoricalColNames = List.empty();
    private Map<String, Integer> colNameToIndex;
    private List<String> colNames;
    private Map<String, Boolean> categorical;

    @Override
    public Model fit(@Nonnull Table train, @Nonnull DoubleColumn trainTarget) {
        var dataToTarget = Util.toList(train)
            .zipWithIndex()
            .map(t -> t.map2(trainTarget::get));
        var target = List.ofAll(trainTarget.asList());

        this.colNames = List.ofAll(train.columnNames());
        this.colNameToIndex = List.ofAll(train.columnNames())
            .zipWithIndex()
            .toMap(Tuple2::_1, Tuple2::_2);
        this.root = Node.builder()
            .target(target)
            .cols(Util.toListOnColumn(train))
            .rowToTarget(dataToTarget)
            .build()
            .t();
        this.categorical = root.getCols()
            .mapValues(v -> v.toSet().size())
            .toMap(t -> t.map2(v -> categoricalColNames.contains(t._1) || v < 5));

        var list = List.of(root);
        while (depth++ < 100000 && list.nonEmpty()) {
            var tmp = List.<Node>empty();
            for (var node : list) {
                tmp = tmp.appendAll(trySplit(node));
            }
            list = tmp;
        }
        return this;
    }

    @Override
    public Seq<Double> predict(@Nonnull Table test) {
        var rowList = Util.toList(test);
        return rowList.map(r -> traverse(root, r));
    }

    public Double traverse(Node node, List<Double> row) {
        var condition = node.getCondition();
        if (condition == null) {
            return node.getTarget().get(0);
        }
        return switch (condition.getOperation()) {
            case EQUAL -> {
                if (row.get(condition.getFeatureIndex()).equals(condition.getValue())) {
                    if (node.hasLeft()) {
                        yield traverse(node.getLeft(), row);
                    }
                } else if (node.hasRight()) {
                    yield traverse(node.getRight(), row);
                }
                yield node.getTarget().get(0);
            }
            case GREATER -> {
                if (row.get(condition.getFeatureIndex()) > condition.getValue()) {
                    if (node.hasLeft()) {
                        yield traverse(node.getLeft(), row);
                    }
                } else if (node.hasRight()) {
                    yield traverse(node.getRight(), row);
                }
                yield node.getTarget().get(0);
            }
        };
    }

    public Model withCategorical(List<String> categoricalCols) {
        this.categoricalColNames = categoricalCols;
        return this;
    }

    private List<Node> trySplit(Node node) {
        split(node);
        var list = List.<Node>empty();
        if (node.hasLeft() && node.getLeft().splittabel()) {
            list = list.append(split(node.getLeft()));
        }
        if (node.hasRight() && node.getRight().splittabel()) {
            list = list.append(split(node.getRight()));
        }
        return list;
    }

    private Node split(Node node) {
        var classes = node.getRowToTarget()
            .map(Tuple2::_2)
            .groupBy(Function.identity())
            .mapValues(Traversable::size);
        var currentEntropy = computeEntropy(classes, node.getRowToTarget().size());
        var splitCandidates = node.getCols()
            .map(t2 -> Tuple.of(t2._1, t2._2, node.getTarget()))
            .map(t3 -> categorical.getOrElse(t3._1, false) ?
                       choose(t3._2, t3._3, currentEntropy).append(t3._1) :
                       chooseWithContinuous(t3._2, t3._3, currentEntropy).append(t3._1)
            );
        var withMaxGain = splitCandidates.maxBy(comparingDouble(t -> t._2)).get();
        var splitCondition = conditionOf(withMaxGain);
        var sortedData = node.getRowToTarget()
            .sorted(comparing(t -> t._1.get(splitCondition.getFeatureIndex())));
        var splittedData = splitByPredicate(sortedData, splitPredicate(splitCondition));
        var splittedData1 = splittedData._1;
        var splitedData2 = splittedData._2;
        if (splittedData1.isEmpty() || splitedData2.isEmpty()) {
            return node;
        }
        node.setCondition(splitCondition)
            .setLeft(nodeOf(splittedData1))
            .setRight(nodeOf(splitedData2));
        return node;
    }

    private Tuple2<Double, Double> choose(List<Double> values, List<Double> target, Double parentEntropy) {
        double n = values.size();
        var featureToEntropy = values // col values
            .zip(target)// zip with target
            .groupBy(Tuple2::_1) // group by unique feature values
            .mapValues(v -> v.map(Tuple2::_2)) // map to: unique val -> target
            .mapValues(v -> Tuple.of(
                v.groupBy(Function.identity())
                    .mapValues(Traversable::size),
                v.size())
            ) // map to: unique val -> (map: unique target -> count)
            .mapValues(t2 -> t2.map1(v -> computeEntropy(v, t2._2))); // calculate entropy for each unique feature value
        var weightedSum = featureToEntropy.values()
            .map(x -> x._1 * (x._2 / n))
            .reduce(Double::sum);
        var minEntropyValue = featureToEntropy
            .minBy(t -> t._2._1)
            .map(Tuple2::_1).get();
        var informationGain = parentEntropy - weightedSum;
        return Tuple.of(minEntropyValue, informationGain);
    }

    private Tuple2<Double, Double> chooseWithContinuous(List<Double> values, List<Double> target, Double parentEntropy) {
        double n = values.size();

        var splitsOfValues =
            List.<Tuple2<List<Tuple2<Double, Double>>, List<Tuple2<Double, Double>>>>empty();
        var sorted = values
            .zip(target)// zip with target
            .sorted(comparing(Tuple2::_1));

        for (var i : List.range(1, sorted.size())) {
            splitsOfValues = splitsOfValues.append(sorted.splitAt(i));
        }

        var maxInformationGainSplit = splitsOfValues
            .map(selectThreshold())
            .map(computeInformationGain(parentEntropy, n))
            .maxBy(t2 -> t2._2)
            .get();
        var minEntropyValue = maxInformationGainSplit._1;
        var maxInformationGain = maxInformationGainSplit._2;
        return Tuple.of(minEntropyValue, maxInformationGain);
    }

    private Tuple2<
        List<Tuple2<List<Double>, Double>>,
        List<Tuple2<List<Double>, Double>>> splitByPredicate(List<Tuple2<List<Double>, Double>> sortedData,
                                                             Predicate<Tuple2<List<Double>, Double>> predicate) {
        var part1 = sortedData
            .filter(predicate);
        var part2 = sortedData.filter(not(predicate));
        return Tuple.of(part1, part2);
    }

    private Condition conditionOf(Tuple3<Double, Double, String> withMaxGain) {
        var splitFeatureIndex = colNameToIndex.get(withMaxGain._3).get();
        return Condition.builder()
            .featureIndex(splitFeatureIndex)
            .featureName(withMaxGain._3)
            .value(withMaxGain._1)
            .operation(
                categorical.getOrElse(withMaxGain._3, false) ?
                Operation.EQUAL :
                Operation.GREATER
            )
            .build();
    }

    private Node nodeOf(List<Tuple2<List<Double>, Double>> data) {
        return Node.builder()
            .rowToTarget(data)
            .target(data.map(Tuple2::_2))
            .cols(Util.transformRowsToCols(data, colNames))
            .build()
            // FIXME debugging param
            .t();
    }

    private Predicate<Tuple2<List<Double>, Double>> splitPredicate(Condition c) {
        return t -> {
            var value = t._1.get(c.getFeatureIndex());
            var splitter = c.getValue();
            return switch (c.getOperation()) {
                case EQUAL -> value.equals(splitter);
                case GREATER -> value > splitter;
            };
        };
    }

    private Function<
        Tuple2<List<Tuple2<Double, Double>>, List<Tuple2<Double, Double>>>,
        Tuple3<Double, List<Double>, List<Double>>> selectThreshold() {
        return t2 -> {
            var splitValue = t2._1.map(Tuple2::_1).min().get();
            return Tuple.of(splitValue, t2._1.map(Tuple2::_2), t2._2.map(Tuple2::_2));
        };
    }

    private Function<
        Tuple3<Double, List<Double>, List<Double>>,
        Tuple2<Double, Double>> computeInformationGain(double parentEntropy, double n) {
        return t3 -> {
            var leftSize = t3._2.size();
            var leftPart = t3._2
                .groupBy(Function.identity())
                .mapValues(Traversable::size);
            var leftPartEntropy = computeEntropy(leftPart, leftSize);
            var rightSize = t3._3.size();
            var rightPart = t3._3
                .groupBy(Function.identity())
                .mapValues(Traversable::size);
            var rightPartEntropy = computeEntropy(rightPart, rightSize);
            var weightedSum = leftPartEntropy * (leftSize / n) + rightPartEntropy * (rightSize / n);
            var informationGain = parentEntropy - weightedSum;
            return Tuple.of(t3._1, informationGain);
        };
    }

    private Double computeEntropy(Map<Double, Integer> map, double n) {
        var logBase = Math.log(2);
        return map
            .mapValues(x -> -(x / n) * Math.log(x / n) / logBase)
            .values()
            .reduce(Double::sum);
    }
}
