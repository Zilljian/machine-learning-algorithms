package ru.machine.learning.algorithms.model.tree;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.Tuple3;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import io.vavr.collection.Traversable;
import lombok.Setter;
import lombok.experimental.Accessors;
import ru.machine.learning.algorithms.model.Model;
import ru.machine.learning.algorithms.model.Util;
import ru.machine.learning.algorithms.model.tree.Condition.Operation;
import ru.machine.learning.algorithms.model.tree.metric.InformationGain;
import ru.machine.learning.algorithms.model.tree.metric.SplitMetric;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import java.util.function.Function;
import java.util.function.Predicate;
import javax.annotation.Nonnull;

import static java.util.Comparator.comparing;
import static java.util.function.Predicate.not;

@Setter
@Accessors(chain = true)
public class DecisionTreeClassifier implements Model {

    private Node root;
    private int N = Integer.MAX_VALUE;
    private int depth = 0;
    private List<String> categoricalColNames = List.empty();
    private Map<String, Integer> colNameToIndex;
    private List<String> colNames;
    private Map<String, Boolean> categorical;
    private SplitMetric splitMetric = new InformationGain();

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
        this.splitMetric.setCategoricalCols(categorical);

        var list = List.of(root);
        while (depth++ < N && list.nonEmpty()) {
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
            return node.getTarget()
                .groupBy(Function.identity())
                .mapValues(Traversable::size)
                .maxBy(Tuple2::_2)
                .get()._1;
        }
        return switch (condition.getOperation()) {
            case EQUAL -> {
                if (row.get(condition.getFeatureIndex()).equals(condition.getValue())) {
                    yield traverse(node.getLeft(), row);
                } else {
                    yield traverse(node.getRight(), row);
                }
            }
            case GREATER -> {
                if (row.get(condition.getFeatureIndex()) > condition.getValue()) {
                    yield traverse(node.getLeft(), row);
                } else {
                    yield traverse(node.getRight(), row);
                }
            }
        };
    }

    public Model withCategorical(List<String> categoricalCols) {
        this.categoricalColNames = categoricalCols;
        return this;
    }

    public Model withSplitMetric(SplitMetric splitMetric) {
        this.splitMetric = splitMetric.setCategoricalCols(categorical);
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
        var withMaxGain = splitMetric.findBestSplitCandidate(node.getCols(), node.getTarget());
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
}
