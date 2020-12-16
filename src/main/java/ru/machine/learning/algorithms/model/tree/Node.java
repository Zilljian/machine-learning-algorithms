package ru.machine.learning.algorithms.model.tree;

import io.vavr.Tuple2;
import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Traversable;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.experimental.Accessors;

import java.util.function.Function;

@Data
@AllArgsConstructor
@Builder
@Accessors(chain = true)
class Node {

    private Condition condition;
    private Node left;
    private Node right;
    private List<Tuple2<List<Double>, Double>> rowToTarget;
    private Map<String, List<Double>> cols;
    private List<Double> target;
    private Map<Double, Integer> t;

    boolean splittabel() {
        return this.target
            .toSet()
            .size() > 1;
    }

    boolean hasLeft() {
        return left != null;
    }

    boolean hasRight() {
        return right != null;
    }

    // For debugging purpose
    Node t() {
        this.t = target
            .groupBy(Function.identity())
            .mapValues(Traversable::size);
        return this;
    }
}
