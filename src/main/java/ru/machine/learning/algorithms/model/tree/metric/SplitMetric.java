package ru.machine.learning.algorithms.model.tree.metric;

import io.vavr.Tuple3;
import io.vavr.collection.List;
import io.vavr.collection.Map;

public interface SplitMetric {

    SplitMetric setCategoricalCols(Map<String, Boolean> categoricalCols);

    Tuple3<Double, Double, String> findBestSplitCandidate(Map<String, List<Double>> colValues, List<Double> target);
}
