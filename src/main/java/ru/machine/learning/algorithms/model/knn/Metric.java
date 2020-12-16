package ru.machine.learning.algorithms.model.knn;

import org.apache.commons.math3.ml.distance.CanberraDistance;
import org.apache.commons.math3.ml.distance.ChebyshevDistance;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.ml.distance.ManhattanDistance;

public enum Metric {

    Euclidean(new EuclideanDistance()),
    Manhattan(new ManhattanDistance()),
    Canberra(new CanberraDistance()),
    Chebyshev(new ChebyshevDistance());

    public final DistanceMeasure metric;

    Metric(DistanceMeasure metric) {
        this.metric = metric;
    }

    DistanceMeasure get() {
        return metric;
    }
}