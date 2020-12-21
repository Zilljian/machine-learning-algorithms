package ru.machine.learning.algorithms.model.svm.kernel;

import io.vavr.collection.List;

import javax.annotation.Nonnull;

public class LinearKernel implements Kernel {

    private final Double C;

    public LinearKernel(Double c) {
        this.C = c;
    }

    public LinearKernel() {
        this.C = 0d;
    }

    @Override
    public Double compute(@Nonnull List<Double> x1, @Nonnull List<Double> x2) {
        return x1.zip(x2)
            .map(t -> t._1 * t._2)
            .reduce(Double::sum) + C;
    }
}
