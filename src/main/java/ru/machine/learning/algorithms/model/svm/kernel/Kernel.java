package ru.machine.learning.algorithms.model.svm.kernel;

import io.vavr.collection.List;

import javax.annotation.Nonnull;

public interface Kernel {

    Double compute(@Nonnull List<Double> x1, @Nonnull List<Double> x2);
}
