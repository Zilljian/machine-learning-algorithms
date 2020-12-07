package ru.machine.learning.algorithms.model;

import io.vavr.collection.Seq;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import javax.annotation.Nonnull;

public interface Model {

    Model fit(@Nonnull Table train, @Nonnull DoubleColumn trainTarget);

    Seq<Double> predict(@Nonnull Table test);
}
