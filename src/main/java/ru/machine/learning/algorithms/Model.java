package ru.machine.learning.algorithms;

import io.vavr.collection.Seq;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.Table;

import javax.annotation.Nonnull;

public interface Model {

    Model fit(@Nonnull Table train, @Nonnull IntColumn trainTarget);
    Seq<Integer> predict(@Nonnull Table test);
}
