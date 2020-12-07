package ru.machine.learning.algorithms.model;

import io.vavr.Tuple2;
import io.vavr.collection.Seq;
import tech.tablesaw.api.Table;

import javax.annotation.Nonnull;

public interface ProbaModel extends Model {

    Seq<Tuple2<Double, Double>> predictProba(@Nonnull Table test);
}
