package ru.machine.learning.algorithms;

import io.vavr.Tuple2;
import io.vavr.collection.Seq;
import tech.tablesaw.api.Table;

import javax.annotation.Nonnull;

public interface ProbaModel extends Model {

    Seq<Tuple2<Integer, Double>> predictProba(@Nonnull Table test);
}
