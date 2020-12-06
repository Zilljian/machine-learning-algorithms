package ru.machine.learning.algorithms.utils;

import io.vavr.Tuple;
import io.vavr.Tuple4;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.Table;

import static java.lang.String.format;

public class TrainTestSplit {

    // train, test, train labels, test labels
    public static Tuple4<Table, Table, IntColumn, IntColumn> split(Table table, String targetName, double alpha) {
        if (0 < alpha && alpha < 1) {
            var splitted = table.sampleSplit(alpha);
            var trainTarget = splitted[0].intColumn(targetName);
            var testTarget = splitted[1].intColumn(targetName);
            return Tuple.of(
                Table.create(splitted[0].removeColumns(targetName).columns()),
                Table.create(splitted[1].removeColumns(targetName).columns()),
                trainTarget,
                testTarget);
        }
        throw new RuntimeException(format("Wrong alpha %s\n", alpha));
    }
}
