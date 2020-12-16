package ru.machine.learning.algorithms.utils;

import io.vavr.Tuple2;
import io.vavr.Tuple4;
import io.vavr.collection.List;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.NumericColumn;
import tech.tablesaw.api.Table;

import java.util.function.Consumer;

@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class Preprocessing {

    private Table table;
    private Tuple4<Table, Table, DoubleColumn, DoubleColumn> splitted;

    public static Preprocessing over(Table table) {
        return new Preprocessing(table, null);
    }

    public Preprocessing dropCols(String... cols) {
        this.table.removeColumns(cols);
        return this;
    }

    public Preprocessing dropNa() {
        this.table = this.table.dropRowsWithMissingValues();
        return this;
    }

    public Preprocessing encodeExactly(String... cols) {
        List.of(cols)
            .map(n -> this.table.stringColumn(n))
            .forEach(
                col -> {
                    var map = List.ofAll(col.unique())
                        .zipWithIndex()
                        .toMap(Tuple2::_1, Tuple2::_2)
                        .mapValues(i -> (double) i);
                    var mappedCol = col
                        .map(s -> map.get(s).get(), s -> DoubleColumn.create(s, this.table.rowCount()));
                    this.table.replaceColumn(col.name(), mappedCol);
                }
            );
        return this;
    }

    public Preprocessing castToDoubleAndNormalize() {
        for (var name : this.table.columnNames()) {
            var original = this.table.numberColumn(name);
            var casted = castCol(original);
            var max = casted.max();
            var min = casted.min();
            casted = casted.map(v -> (v - min) / (max - min));
            this.table.removeColumns(original);
            this.table.addColumns(casted);
        }
        return this;
    }

    public Preprocessing trainTestSplitWith(String target, double alpha) {
        this.splitted = TrainTestSplit.split(this.table, target, alpha);
        return this;
    }

    public Preprocessing trainTestSplitWith(String target, double alpha, long seed) {
        TrainTestSplit.setSeed(seed);
        this.splitted = TrainTestSplit.split(this.table, target, alpha);
        return this;
    }

    // For metrics purpose only
    public Preprocessing peek(Consumer<Table> func) {
        func.accept(this.table.copy());
        return this;
    }

    // For metrics purpose only
    public Preprocessing peekSplitted(Consumer<Tuple4<Table, Table, DoubleColumn, DoubleColumn>> func) {
        func.accept(splitted);
        return this;
    }

    public Preprocessing castToDouble() {
        for (var name : this.table.columnNames()) {
            var original = this.table.numberColumn(name);
            var casted = castCol(original);
            this.table.removeColumns(original);
            this.table.addColumns(casted);
        }
        return this;
    }

    public Table table() {
        return this.table;
    }

    public Tuple4<Table, Table, DoubleColumn, DoubleColumn> splitted() {
        return this.splitted;
    }

    private DoubleColumn castCol(NumericColumn<?> col) {
        return col
            .map(o -> {
                if (o instanceof Integer i) {
                    return (double) i;
                } else if (o instanceof Float f) {
                    return (double) f;
                } else {
                    return (double) o;
                }
            }, s -> DoubleColumn.create(s, this.table.rowCount()));
    }
}
