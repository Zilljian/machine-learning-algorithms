package ru.machine.learning.algorithms.utils;

import io.vavr.Tuple;
import io.vavr.Tuple4;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.selection.BitmapBackedSelection;
import tech.tablesaw.selection.Selection;
import tech.tablesaw.table.Rows;

import java.util.BitSet;
import java.util.Random;

import static java.lang.String.format;

public class TrainTestSplit {

    private static Random random = new Random(3);

    protected static void setSeed(long seed) {
        random = new Random(seed);
    }

    // train, test, train labels, test labels
    protected static Tuple4<Table, Table, DoubleColumn, DoubleColumn> split(Table table, String targetName, double alpha) {
        if (0 < alpha && alpha < 1) {
            var splitted = sampleSplit(table, alpha);
            var trainTarget = splitted[0].doubleColumn(targetName);
            var testTarget = splitted[1].doubleColumn(targetName);
            return Tuple.of(
                Table.create(splitted[0].removeColumns(targetName).columns()),
                Table.create(splitted[1].removeColumns(targetName).columns()),
                trainTarget,
                testTarget);
        }
        throw new RuntimeException(format("Wrong alpha %s\n", alpha));
    }

    private static Table[] sampleSplit(Table table, double table1Proportion) {
        var tables = new Table[2];
        var table1Count = (int) Math.round(table.rowCount() * table1Proportion);

        var table2Selection = new BitmapBackedSelection();
        for (var i = 0; i < table.rowCount(); i++) {
            table2Selection.add(i);
        }
        var table1Selection = new BitmapBackedSelection();

        var table1Records = selectNRowsAtRandom(table1Count, table.rowCount());
        for (var table1Record : table1Records) {
            table1Selection.add(table1Record);
        }
        table2Selection.andNot(table1Selection);
        tables[0] = where(table, table1Selection);
        tables[1] = where(table, table2Selection);
        return tables;
    }

    public static Table where(Table table, Selection selection) {
        var newTable = table.emptyCopy(selection.size());
        Rows.copyRowsToTable(selection, table, newTable);
        return newTable;
    }

    private static Selection selectNRowsAtRandom(int n, int max) {
        var selection = new BitmapBackedSelection();
        if (n > max) {
            throw new IllegalArgumentException(
                "Illegal arguments: N (" + n + ") greater than Max (" + max + ")");
        }

        var rows = new int[n];
        if (n == max) {
            for (var k = 0; k < n; ++k) {
                selection.add(k);
            }
            return selection;
        }

        var bs = new BitSet(max);
        var cardinality = 0;
        while (cardinality < n) {
            int v = random.nextInt(max);
            if (!bs.get(v)) {
                bs.set(v);
                cardinality++;
            }
        }
        int pos = 0;
        for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) {
            rows[pos++] = i;
        }
        for (int row : rows) {
            selection.add(row);
        }
        return selection;
    }
}
