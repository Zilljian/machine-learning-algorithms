package ru.machine.learning.algorithms.model;

import io.vavr.Tuple2;
import io.vavr.collection.List;
import io.vavr.collection.List.Nil;
import io.vavr.collection.Map;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import java.util.ArrayList;
import java.util.function.Function;

public class Util {

    public static Map<String, List<Double>> toListOnColumn(Table train) {
        return List.ofAll(train.columns())
            .map(c -> List.ofAll(((DoubleColumn) c).asList()))
            .map(List::ofAll)
            .zip(List.ofAll(train.columnNames()))
            .toMap(Tuple2::_2, Tuple2::_1);
    }

    public static List<List<Double>> toList(Table table) {
        List<List<Double>> result = Nil.instance();
        for (var e : table) {
            result = result.prepend(List.ofAll(e.columnNames()).map(e::getNumber));
        }
        return result;
    }

    public static Map<String, List<Double>> transformRowsToCols(List<Tuple2<List<Double>, Double>> rows, List<String> colNames) {
        var colNum = colNames.size();
        var cols = new ArrayList<List<Double>>(colNum);
        for (var r : rows.map(Tuple2::_1)) {
            for (var i : List.range(0, colNum)) {
                if (i >= cols.size()) {
                    cols.add(List.of(r.get(i)));
                } else {
                    cols.set(i, cols.get(i).append(r.get(i)));
                }
            }
        }
        return colNames.zip(cols).toMap(Function.identity());
    }

    public static double[] listToArray(List<Double> list) {
        var array = new double[list.size()];
        list.zipWithIndex()
            .forEach(t -> array[t._2] = t._1);
        return array;
    }
}
