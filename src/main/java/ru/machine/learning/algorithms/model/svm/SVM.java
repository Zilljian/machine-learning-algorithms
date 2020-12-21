package ru.machine.learning.algorithms.model.svm;

import io.vavr.collection.List;
import io.vavr.collection.List.Nil;
import org.nd4j.linalg.factory.Nd4j;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.NumericColumn;
import tech.tablesaw.api.Table;

import javax.annotation.Nonnull;

import static java.util.stream.Collectors.toList;

public class SVM {

    private int C = 1;
    private String kernel;
    private Double gamma;
    private Integer coef;

    SVM fit(@Nonnull Table train, @Nonnull DoubleColumn trainTarget) {
        var features = train.columnCount();
        var size = train.rowCount();
        this.gamma = gamma == null ? 1 / features : gamma;
        var data = toRowsList(train);

        // kernel init
        //compute each with each target and

        var d = train.columns().stream()
            .map(c -> (DoubleColumn) c)
            .map(NumericColumn::asDoubleArray)
            .collect(toList())
            .toArray(new double[][]{});

        var matrix = Nd4j.create(d);

        var arr1 = new double[features * features];
        data
            .flatMap(r1 -> data.map(r2 -> linearKernel(r1, r2)))
            .forEachWithIndex((v, i) -> arr1[i] = v);
        var target = List.ofAll(trainTarget.asList());
        var kernelMatrix = Nd4j.create(arr1, features, features);

        var arr = new double[size * size];
        target
            .flatMap(t1 -> target.map(t2 -> t2 * t1))
            .forEachWithIndex((v, i) -> arr[i] = v);
        var outer = Nd4j.create(arr, size, size);
        var P = outer.mul(kernelMatrix);
        return this;
    }

    private Double linearKernel(List<Double> row1, List<Double> row2) {
        return row1.zip(row2)
            .map(t -> t._1 * t._2)
            .reduce(Double::sum) + C;
    }

    private List<List<Double>> toRowsList(Table table) {
        List<List<Double>> result = Nil.instance();
        for (var e : table) {
            result = result.prepend(List.ofAll(e.columnNames()).map(e::getNumber));
        }
        return result;
    }
}
