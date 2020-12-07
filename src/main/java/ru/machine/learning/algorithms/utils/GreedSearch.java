package ru.machine.learning.algorithms.utils;

import io.vavr.collection.List;
import io.vavr.collection.Map;
import io.vavr.collection.Seq;
import ru.machine.learning.algorithms.model.Model;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.Table;

import javax.annotation.Nonnull;

public class GreedSearch implements Model {

    private final ModelType modelType;
    private final Map<String, List<Object>> params;

    private Table train;
    private DoubleColumn trainTarget;

    public GreedSearch(ModelType modelType, Map<String, List<Object>> params) {
        this.modelType = modelType;
        this.params = params;
    }

    @Override
    public Model fit(@Nonnull Table train, @Nonnull DoubleColumn trainTarget) {
        this.train = train;
        this.trainTarget = trainTarget;
        return this;
    }

    @Override
    public Seq<Double> predict(@Nonnull Table test) {
        /*var allParams = params
            .toMap(Tuple2::_1, t -> t._2.map(o -> Tuple.of(t._1, o)));
        Set<Map<String, Object>> all;
        allParams.forEach(
            t -> {
                var key = t._1;
                var list = t._2;
                all.add(
                    allParams.
                )
            }
        );*/
        return null;
    }

    enum ModelType {
        KNN,
        GaussianNB
    }
}
