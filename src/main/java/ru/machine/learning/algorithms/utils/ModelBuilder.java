package ru.machine.learning.algorithms.utils;

import io.vavr.collection.Map;
import ru.machine.learning.algorithms.model.Model;
import ru.machine.learning.algorithms.model.bayes.GaussianNaiveBayes;
import ru.machine.learning.algorithms.model.knn.Knn;

import javax.annotation.Nonnull;

import static java.lang.String.format;

public class ModelBuilder {

    private Map<String, Object> params;

    public ModelBuilder withParams(@Nonnull Map<String, Object> params) {
        this.params = params;
        return this;
    }

    public ModelBuilder addParam(String key, Object value) {
        this.params = params.put(key, value);
        return this;
    }

    public Model buildExactly(Class<? extends Model> clazz) {
        if (clazz == Knn.class) {
            return Knn.withParams(params);
        }
        if (clazz == GaussianNaiveBayes.class) {
            return GaussianNaiveBayes.withParams(params);
        }
        throw new IllegalArgumentException(format("Model with class %s doesn't exist", clazz.getCanonicalName()));
    }
}
