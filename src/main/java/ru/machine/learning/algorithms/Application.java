package ru.machine.learning.algorithms;

import io.vavr.collection.List;
import io.vavr.collection.Traversable;
import ru.machine.learning.algorithms.model.bayes.GaussianNaiveBayes;
import ru.machine.learning.algorithms.utils.Metrics;
import ru.machine.learning.algorithms.utils.Pipeline;

import java.io.IOException;
import java.io.InputStream;

public class Application {

    private static final InputStream DATA = Thread.currentThread().getContextClassLoader().getResourceAsStream("train.csv");

    public static void main(String[] args) throws IOException {
        var splitted = Pipeline
            .loadFrom(DATA)
            .peek(t -> System.out.printf("Init table shape: %s\n\n", t.shape()))
            .dropCols("Cabin", "Name", "PassengerId", "Ticket", "SibSp", "Parch")
            .dropNa()
            .encodeExactly("Sex", "Embarked")
            .castToDoubleAndNormalize()
            .peek(t -> System.out.printf("Table shape after preprocessing: %s\n\n", t.shape()))
            .trainTestSplitWith("Survived", 0.8)
            .peekSplitted(Metrics::describeData)
            .splitted();

        for (var n : List.range(1, 2)) {
            System.out.printf("\n\nN = %d\n\n", n);
            var predicted = Metrics.withTime(
                () -> new GaussianNaiveBayes()
                    .fit(splitted._1, splitted._3)
                    .predict(splitted._2)
            );

            var merge =
                List.ofAll(splitted._4.asList()).zip(predicted)
                    .groupBy(t -> {
                        if (t._1.equals(t._2()) && t._1.equals(0d)) {
                            return "tn";
                        } else if (t._1.equals(t._2) && t._1.equals(1d)) {
                            return "tp";
                        } else if (!t._1.equals(t._2) && t._1.equals(0d)) {
                            return "fp";
                        } else {
                            return "fn";
                        }
                    })
                    .mapValues(Traversable::size);

            Metrics.printClassificationMetrics(merge);
        }
    }
}
