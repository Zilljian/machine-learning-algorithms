package ru.machine.learning.algorithms;

import io.vavr.collection.List;
import ru.machine.learning.algorithms.model.bayes.GaussianNaiveBayes;
import ru.machine.learning.algorithms.model.knn.Knn;
import ru.machine.learning.algorithms.model.knn.Metric;
import ru.machine.learning.algorithms.model.tree.DecisionTreeClassifier;
import ru.machine.learning.algorithms.model.tree.metric.GiniIndex;
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

        System.out.println("\nKnn with n = 5:\n");
        var predicted = Metrics.withTime(
            () -> new Knn(10)
                .withMetric(Metric.Chebyshev)
                .fit(splitted._1, splitted._3)
                .predict(splitted._2)
        );
        Metrics.printBinaryClassificationMetrics(List.ofAll(splitted._4.asList()), predicted);

        System.out.println("\nDecision tree:\n");
        predicted = Metrics.withTime(
            () -> new DecisionTreeClassifier()
                .withSplitMetric(new GiniIndex().setMinEntries(22))
                .fit(splitted._1, splitted._3)
                .predict(splitted._2)
        );
        Metrics.printBinaryClassificationMetrics(List.ofAll(splitted._4.asList()), predicted);

        System.out.println("\nGaussian Naive Bayes:\n");
        predicted = Metrics.withTime(
            () -> new GaussianNaiveBayes()
                .fit(splitted._1, splitted._3)
                .predict(splitted._2)
        );
        Metrics.printBinaryClassificationMetrics(List.ofAll(splitted._4.asList()), predicted);
    }
}
