package ru.k.nearest.neighbors;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.checkerframework.checker.nullness.qual.NonNull;
import scala.Tuple2;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import javax.annotation.Nonnull;

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toList;

public class Knn {

    private static final String DELIMITER = ",";

    private final SparkSession session;
    private final int k;

    private Dataset<Row> train;

    public Knn(SparkSession session, int k) {
        this.k = k;
        this.session = session;
    }

    public Knn fit(@Nonnull Dataset<Row> train) {
        this.train = train;
        return this;
    }

    public Map<Integer, ?> predict(Dataset<Row> test) {
        var ctx = JavaSparkContext.fromSparkContext(session.sparkContext());
        var k = ctx.broadcast(this.k);
        return test.javaRDD()
            .cartesian(train.javaRDD())
            .mapToPair(t -> {
                var testStr = t._1.toString();
                var trainStr = t._2.toString();
                var id = Objects.hash(testStr);
                var label = label(trainStr);
                var distance = calculateDistance(testStr, trainStr);
                return new Tuple2<>(id, new Tuple2<>(distance, label));
            })
            .groupByKey()
            .mapValues(n -> resolveLabel(n, k.getValue()))
            .collectAsMap();
    }

    static String label(String s) {
        return s.substring(1, s.indexOf(DELIMITER)).trim();
    }

    static <T> T resolveLabel(Iterable<Tuple2<Double, T>> neighbors, int K) {
        return StreamSupport.stream(neighbors.spliterator(), false)
            .sorted(comparing(Tuple2::_1))
            .limit(K)
            .collect(groupingBy(Tuple2::_2, counting()))
            .entrySet()
            .stream()
            .max(Entry.comparingByValue())
            .orElseThrow()
            .getKey();
    }

    static double calculateDistance(@NonNull String testRow, @NonNull String trainRow) {
        var test = asDoubleList(testRow);
        trainRow = trainRow.substring(0, trainRow.lastIndexOf(","));
        var train = asDoubleList(trainRow);
        return Math.sqrt(
            IntStream.range(0, Math.min(test.size(), train.size()))
                .mapToObj(i -> Math.pow(test.get(i) - train.get(i), 2))
                .reduce(Double::sum)
                .orElseThrow()
        );
    }

    private static List<Double> asDoubleList(String string) {
        return Stream.of(
            string.replaceAll("[\\[\\]]", "")
                .split(DELIMITER)
        )
            .map(String::trim)
            .map(Double::parseDouble)
            .collect(toList());
    }
}
