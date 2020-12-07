package ru.machine.learning.algorithms.utils;

import tech.tablesaw.api.Table;

import java.io.IOException;
import java.io.InputStream;

public class Pipeline {

    public static Preprocessing over(Table table) {
        return Preprocessing.over(table);
    }

    public static Preprocessing loadFrom(InputStream is) throws IOException {
        var table = Table.read().csv(is);
        return Preprocessing.over(table);
    }
}
