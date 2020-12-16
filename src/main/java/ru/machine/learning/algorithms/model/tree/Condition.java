package ru.machine.learning.algorithms.model.tree;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
class Condition {

    private Integer featureIndex;
    private String featureName;
    private Double value;
    private Operation operation;

    enum Operation {
        EQUAL, GREATER
    }
}
