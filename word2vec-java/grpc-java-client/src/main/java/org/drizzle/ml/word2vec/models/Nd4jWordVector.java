package org.drizzle.ml.word2vec.models;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Objects;

/**
 * A Nd4j friendly representation of a word and its vector
 */
public class Nd4jWordVector {
    private final String word;
    private final INDArray vector;

    public Nd4jWordVector(String word, INDArray vector) {
        this.word = word;
        this.vector = vector;
    }

    public String getWord() {
        return word;
    }

    public INDArray getVector() {
        return vector;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Nd4jWordVector that = (Nd4jWordVector) o;
        return Objects.equals(word, that.word) &&
                Objects.equals(vector, that.vector);
    }

    @Override
    public int hashCode() {
        return Objects.hash(word, vector);
    }

    @Override
    public String toString() {
        return "Nd4jWordVector{" +
                "word='" + word + '\'' +
                ", vector=" + vector +
                '}';
    }
}
