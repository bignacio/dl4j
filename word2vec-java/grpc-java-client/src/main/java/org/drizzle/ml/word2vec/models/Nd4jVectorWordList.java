package org.drizzle.ml.word2vec.models;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Objects;

/**
 * Represents words that match or are near a vector
 */
public class Nd4jVectorWordList {
    private final INDArray vector;
    private final List<String> words;


    public Nd4jVectorWordList(INDArray vector, List<String> words) {
        this.vector = vector;
        this.words = words;
    }

    public INDArray getVector() {
        return vector;
    }

    public List<String> getWords() {
        return words;
    }

    @Override
    public String toString() {
        return "Nd4jVectorWordList{" +
                "vector=" + vector +
                ", words=" + words +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Nd4jVectorWordList that = (Nd4jVectorWordList) o;
        return Objects.equals(vector, that.vector) &&
                Objects.equals(words, that.words);
    }

    @Override
    public int hashCode() {
        return Objects.hash(vector, words);
    }
}
