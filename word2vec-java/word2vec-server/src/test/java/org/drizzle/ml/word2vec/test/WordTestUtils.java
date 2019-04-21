package org.drizzle.ml.word2vec.test;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class WordTestUtils {
    public static final File modelFile = new File(WordTestUtils.class.getResource("/w2vmodel.bin").getFile());

    private WordTestUtils() {

    }

    public static Map<String, DoubleArrayList> buildTestWordMap(Iterable<String> testWords) {
        WordVectors model = WordVectorSerializer.readWord2VecModel(modelFile, true);

        Map<String, DoubleArrayList> result = new HashMap<>();

        for (String word : testWords) {
            DoubleArrayList vector = new DoubleArrayList(model.getWordVector(word));
            result.put(word, vector);
        }

        return result;
    }

}
