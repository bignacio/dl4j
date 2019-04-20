package org.drizzle.ml.word2vec.handlers;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.net.URISyntaxException;

import static org.junit.jupiter.api.Assertions.assertNotNull;

public class RequestHandlerTest {

    @Test
    public void train() throws URISyntaxException {
        File testFile = new File(getClass().getResource("/training/faust.txt").toURI());
        DefaultTokenizerFactory tokenizer = new DefaultTokenizerFactory();
        tokenizer.setTokenPreProcessor(new CommonPreprocessor());
        SentenceIterator iterator = new FileSentenceIterator(testFile);
        Word2Vec model = new Word2Vec.Builder()
                .epochs(10)
                .workers(2)
                .iterate(iterator)
                .tokenizerFactory(tokenizer)
                .build();

        model.fit();
    }

    @Test
    public void loadModel() throws URISyntaxException {
        File modelFile = new File(getClass().getResource("/faust.bin").toURI());
        WordVectors model = WordVectorSerializer.loadStaticModel(modelFile);

        double[] vector = model.getWordVector("sun");
        assertNotNull(vector);
    }

}
