package org.drizzle.ml.word2vec.handlers;

import io.grpc.stub.StreamObserver;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.drizzle.ml.word2vec.service.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.net.URISyntaxException;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

public class ServiceRequestHandlerTest {
    private static final long MAX_WAIT_FOR_READY_TIME_MS = 10000L;
    private static File modelFile;
    private static Map<String, DoubleArrayList> wordMap;

    private final String[] testWords = {"fire", "rocks", "water"};

    @BeforeEach
    public void setup() throws URISyntaxException {
        modelFile = new File(getClass().getResource("/w2vmodel.bin").toURI());
        wordMap = buildTestWordMap();
    }

    @Test
    public void getVectorMapFlatModel() {
        verifyGetVectorMap(false);
    }

    @Test
    public void getVectorMapTreeModel() {
        verifyGetVectorMap(true);
    }

    @Test
    public void getNearestWordFlatModel() {
        verifyGetNearestWord(false);
    }

    @Test
    public void getNearestWordTreeModel() {
        verifyGetNearestWord(true);
    }

    private void verifyGetVectorMap(boolean useTreeModel) {
        ServiceRequestHandler handler = new ServiceRequestHandler(modelFile, useTreeModel);
        waitForReady(handler);

        WordVectorResponseObserver callObserver = new WordVectorResponseObserver();

        StreamObserver<Word> wordList = handler.getVectorMap(callObserver);


        for (String word : testWords) {
            wordList.onNext(Word.newBuilder().setWord(word).build());
        }
        wordList.onCompleted();

        Set<WordVector> actual = new HashSet<>(callObserver.getWordVecs());
        Set<WordVector> expected = wordMap.entrySet().stream()
                .map(entry -> WordVector.newBuilder()
                        .setWord(Word.newBuilder().setWord(entry.getKey()).build())
                        .addAllVector(entry.getValue())
                        .build()
                ).collect(Collectors.toSet());

        assertEquals(expected, actual);
    }

    private void verifyGetNearestWord(boolean useTreeModel) {
        ServiceRequestHandler handler = new ServiceRequestHandler(modelFile, useTreeModel);
        waitForReady(handler);

        VectorWordListResponseObserver callObserver = new VectorWordListResponseObserver();

        StreamObserver<NearestToVector> nearestWords = handler.getNearestWord(callObserver);
        for (DoubleArrayList vector : wordMap.values()) {
            nearestWords.onNext(NearestToVector.newBuilder()
                    .addAllVector(vector)
                    .setLimit(1)
                    .build());
        }
        nearestWords.onCompleted();

        Set<String> actualWords = callObserver.getWordList()
                .stream()
                .flatMap(wordList -> wordList.getWordsList().stream())
                .map(Word::getWord)
                .collect(Collectors.toSet());

        assertEquals(wordMap.keySet(), actualWords);
    }


    private void waitForReady(ServiceRequestHandler handler) {
        final long startTime = System.currentTimeMillis();

        StatusResponse response = new StatusResponse();

        while (System.currentTimeMillis() - startTime < MAX_WAIT_FOR_READY_TIME_MS && !response.isReady()) {
            quietlySleep(100L);
            handler.getStatus(VoidMessage.newBuilder().build(), response);
        }

        assertTrue(response.isReady());
    }

    private void quietlySleep(long sleepTime) {
        try {
            Thread.sleep(sleepTime);
        } catch (InterruptedException e) {

        }
    }


    private void train() throws URISyntaxException {
        File testFile = new File(getClass().getResource("/training/sample.txt").toURI());
        DefaultTokenizerFactory tokenizer = new DefaultTokenizerFactory();
        tokenizer.setTokenPreProcessor(new CommonPreprocessor());
        SentenceIterator iterator = new FileSentenceIterator(testFile);
        Word2Vec model = new Word2Vec.Builder()
                .epochs(20)
                .workers(2)
                .iterate(iterator)
                .tokenizerFactory(tokenizer)
                .build();

        model.fit();
        WordVectorSerializer.writeWord2VecModel(model, "w2vmodel.bin");
    }

    private void loadModel() throws URISyntaxException {
        WordVectors model = WordVectorSerializer.readWord2VecModel(modelFile, true);

        double[] vector = model.getWordVector("water");
        Collection<String> similars = model.wordsNearest("water", 2);
        assertNotNull(vector);
    }

    private Map<String, DoubleArrayList> buildTestWordMap() {
        WordVectors model = WordVectorSerializer.readWord2VecModel(modelFile, true);

        Map<String, DoubleArrayList> result = new HashMap<>();

        for (String word : testWords) {
            DoubleArrayList vector = new DoubleArrayList(model.getWordVector(word));
            result.put(word, vector);
        }

        return result;
    }

    /**
     *
     */
    class WordVectorResponseObserver implements StreamObserver<WordVector> {
        private final List<WordVector> wordVecs = new ArrayList<>();

        @Override
        public void onNext(WordVector wordVector) {
            wordVecs.add(wordVector);
        }

        @Override
        public void onError(Throwable throwable) {
            fail();
        }

        @Override
        public void onCompleted() {

        }

        public List<WordVector> getWordVecs() {
            return wordVecs;
        }
    }

    /**
     *
     */
    class StatusResponse implements StreamObserver<Status> {

        private boolean isReady = false;

        @Override
        public void onNext(Status status) {
            this.isReady = status.getReady();
        }

        @Override
        public void onError(Throwable throwable) {

        }

        @Override
        public void onCompleted() {

        }

        public boolean isReady() {
            return isReady;
        }
    }

    class VectorWordListResponseObserver implements StreamObserver<VectorWordList> {
        private final List<VectorWordList> wordList = new ArrayList<>();

        @Override
        public void onNext(VectorWordList vectorWordList) {
            wordList.add(vectorWordList);
        }

        @Override
        public void onError(Throwable throwable) {
            fail();
        }

        @Override
        public void onCompleted() {

        }

        public List<VectorWordList> getWordList() {
            return wordList;
        }
    }

}
