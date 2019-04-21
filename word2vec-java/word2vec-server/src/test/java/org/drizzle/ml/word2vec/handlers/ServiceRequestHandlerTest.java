package org.drizzle.ml.word2vec.handlers;

import io.grpc.stub.StreamObserver;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import org.drizzle.ml.word2vec.service.*;
import org.drizzle.ml.word2vec.test.WordTestUtils;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

public class ServiceRequestHandlerTest {
    private static final long MAX_WAIT_FOR_READY_TIME_MS = 10000L;

    private final List<String> testWords = List.of("fire", "rocks", "water");
    private final Map<String, DoubleArrayList> wordMap = WordTestUtils.buildTestWordMap(testWords);


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

    @Test
    public void testStatus() {
        ServiceRequestHandler handler = new ServiceRequestHandler(WordTestUtils.modelFile, false);
        assertWaitForReady(handler);
    }

    private void verifyGetVectorMap(boolean useTreeModel) {
        ServiceRequestHandler handler = new ServiceRequestHandler(WordTestUtils.modelFile, useTreeModel);
        assertWaitForReady(handler);

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
        ServiceRequestHandler handler = new ServiceRequestHandler(WordTestUtils.modelFile, useTreeModel);
        assertWaitForReady(handler);

        VectorWordListResponseObserver callObserver = new VectorWordListResponseObserver();

        StreamObserver<NearestToVector> nearestWords = handler.getNearestWords(callObserver);
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


    private void assertWaitForReady(ServiceRequestHandler handler) {
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
