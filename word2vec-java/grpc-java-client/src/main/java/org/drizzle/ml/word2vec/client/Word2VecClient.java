package org.drizzle.ml.word2vec.client;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;
import org.drizzle.ml.word2vec.models.Nd4jVectorWordList;
import org.drizzle.ml.word2vec.models.Nd4jWordVector;
import org.drizzle.ml.word2vec.service.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class Word2VecClient {
    private static final Logger logger = LoggerFactory.getLogger(Word2VecClient.class);

    private final Word2VecServiceGrpc.Word2VecServiceStub asyncStub;
    private final Word2VecServiceGrpc.Word2VecServiceBlockingStub blockingStub;
    private final long asyncTimeoutMs;

    /**
     * Creates a socket based client, connecting to a specific host and port
     *
     * @param host           address of the Word2Vec service
     * @param port           connection port
     * @param asyncTimeoutMs timeout for async calls in milliseconds
     */
    public Word2VecClient(String host, int port, long asyncTimeoutMs) {
        this(ManagedChannelBuilder.forAddress(host, port), asyncTimeoutMs);

    }

    /**
     * Creates a client with a custom channel
     *
     * @param channelBuilder channel builder
     * @param asyncTimeoutMs timeout for async calls in milliseconds
     */
    public Word2VecClient(ManagedChannelBuilder<?> channelBuilder, long asyncTimeoutMs) {
        ManagedChannel channel = channelBuilder.build();
        asyncStub = Word2VecServiceGrpc.newStub(channel);
        blockingStub = Word2VecServiceGrpc.newBlockingStub(channel);
        this.asyncTimeoutMs = asyncTimeoutMs;
    }

    /**
     * Checks the service status and returns true if it is ready to
     * process requests
     *
     * @return true if ready to process requests
     */
    public boolean isReady() {
        return blockingStub.getStatus(VoidMessage.newBuilder().build())
                .getReady();
    }

    public List<Nd4jWordVector> getVectorMap(List<String> words) throws InterruptedException {
        var responseObserver = new WordVectorResponseObserver();
        StreamObserver<Word> requestObserver = asyncStub.getVectorMap(responseObserver);

        words.forEach(word -> requestObserver.onNext(Word.newBuilder().setWord(word).build()));
        requestObserver.onCompleted();

        List<Nd4jWordVector> results = new ArrayList<>();

        for (WordVector wordVector : responseObserver.getWordVectors()) {
            results.add(
                    new Nd4jWordVector(wordVector.getWord().getWord(),
                            Nd4j.create(wordVector.getVectorList()))
            );
        }


        return results;
    }

    public List<Nd4jVectorWordList> getNearestWords(List<INDArray> vectors, int limit) throws InterruptedException {
        var responseObserver = new VectorWordListResponseObserver();
        StreamObserver<NearestToVector> requestObserver = asyncStub.getNearestWords(responseObserver);

        for (INDArray vector : vectors) {
            NearestToVector.Builder builder = NearestToVector.newBuilder()
                    .setLimit(limit);

            for (long i = 0; i < vector.length(); i++) {
                builder.addVector(vector.getDouble(i));
            }

            requestObserver.onNext(builder.build());
        }

        requestObserver.onCompleted();

        List<Nd4jVectorWordList> results = new ArrayList<>();
        for (VectorWordList vectorWord : responseObserver.getVectorWords()) {
            List<String> words = new ArrayList<>();
            vectorWord.getWordsList().forEach(requestWord -> words.add(requestWord.getWord()));

            results.add(
                    new Nd4jVectorWordList(
                            Nd4j.create(vectorWord.getVectorList()),
                            words
                    )
            );
        }
        return results;
    }

    /**
     * Inner classes
     */

    /**
     * Processes responses to WordVector requests
     */
    private class WordVectorResponseObserver implements StreamObserver<WordVector> {
        private final List<WordVector> wordVectors = new ArrayList<>();
        private final CountDownLatch countLatch = new CountDownLatch(1);

        @Override
        public void onNext(WordVector wordVector) {
            logger.trace("Received word vector {}", wordVector);
            wordVectors.add(wordVector);
        }

        @Override
        public void onError(Throwable throwable) {
            logger.error("Error processing getVectorMap response", throwable);
            countLatch.countDown();
        }

        @Override
        public void onCompleted() {
            countLatch.countDown();
        }

        List<WordVector> getWordVectors() throws InterruptedException {
            if (countLatch.await(asyncTimeoutMs, TimeUnit.MILLISECONDS)) {
                return wordVectors;
            }
            return List.of();
        }
    }

    /**
     * Processes responses for VectorWordList requests
     */
    private class VectorWordListResponseObserver implements StreamObserver<VectorWordList> {
        private final List<VectorWordList> vectorWords = new ArrayList<>();
        private final CountDownLatch countLatch = new CountDownLatch(1);

        @Override
        public void onNext(VectorWordList vectorWordList) {
            logger.trace("Received vector word list {}", vectorWordList);
            vectorWords.add(vectorWordList);
        }

        @Override
        public void onError(Throwable throwable) {
            logger.error("Error processing getNearestWords response", throwable);
            countLatch.countDown();
        }

        @Override
        public void onCompleted() {
            countLatch.countDown();
        }

        public List<VectorWordList> getVectorWords() throws InterruptedException {
            if (countLatch.await(asyncTimeoutMs, TimeUnit.MILLISECONDS)) {
                return vectorWords;
            }
            return List.of();
        }
    }
}
