package org.drizzle.ml.word2vec.handlers;

import io.grpc.stub.StreamObserver;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.impl.TreeModelUtils;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.drizzle.ml.word2vec.service.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class ServiceRequestHandler extends Word2VecServiceGrpc.Word2VecServiceImplBase {
    private static final Logger logger = LoggerFactory.getLogger(ServiceRequestHandler.class);

    private final AtomicBoolean modelLoaded = new AtomicBoolean(false);
    private Word2Vec model;

    public ServiceRequestHandler(File word2vecModelFile, boolean useTreeModel) {
        loadModel(word2vecModelFile, useTreeModel);
    }

    @Override
    public StreamObserver<Word> getVectorMap(StreamObserver<WordVector> responseObserver) {
        logger.debug("getVectorMap called, model loaded {}", modelLoaded);
        return new WordStreamObserver(responseObserver);
    }

    @Override
    public StreamObserver<NearestToVector> getNearestWords(StreamObserver<VectorWordList> responseObserver) {
        logger.debug("getNearestWord called, model loaded {}", modelLoaded);
        return new NearestToVectorStreamObserver(responseObserver);
    }

    @Override
    public void getStatus(VoidMessage request, StreamObserver<Status> responseObserver) {
        logger.debug("getStatus called, model loaded {}", modelLoaded);
        responseObserver.onNext(Status.newBuilder().setReady(modelLoaded.get()).build());
        responseObserver.onCompleted();
    }

    private void loadModel(File word2vecModelFile, boolean treeModel) {
        Thread loadingThread = new Thread(() -> {
            logger.info("Loading word2vec model file {}, use tree model {}", word2vecModelFile, treeModel);
            this.model = WordVectorSerializer.readWord2VecModel(word2vecModelFile, true);

            logger.info("Model file '{}' loaded", word2vecModelFile);

            if (treeModel) {
                logger.info("Use tree model flag set, warming up model");

                ServiceTreeModelUtils modeUtils = new ServiceTreeModelUtils();
                model.setModelUtils(modeUtils);
                modeUtils.buildTree();
            }

            modelLoaded.set(true);
        }, "Word2Vec-model-loader");

        loadingThread.start();
    }

    private INDArray toINDArray(List<Double> vectorList) {
        return Nd4j.create(vectorList);
    }

    /**
     * internal classes
     */

    private class ServiceTreeModelUtils extends TreeModelUtils {
        void buildTree() {
            checkTree();
        }

    }

    /**
     * Observer for getVectorMap method
     */
    private class WordStreamObserver implements StreamObserver<Word> {
        private final StreamObserver<WordVector> responseObserver;

        WordStreamObserver(StreamObserver<WordVector> responseObserver) {
            this.responseObserver = responseObserver;
        }

        @Override
        public void onNext(Word word) {
            logger.trace("getVectorMap onNext called for word {}", word);
            if (modelLoaded.get()) {
                double[] vector = model.getWordVector(word.getWord());

                WordVector.Builder wordVectorBuilder = WordVector.newBuilder();
                if (vector != null) {
                    wordVectorBuilder.setWord(word);
                    for (double value : vector) {
                        wordVectorBuilder.addVector(value);
                    }
                }

                responseObserver.onNext(wordVectorBuilder.build());
            }
        }

        @Override
        public void onError(Throwable throwable) {
            logger.error("Error retrieving word vector", throwable);
        }

        @Override
        public void onCompleted() {
            responseObserver.onCompleted();
        }
    }

    /**
     * Observer for getNearestWord method
     */
    private class NearestToVectorStreamObserver implements StreamObserver<NearestToVector> {
        private final StreamObserver<VectorWordList> responseObserver;

        NearestToVectorStreamObserver(StreamObserver<VectorWordList> responseObserver) {
            this.responseObserver = responseObserver;
        }

        @Override
        public void onNext(NearestToVector nearestToVector) {
            logger.trace("getNearestWord onNext called with vector {}", nearestToVector);
            if (modelLoaded.get()) {
                Collection<String> nearest = model.wordsNearest(toINDArray(nearestToVector.getVectorList()), nearestToVector.getLimit());
                List<Word> words = new ArrayList<>();
                for (String nearWord : nearest) {
                    words.add(Word.newBuilder().setWord(nearWord).build());
                }

                VectorWordList vectorWordList = VectorWordList.newBuilder()
                        .addAllVector(nearestToVector.getVectorList())
                        .addAllWords(words)
                        .build();

                responseObserver.onNext(vectorWordList);
            }
        }

        @Override
        public void onError(Throwable throwable) {
            logger.error("Error getting nearest word", throwable);
        }

        @Override
        public void onCompleted() {
            responseObserver.onCompleted();
        }
    }

}
