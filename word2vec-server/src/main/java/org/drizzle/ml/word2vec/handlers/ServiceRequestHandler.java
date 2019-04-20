package org.drizzle.ml.word2vec.handlers;

import io.grpc.stub.StreamObserver;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.impl.TreeModelUtils;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.drizzle.ml.word2vec.service.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

public class ServiceRequestHandler extends Word2VecServiceGrpc.Word2VecServiceImplBase {
    private static final Logger logger = LoggerFactory.getLogger(ServiceRequestHandler.class);

    private final AtomicBoolean modelLoaded = new AtomicBoolean(false);
    private Word2Vec model;

    public ServiceRequestHandler(File word2vecModelFile, boolean useTreeModel) {
        loadModel(word2vecModelFile, useTreeModel);
    }

    @Override
    public StreamObserver<Word> getVectorMap(StreamObserver<WordVector> responseObserver) {
        return new StreamObserver<Word>() {
            @Override
            public void onNext(Word word) {
                if (modelLoaded.get()) {
                    double[] vector = model.getWordVector(word.getWord());

                    WordVector.Builder wordVectorBuilder = WordVector.newBuilder();
                    if (vector != null) {
                        wordVectorBuilder.setWord(word).addAllVector(new DoubleArrayList(vector));
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
        };
    }

    @Override
    public StreamObserver<NearestToVector> getNearestWord(StreamObserver<VectorWordList> responseObserver) {
        return new StreamObserver<NearestToVector>() {
            @Override
            public void onNext(NearestToVector nearestToVector) {
                if (modelLoaded.get()) {
                    List<Word> words = model.wordsNearest(toINDArray(nearestToVector.getVectorList()), nearestToVector.getLimit())
                            .stream()
                            .map(word -> Word.newBuilder().setWord(word).build())
                            .collect(Collectors.toList());

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
        };
    }

    private INDArray toINDArray(List<Double> vectorList) {
        return Nd4j.create(vectorList);
    }

    @Override
    public void getStatus(VoidMessage request, StreamObserver<Status> responseObserver) {
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

    private class ServiceTreeModelUtils extends TreeModelUtils {
        void buildTree() {
            checkTree();
        }

    }
}
