package org.drizzle.ml.word2vec.handlers;

import io.grpc.stub.StreamObserver;
import org.drizzle.ml.word2vec.service.VectorList;
import org.drizzle.ml.word2vec.service.Word2VecServiceGrpc;
import org.drizzle.ml.word2vec.service.WordList;
import org.drizzle.ml.word2vec.service.WordVectorMap;

public class ServiceRequestHandler extends Word2VecServiceGrpc.Word2VecServiceImplBase {

    public ServiceRequestHandler() {
        loadModel();
    }

    @Override
    public StreamObserver<WordList> getVectorMap(StreamObserver<WordVectorMap> responseObserver) {
        throw new UnsupportedOperationException();
    }

    @Override
    public StreamObserver<VectorList> getNearestWord(StreamObserver<WordVectorMap> responseObserver) {
        throw new UnsupportedOperationException();
    }


    private void loadModel() {

    }
}
