package org.drizzle.ml.word2vec.client;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.drizzle.ml.word2vec.service.VectorWordList;
import org.drizzle.ml.word2vec.service.VoidMessage;
import org.drizzle.ml.word2vec.service.Word2VecServiceGrpc;
import org.drizzle.ml.word2vec.service.WordVector;

import java.util.List;

public class Word2VecClient {
    private final Word2VecServiceGrpc.Word2VecServiceStub asyncStub;
    private final Word2VecServiceGrpc.Word2VecServiceBlockingStub blockingStub;
    private final ManagedChannel channel;

    public Word2VecClient(String host, int port) {
        this(ManagedChannelBuilder.forAddress(host, port));
    }

    public Word2VecClient(ManagedChannelBuilder<?> channelBuilder) {
        channel = channelBuilder.build();
        asyncStub = Word2VecServiceGrpc.newStub(channel);
        blockingStub = Word2VecServiceGrpc.newBlockingStub(channel);
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

    public List<WordVector> getVectorMap(List<String> words) {
        throw new UnsupportedOperationException();
    }

    public List<VectorWordList> getNearestWords() {
        throw new UnsupportedOperationException();
    }
}
