package org.drizzle.ml.word2vec.server;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import org.drizzle.ml.word2vec.handlers.ServiceRequestHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class ServerRunner {
    private final Logger logger = LoggerFactory.getLogger(ServerRunner.class);
    private final Server server;

    public ServerRunner(ServerBuilder<?> serverBuilder, File word2vecModelFile, boolean useTreeModel) {
        server = serverBuilder.addService(new ServiceRequestHandler(word2vecModelFile, useTreeModel))
                .build();

        logger.info("Created server {}", server);
    }

    public void start() throws IOException {
        server.start();
        logger.info("Server started on port {}", server.getPort());

        Runtime.getRuntime().addShutdownHook(shutdownOnStopHook());
    }

    public void awaitTermination() throws InterruptedException {
        server.awaitTermination();
    }

    private Thread shutdownOnStopHook() {
        return new Thread() {
            @Override
            public void run() {
                logger.warn("Stopping server on JVM shutdown");
                ServerRunner.this.stop();
                logger.warn("Server stopped");
            }
        };
    }

    public void stop() {
        logger.info("Shutting down server");
        server.shutdown();
        logger.info("Server shutdown");
    }

    @Override
    public String toString() {
        return "ServerRunner{" +
                "server=" + server +
                '}';
    }
}