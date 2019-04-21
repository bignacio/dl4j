package org.drizzle.ml.word2vec.server;

import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import org.drizzle.ml.word2vec.client.Word2VecClient;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ServerRunnerTest {

    private static final String IN_PROCESS_SERVICE_NAME = "inprocess-w2v-service";
    private static final long MAX_CONNECT_TIME_MS = 10000L;
    private static ServerRunner serverRunner;
    private static Word2VecClient client;

    @BeforeAll
    public static void setup() throws URISyntaxException, IOException, InterruptedException {
        File modelFile = new File(ServerRunnerTest.class.getResource("/w2vmodel.bin").toURI());
        serverRunner = new ServerRunner(InProcessServerBuilder.forName(IN_PROCESS_SERVICE_NAME),
                modelFile,
                false);
        serverRunner.start();

        client = new Word2VecClient(InProcessChannelBuilder.forName(IN_PROCESS_SERVICE_NAME));

        waitForReady();
    }

    @AfterAll
    public static void tearDown() {
        serverRunner.stop();
    }

    @Test
    public void isReady() {
        assertTrue(client.isReady());
    }


    private static void waitForReady() throws InterruptedException {
        final long startTime = System.currentTimeMillis();
        boolean isServiceReady = client.isReady();
        while (System.currentTimeMillis() - startTime < MAX_CONNECT_TIME_MS && !isServiceReady) {
            isServiceReady = client.isReady();

            Thread.sleep(500L);
        }

        assertTrue(isServiceReady);
    }
}
