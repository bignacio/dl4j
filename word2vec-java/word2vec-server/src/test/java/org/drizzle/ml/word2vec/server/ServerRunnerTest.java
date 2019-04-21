package org.drizzle.ml.word2vec.server;

import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import org.drizzle.ml.word2vec.client.Word2VecClient;
import org.drizzle.ml.word2vec.models.Nd4jVectorWordList;
import org.drizzle.ml.word2vec.models.Nd4jWordVector;
import org.drizzle.ml.word2vec.test.WordTestUtils;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ServerRunnerTest {

    private static final String IN_PROCESS_SERVICE_NAME = "inprocess-w2v-service";
    private static final long MAX_CONNECT_TIME_MS = 10000L;
    private static final long ASYNC_TIMEOUT_MS = 1000L;
    private final List<String> testWords = List.of("enterprise", "public", "poet");
    private final Map<String, DoubleArrayList> testWordMap = WordTestUtils.buildTestWordMap(testWords);

    private static ServerRunner serverRunner;
    private static Word2VecClient client;

    @BeforeAll
    public static void setup() throws URISyntaxException, IOException, InterruptedException {
        File modelFile = new File(ServerRunnerTest.class.getResource("/w2vmodel.bin").toURI());
        serverRunner = new ServerRunner(InProcessServerBuilder.forName(IN_PROCESS_SERVICE_NAME),
                modelFile,
                false);
        serverRunner.start();

        client = new Word2VecClient(InProcessChannelBuilder.forName(IN_PROCESS_SERVICE_NAME), ASYNC_TIMEOUT_MS);

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

    @Test
    public void getVectorMap() throws InterruptedException {
        List<Nd4jWordVector> wordVecMap = client.getVectorMap(testWords);

        Set<String> actual = wordVecMap.stream().map(Nd4jWordVector::getWord).collect(Collectors.toSet());
        assertEquals(testWordMap.keySet(), actual);
    }

    @Test
    public void getNearestWords() throws InterruptedException {
        List<INDArray> vectorList = testWordMap.values().stream()
                .map(Nd4j::create)
                .collect(Collectors.toList());

        List<Nd4jVectorWordList> nearestWords = client.getNearestWords(vectorList, 1);
        Set<String> actualWords = nearestWords.stream()
                .flatMap(wordList -> wordList.getWords().stream())
                .collect(Collectors.toSet());

        assertEquals(Set.copyOf(testWords), actualWords);
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
