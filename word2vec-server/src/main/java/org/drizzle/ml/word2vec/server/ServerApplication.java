package org.drizzle.ml.word2vec.server;

import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;

import java.io.File;
import java.util.concurrent.Callable;

@CommandLine.Command(description = "Word2Vec server", name = "w2v-server")
public class ServerApplication implements Callable<ServerApplication> {
    private static final Logger logger = LoggerFactory.getLogger(ServerApplication.class);

    @CommandLine.Option(names = {"--help"}, description = "Show help.")
    private boolean showHelp;

    @CommandLine.Option(names = {"-p", "--port"}, description = "Service port. Default 8080.")
    private int port = 8080;

    @CommandLine.Option(names = {"-f", "--file"}, required = true,
            description = "Path to google vector file. See https://code.google.com/archive/p/word2vec/ for more information.")
    private File word2vecModelFile;

    @CommandLine.Option(names = {"-t", "--tree"}, description = "Use tree model for faster lookup. Uses more memory. Default false.")
    private boolean useTreeModel = false;

    public static void main(String[] args) {
        CommandLine.call(new ServerApplication(), args);
    }

    @Override
    public ServerApplication call() throws Exception {
        if (showHelp) {
            CommandLine.usage(this, System.out);
            return this;
        }

        ServerRunner runner = new ServerRunner(
                NettyServerBuilder.forPort(port),
                word2vecModelFile,
                useTreeModel
        );
        runner.start();
        logger.info("Server runner stopped {}", runner);
        return this;
    }
}