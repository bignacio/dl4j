# Word2Vec server

Comprehensive vector representations of words such as [Google word2vec](https://code.google.com/archive/p/word2vec) can be quite large and
take a long time to load, making debugging and troubleshooting a little frustrating.

The Word2Vec Server is a simple gRPC based java server created to allow multiple clients to query a single Word2Vec model.

The service user the [deeplearning4j word2vec implementation](https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec).

## Building and running
Requirements
* JDK 11
* Maven 3.6 or newer

clone the repository and run
`mvn package`

This will create a shaded (fat) jar in `word2vec-server/target/word2vec-server-1.0.0.jar`. Copy it to your favourite location and it's ready to roll.

Run `java -jar word2vec-server-1.0.0.jar --help` for all command line parameters

``` bash
Usage: w2v-server [-t] [--help] -f=<word2vecModelFile> [-p=<port>]
Word2Vec server
      --help          Show help.
  -f, --file=<word2vecModelFile>
                      Path to google vector file. See https://code.google.
                        com/archive/p/word2vec/ for more information.
  -p, --port=<port>   Service port. Default 8080.
  -t, --tree          Use tree model for faster lookup. Uses more memory. Default
                        false.
```

By default, the service will listen on port `8080` but that can be changed through the parameter `-p` or `--port`.

The only mandatory parameter is the path to the word2vec model. You can use any embeddings model compatible with [Google word2vec](https://code.google.com/archive/p/word2vec).
(for example, the [Google news pre-trained model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM))

The `-t`, `--tree` flag makes the service use a tree based internal representation of the model, allowing faster searching for nearest words
at the cost of substantial memory and loading time.
This behaviour is disabled by default but recommended if you intend to have the service running constantly and have resources
available.

## Current features
* retrieve the vector representation for one or more word
* retrieve the top `n` nearest words given a vector

Currently only the ability to get vectors for one or more words and find the top `n` nearest words have been implemented.

If the word or vector cannot be found, a placeholder object with null content is returned.

The service uses gRPC's bidirectional streaming functionality, allowing multiple words or vectors to be requested and returned
concurrently. 

