package splade

import (
	"context"
	"errors"
	"log"
	"sync"
	"time"

	"github.com/lordtatty/splade-server-goclient/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/status"
)

// IntegerEmbeddingResult holds the integer tokens and their corresponding values in slices to maintain order.
type IntegerEmbeddingResult struct {
	tokens []uint32
	values []float32
}

// NewIntegerEmbeddingResult creates a new IntegerEmbeddingResult from slices of integer tokens and values.
func NewIntegerEmbeddingResult(tokens []uint32, values []float32) *IntegerEmbeddingResult {
	return &IntegerEmbeddingResult{
		tokens: tokens,
		values: values,
	}
}

// ToMap returns a map representation of the embeddings with integer keys.
func (ier *IntegerEmbeddingResult) ToMap() map[uint32]float32 {
	embeddingMap := make(map[uint32]float32)
	for i, token := range ier.tokens {
		embeddingMap[token] = ier.values[i]
	}
	return embeddingMap
}

// Count returns the number of elements in the embedding.
func (ier *IntegerEmbeddingResult) Count() int {
	return len(ier.tokens)
}

// Indices returns a slice of integer tokens.
func (ier *IntegerEmbeddingResult) Indices() []uint32 {
	return ier.tokens
}

// Vals returns a slice of values.
func (ier *IntegerEmbeddingResult) Vals() []float32 {
	return ier.values
}

// EmbeddingResult holds the string tokens and their corresponding values in slices to maintain order.
type EmbeddingResult struct {
	tokens []string
	values []float32
}

// NewEmbeddingResult creates a new EmbeddingResult from slices of string tokens and values.
func NewEmbeddingResult(tokens []string, values []float32) *EmbeddingResult {
	return &EmbeddingResult{
		tokens: tokens,
		values: values,
	}
}

// ToMap returns a map representation of the embeddings with string keys.
func (er *EmbeddingResult) ToMap() map[string]float32 {
	embeddingMap := make(map[string]float32)
	for i, token := range er.tokens {
		embeddingMap[token] = er.values[i]
	}
	return embeddingMap
}

// Count returns the number of elements in the embedding.
func (er *EmbeddingResult) Count() int {
	return len(er.tokens)
}

// GetTokens returns a slice of string tokens.
func (er *EmbeddingResult) GetTokens() []string {
	return er.tokens
}

// GetValues returns a slice of values.
func (er *EmbeddingResult) GetValues() []float32 {
	return er.values
}

// SpladeClient provides methods to interact with the SPLADE gRPC service
type SpladeClient struct {
	client       pb.EmbeddingServiceClient
	conn         *grpc.ClientConn
	serverAddr   string
	mu           sync.Mutex
	isConnecting bool
}

// NewSpladeClient creates a new SpladeClient and attempts to connect to the gRPC server
func NewSpladeClient(serverAddr string) (*SpladeClient, error) {
	sc := &SpladeClient{serverAddr: serverAddr}
	err := sc.connect()
	if err != nil {
		return nil, err
	}
	return sc, nil
}

// connect establishes a connection to the gRPC server
func (sc *SpladeClient) connect() error {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	// Prevent concurrent connection attempts
	if sc.isConnecting {
		return errors.New("already attempting to connect")
	}
	sc.isConnecting = true
	defer func() { sc.isConnecting = false }()

	// Set up a connection to the gRPC server with backoff options for reconnection
	conn, err := grpc.Dial(
		sc.serverAddr,
		grpc.WithInsecure(),
		grpc.WithBlock(),
		grpc.WithBackoffMaxDelay(time.Second*5), // Maximum delay for reconnection attempts
		grpc.WithReturnConnectionError(),        // Return error if connection fails
	)
	if err != nil {
		return err
	}

	sc.conn = conn
	sc.client = pb.NewEmbeddingServiceClient(conn)
	log.Println("Connected to gRPC server")
	go sc.monitorConnection() // Start monitoring the connection state
	return nil
}

// monitorConnection watches the gRPC connection and attempts to reconnect if it goes down
func (sc *SpladeClient) monitorConnection() {
	for {
		state := sc.conn.GetState()
		if state == connectivity.Shutdown {
			log.Println("gRPC connection has been shut down. Attempting to reconnect...")
			if err := sc.connect(); err != nil {
				log.Printf("Reconnection failed: %v", err)
			}
		}
		sc.conn.WaitForStateChange(context.Background(), state)
	}
}

// Close closes the gRPC connection
func (sc *SpladeClient) Close() {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	if sc.conn != nil {
		sc.conn.Close()
		log.Println("gRPC connection closed")
	}
}

// GetEmbedding calls the GetEmbedding method on the gRPC server and returns an IntegerEmbeddingResult
func (sc *SpladeClient) GetEmbedding(text string) (*IntegerEmbeddingResult, error) {
	if err := sc.ensureConnected(); err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	req := &pb.TextRequest{Text: text}

	res, err := sc.client.GetEmbedding(ctx, req)
	if err != nil {
		st, _ := status.FromError(err)
		log.Printf("Error calling GetEmbedding: %v", st.Message())
		return nil, err
	}

	// Convert the map to ordered slices for IntegerEmbeddingResult
	tokens := make([]uint32, 0, len(res.Embedding))
	values := make([]float32, 0, len(res.Embedding))

	for tokenIndex, value := range res.Embedding {
		tokens = append(tokens, tokenIndex)
		values = append(values, value)
	}

	return NewIntegerEmbeddingResult(tokens, values), nil
}

// GetTokenEmbedding calls the GetTokenEmbedding method on the gRPC server and returns an EmbeddingResult
func (sc *SpladeClient) GetTokenEmbedding(text string) (*EmbeddingResult, error) {
	if err := sc.ensureConnected(); err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*10)
	defer cancel()

	req := &pb.TextRequest{Text: text}

	res, err := sc.client.GetTokenEmbedding(ctx, req)
	if err != nil {
		st, _ := status.FromError(err)
		log.Printf("Error calling GetTokenEmbedding: %v", st.Message())
		return nil, err
	}

	// Convert the map to ordered slices for EmbeddingResult
	tokens := make([]string, 0, len(res.Embedding))
	values := make([]float32, 0, len(res.Embedding))

	for token, value := range res.Embedding {
		tokens = append(tokens, token)
		values = append(values, value)
	}

	return NewEmbeddingResult(tokens, values), nil
}

// ensureConnected checks the connection state and attempts to reconnect if needed
func (sc *SpladeClient) ensureConnected() error {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	if sc.conn == nil || sc.conn.GetState() == connectivity.Shutdown {
		log.Println("gRPC connection is down. Attempting to reconnect...")
		return sc.connect()
	}
	return nil
}
