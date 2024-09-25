// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.5.1
// - protoc             v5.28.1
// source: splade.proto

package pb

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.64.0 or later.
const _ = grpc.SupportPackageIsVersion9

const (
	EmbeddingService_GetEmbedding_FullMethodName      = "/splade.EmbeddingService/GetEmbedding"
	EmbeddingService_GetTokenEmbedding_FullMethodName = "/splade.EmbeddingService/GetTokenEmbedding"
)

// EmbeddingServiceClient is the client API for EmbeddingService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type EmbeddingServiceClient interface {
	GetEmbedding(ctx context.Context, in *TextRequest, opts ...grpc.CallOption) (*EmbeddingResponse, error)
	GetTokenEmbedding(ctx context.Context, in *TextRequest, opts ...grpc.CallOption) (*TokenEmbeddingResponse, error)
}

type embeddingServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewEmbeddingServiceClient(cc grpc.ClientConnInterface) EmbeddingServiceClient {
	return &embeddingServiceClient{cc}
}

func (c *embeddingServiceClient) GetEmbedding(ctx context.Context, in *TextRequest, opts ...grpc.CallOption) (*EmbeddingResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(EmbeddingResponse)
	err := c.cc.Invoke(ctx, EmbeddingService_GetEmbedding_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *embeddingServiceClient) GetTokenEmbedding(ctx context.Context, in *TextRequest, opts ...grpc.CallOption) (*TokenEmbeddingResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(TokenEmbeddingResponse)
	err := c.cc.Invoke(ctx, EmbeddingService_GetTokenEmbedding_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// EmbeddingServiceServer is the server API for EmbeddingService service.
// All implementations must embed UnimplementedEmbeddingServiceServer
// for forward compatibility.
type EmbeddingServiceServer interface {
	GetEmbedding(context.Context, *TextRequest) (*EmbeddingResponse, error)
	GetTokenEmbedding(context.Context, *TextRequest) (*TokenEmbeddingResponse, error)
	mustEmbedUnimplementedEmbeddingServiceServer()
}

// UnimplementedEmbeddingServiceServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedEmbeddingServiceServer struct{}

func (UnimplementedEmbeddingServiceServer) GetEmbedding(context.Context, *TextRequest) (*EmbeddingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetEmbedding not implemented")
}
func (UnimplementedEmbeddingServiceServer) GetTokenEmbedding(context.Context, *TextRequest) (*TokenEmbeddingResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetTokenEmbedding not implemented")
}
func (UnimplementedEmbeddingServiceServer) mustEmbedUnimplementedEmbeddingServiceServer() {}
func (UnimplementedEmbeddingServiceServer) testEmbeddedByValue()                          {}

// UnsafeEmbeddingServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to EmbeddingServiceServer will
// result in compilation errors.
type UnsafeEmbeddingServiceServer interface {
	mustEmbedUnimplementedEmbeddingServiceServer()
}

func RegisterEmbeddingServiceServer(s grpc.ServiceRegistrar, srv EmbeddingServiceServer) {
	// If the following call pancis, it indicates UnimplementedEmbeddingServiceServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&EmbeddingService_ServiceDesc, srv)
}

func _EmbeddingService_GetEmbedding_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(TextRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EmbeddingServiceServer).GetEmbedding(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: EmbeddingService_GetEmbedding_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EmbeddingServiceServer).GetEmbedding(ctx, req.(*TextRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _EmbeddingService_GetTokenEmbedding_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(TextRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EmbeddingServiceServer).GetTokenEmbedding(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: EmbeddingService_GetTokenEmbedding_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EmbeddingServiceServer).GetTokenEmbedding(ctx, req.(*TextRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// EmbeddingService_ServiceDesc is the grpc.ServiceDesc for EmbeddingService service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var EmbeddingService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "splade.EmbeddingService",
	HandlerType: (*EmbeddingServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "GetEmbedding",
			Handler:    _EmbeddingService_GetEmbedding_Handler,
		},
		{
			MethodName: "GetTokenEmbedding",
			Handler:    _EmbeddingService_GetTokenEmbedding_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "splade.proto",
}
