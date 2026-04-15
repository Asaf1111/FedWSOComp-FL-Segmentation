def get_grpc_options():
    return [
        ("grpc.keepalive_time_ms", 120000000),            
        ("grpc.keepalive_timeout_ms", 600000000),          
        ("grpc.keepalive_permit_without_calls", 1),     
        ("grpc.http2.max_pings_without_data", 0),       
        ("grpc.http2.min_time_between_pings_ms", 10000000),
        ("grpc.http2.min_ping_interval_without_data_ms", 5000000),
        ("grpc.max_receive_message_length", 1024 * 1024 * 1024),  
        ("grpc.max_send_message_length", 1024 * 1024 * 1024),    
    ]


