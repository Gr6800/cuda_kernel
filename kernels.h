// base
void shared_mem(
    torch::Tensor &output,  // [seq_len, hidden_dim]
    torch::Tensor &input    // [seq_len, hidden_dim]
);
void transpose(
    torch::Tensor &output,
    torch::Tensor &input
);
void tma(
    torch::Tensor &output,
    torch::Tensor &input
);
void global_mem(
    torch::Tensor &output,
    torch::Tensor &input,
    torch::Tensor &weight
);

// transformer
void rmsnorm();