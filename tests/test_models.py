# mypy: ignore-errors
import pytest
import torch

from src.models import (
    AttentionHead,
    MultiHeadAttention,
    FeedForward,
    TransformerEncoderLayer,
    Embeddings,
    TransformerEncoder,
    ClassificationHead,
    TransformerForSequenceClassification,
)


# -------- AttentionHead --------

@pytest.fixture
def attention_head():
    torch.manual_seed(0)
    d_model, d_k, d_q, d_v = 32, 8, 8, 8
    return AttentionHead(d_model, d_k, d_q, d_v)


def test_attention_head_forward_shape(attention_head):
    batch_size, seq_len, d_model = 2, 5, 32
    x = torch.randn(batch_size, seq_len, d_model)
    y = attention_head(x)
    assert y.shape == (batch_size, seq_len, attention_head.wv.out_features)


def test_scaled_dot_product_attention_shapes_and_row_sums(attention_head):
    b, t, d = 3, 7, 8
    q = torch.randn(b, t, d)
    k = torch.randn(b, t, d)
    v = torch.randn(b, t, d)
    out, w = attention_head.scaled_dot_product_attention(q, k, v)
    assert out.shape == (b, t, d)
    assert w.shape == (b, t, t)
    # rows of attention weights should sum to 1
    row_sums = w.sum(dim=-1) #no entiendo pq son las columnas las que tienen que sumar 1, srgún las diapos serían las filas?
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_scaled_dot_product_attention_uniform_when_scores_zero(attention_head):
    # When scores are all zeros, softmax should be uniform
    b, t, d = 1, 4, 8
    q = torch.zeros(b, t, d)
    k = torch.zeros(b, t, d)
    v = torch.randn(b, t, d)
    _, w = attention_head.scaled_dot_product_attention(q, k, v)
    expected = torch.full((b, t, t), 1.0 / t)
    assert torch.allclose(w, expected, atol=1e-6)


# -------- MultiHeadAttention --------

@pytest.fixture
def multihead_attention():
    torch.manual_seed(0)
    return MultiHeadAttention(d_model=32, num_attention_heads=4)


def test_multihead_attention_output_shape(multihead_attention):
    b, t, d = 2, 6, 32
    x = torch.randn(b, t, d)
    y = multihead_attention(x)
    assert y.shape == (b, t, d)


def test_multihead_attention_invalid_heads_raises():
    # d_model not divisible by heads should error on forward due to shape mismatch
    m = MultiHeadAttention(d_model=30, num_attention_heads=4)
    x = torch.randn(1, 3, 30)
    with pytest.raises(RuntimeError):
        _ = m(x)


# -------- FeedForward --------

@pytest.fixture
def ff():
    torch.manual_seed(0)
    return FeedForward(d_model=32, intermediate_size=64)


def test_feedforward_shape_and_change(ff):
    b, t, d = 2, 5, 32
    x = torch.randn(b, t, d)
    y = ff(x)
    assert y.shape == (b, t, d)
    assert not torch.equal(x, y)


def test_feedforward_backward(ff):
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = ff(x).sum()
    y.backward()
    assert x.grad is not None


# -------- TransformerEncoderLayer --------

@pytest.fixture
def encoder_layer():
    torch.manual_seed(0)
    return TransformerEncoderLayer(d_model=32, num_attention_heads=4, intermediate_size=64)


def test_encoder_layer_shape_and_residual(encoder_layer):
    b, t, d = 2, 7, 32
    x = torch.randn(b, t, d)
    y = encoder_layer(x)
    assert y.shape == (b, t, d)
    assert not torch.equal(x, y)


# -------- Embeddings --------

@pytest.fixture
def embeddings():
    return Embeddings(vocab_size=100, max_position_embeddings=64, d_model=32)


def test_embeddings_shape_and_layer_norm(embeddings):
    b, t = 3, 12
    input_ids = torch.randint(0, 100, (b, t))
    y = embeddings(input_ids)
    assert y.shape == (b, t, 32)
    # LayerNorm should produce ~zero mean along last dim initially
    mean = y.mean(dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


# -------- TransformerEncoder --------

@pytest.fixture
def encoder():
    return TransformerEncoder(
        vocab_size=100,
        max_position_embeddings=64,
        d_model=32,
        num_attention_heads=4,
        intermediate_size=64,
        num_hidden_layers=2,
    )


def test_encoder_output_shape_and_stacking(encoder):
    b, t = 2, 10
    input_ids = torch.randint(0, 100, (b, t))
    y = encoder(input_ids)
    assert y.shape == (b, t, 32)
    # different from pure embeddings
    emb = encoder.embeddings(input_ids)
    assert not torch.equal(y, emb)


# -------- Classification head --------

def test_classification_head_train_vs_eval_dropout():
    torch.manual_seed(0)
    head = ClassificationHead(d_model=32, num_classes=5, dropout_prob=0.5)
    x = torch.randn(4, 32)
    head.train()
    y1 = head(x)
    y2 = head(x)
    # In train mode with dropout, outputs should likely differ
    assert not torch.allclose(y1, y2)
    head.eval()
    y3 = head(x)
    y4 = head(x)
    assert torch.allclose(y3, y4)


# -------- TransformerForSequenceClassification --------

def test_transformer_for_sequence_classification_shape_and_backward():
    torch.manual_seed(0)
    model = TransformerForSequenceClassification(
        vocab_size=100,
        max_position_embeddings=64,
        d_model=32,
        num_attention_heads=4,
        intermediate_size=64,
        num_hidden_layers=2,
        num_classes=3,
        dropout_prob=0.1,
    )
    b, t = 2, 9
    input_ids = torch.randint(0, 100, (b, t))
    logits = model(input_ids)
    assert logits.shape == (b, 3)
    loss = logits.sum()
    loss.backward()
    # verify some encoder param has gradient
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad

