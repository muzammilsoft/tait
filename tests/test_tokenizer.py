from tait.core.tokenizer import BPETokenizer, SEP, END


def test_bpe_roundtrip_control_tokens():
    tok = BPETokenizer()
    tok.train(["hello world", "مرحبا بالعالم", "你好世界"], vocab_size=300, sample_size=20, show_progress=False)
    text = "مرحبا" + SEP + "你好" + END
    assert tok.decode(tok.encode(text)) == text
