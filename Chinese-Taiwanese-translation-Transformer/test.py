# 載入訓練好的模型並計算 Test Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import math
import os

# 設置隨機種子
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. 重新定義所有必要的類 ====================

class Vocabulary:
    """詞彙表類"""
    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold
        self.itos = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.stoi = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        """從句子列表構建詞彙表"""
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in str(sentence).split():
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        return frequencies

    def numericalize(self, text):
        """將文本轉換為數字序列"""
        return [self.stoi.get(token, self.stoi['<unk>']) for token in str(text).split()]


class PositionalEncoding(nn.Module):
    """位置編碼"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多頭注意力機制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    """位置前饋網絡"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """編碼器層"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    """解碼器層"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    """完整的 Transformer 模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_seq_length=100, dropout=0.1):
        super().__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length),
                                      diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(tgt.device)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(
            self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)))
        tgt_embedded = self.dropout(self.positional_encoding(
            self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


# ==================== 2. 翻譯函數 ====================

def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_len=50):
    """翻譯單個句子"""
    model.eval()

    tokens = [src_vocab.stoi['<sos>']]
    tokens += src_vocab.numericalize(sentence)
    tokens.append(src_vocab.stoi['<eos>'])

    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
        src_embedded = model.positional_encoding(
            model.encoder_embedding(src_tensor) * math.sqrt(model.encoder_embedding.embedding_dim))

        enc_output = src_embedded
        for enc_layer in model.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        tgt_indices = [tgt_vocab.stoi['<sos>']]

        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            tgt_mask = model.generate_mask(src_tensor, tgt_tensor)[1]

            tgt_embedded = model.positional_encoding(
                model.decoder_embedding(tgt_tensor) * math.sqrt(model.decoder_embedding.embedding_dim))

            dec_output = tgt_embedded
            for dec_layer in model.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

            output = model.fc(dec_output)
            next_token = output.argmax(2)[:, -1].item()
            tgt_indices.append(next_token)

            if next_token == tgt_vocab.stoi['<eos>']:
                break

        tgt_tokens = [tgt_vocab.itos[idx] for idx in tgt_indices]
        return ' '.join(tgt_tokens[1:-1])


# ==================== 3. 準確率計算 ====================

def calculate_test_accuracy(model, src_data, tgt_data, src_vocab, tgt_vocab, device):
    """計算測試集準確率"""
    print("\n" + "="*60)
    print("計算測試集準確率")
    print("="*60)

    num_samples = len(src_data)

    # 不同層級的準確率
    exact_match = 0
    word_correct = 0
    word_total = 0
    char_correct = 0
    char_total = 0

    model.eval()

    print(f"\n評估樣本數: {num_samples}")
    print("開始計算...\n")

    for i in range(num_samples):
        src_text = src_data[i]
        tgt_text = tgt_data[i]

        # 翻譯
        pred_text = translate_sentence(model, src_text, src_vocab, tgt_vocab, device)

        # 1. 完全匹配準確率
        if pred_text.strip() == tgt_text.strip():
            exact_match += 1

        # 2. 詞級別準確率
        pred_words = pred_text.split()
        tgt_words = tgt_text.split()

        min_len = min(len(pred_words), len(tgt_words))
        for j in range(min_len):
            if pred_words[j] == tgt_words[j]:
                word_correct += 1
        word_total += len(tgt_words)

        # 3. 字符級別準確率
        pred_chars = list(pred_text.replace(' ', ''))
        tgt_chars = list(tgt_text.replace(' ', ''))

        min_char_len = min(len(pred_chars), len(tgt_chars))
        for j in range(min_char_len):
            if pred_chars[j] == tgt_chars[j]:
                char_correct += 1
        char_total += len(tgt_chars)

        # 顯示進度
        if (i + 1) % 50 == 0:
            print(f"  已完成: {i + 1}/{num_samples}")

    # 計算準確率
    exact_match_acc = (exact_match / num_samples) * 100
    word_acc = (word_correct / word_total) * 100 if word_total > 0 else 0
    char_acc = (char_correct / char_total) * 100 if char_total > 0 else 0

    print("\n" + "="*60)
    print("🎯 測試集準確率統計")
    print("="*60)
    print(f"評估樣本數: {num_samples}")
    print(f"\n1. 完全匹配準確率 (Exact Match):")
    print(f"   {exact_match}/{num_samples} = {exact_match_acc:.2f}%")
    print(f"\n2. 詞級別準確率 (Word-level):")
    print(f"   {word_correct}/{word_total} = {word_acc:.2f}%")
    print(f"\n3. 字符級別準確率 (Character-level):")
    print(f"   {char_correct}/{char_total} = {char_acc:.2f}%")

    # 顯示翻譯範例
    print("\n" + "="*60)
    print("翻譯範例 (前 20 個):")
    print("="*60)
    for i in range(min(20, num_samples)):
        src_text = src_data[i]
        tgt_text = tgt_data[i]
        pred_text = translate_sentence(model, src_text, src_vocab, tgt_vocab, device)

        match_symbol = "✅" if pred_text.strip() == tgt_text.strip() else "❌"

        print(f"\n[{i+1}] {match_symbol}")
        print(f"  閩南語: {src_text}")
        print(f"  真實:   {tgt_text}")
        print(f"  預測:   {pred_text}")

    return {
        'exact_match_acc': exact_match_acc,
        'word_acc': word_acc,
        'char_acc': char_acc,
        'exact_match': exact_match,
        'num_samples': num_samples
    }


# ==================== 4. 載入測試數據 ====================

def load_test_data(src_file, tgt_file, test_split=0.1):
    """載入測試數據"""
    from sklearn.model_selection import train_test_split

    print("正在讀取數據...")

    src_df = pd.read_csv(src_file)
    tgt_df = pd.read_csv(tgt_file)

    # 自動選擇欄位
    if 'text' in src_df.columns:
        src_col = 'text'
    elif 'txt' in src_df.columns:
        src_col = 'txt'
    else:
        src_col = [col for col in src_df.columns if col.lower() != 'id'][0]

    if 'text' in tgt_df.columns:
        tgt_col = 'text'
    elif 'txt' in tgt_df.columns:
        tgt_col = 'txt'
    else:
        tgt_col = [col for col in tgt_df.columns if col.lower() != 'id'][0]

    src_data = src_df[src_col].tolist()
    tgt_data = tgt_df[tgt_col].tolist()

    # 過濾空值
    valid_pairs = [(s, t) for s, t in zip(src_data, tgt_data)
                   if pd.notna(s) and pd.notna(t)]
    src_data = [pair[0] for pair in valid_pairs]
    tgt_data = [pair[1] for pair in valid_pairs]

    # 分割出測試集 (與訓練時相同的分割方式)
    val_split = 0.1
    src_train, src_temp, tgt_train, tgt_temp = train_test_split(
        src_data, tgt_data, test_size=(val_split + test_split), random_state=42
    )

    val_ratio = val_split / (val_split + test_split)
    src_val, src_test, tgt_val, tgt_test = train_test_split(
        src_temp, tgt_temp, test_size=(1 - val_ratio), random_state=42
    )

    print(f"測試集大小: {len(src_test)}")

    return src_test, tgt_test


# ==================== 5. 主程序 ====================

def main():
    print("="*60)
    print("載入訓練好的模型並計算測試集準確率")
    print("="*60)

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")

    # 模型路徑
    MODEL_PATH = './transformer_results/best_model.pth'

    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ 錯誤: 找不到模型檔案 {MODEL_PATH}")
        print("請確保模型已訓練並保存")
        return

    # 🔥 載入模型 (使用 weights_only=False)
    print(f"\n載入模型: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # 從 checkpoint 中獲取詞彙表
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    config = checkpoint['config']

    print(f"✅ 模型配置:")
    print(f"  - Epoch: {checkpoint['epoch'] + 1}")
    print(f"  - Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"  - Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  - d_model: {config['d_model']}")
    print(f"  - num_heads: {config['num_heads']}")
    print(f"  - num_layers: {config['num_layers']}")
    print(f"  - d_ff: {config['d_ff']}")
    print(f"  - dropout: {config['dropout']}")

    # 創建模型
    print("\n創建模型...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)

    # 載入模型權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ 模型載入成功")

    # 載入測試數據
    print("\n載入測試數據...")
    src_test, tgt_test = load_test_data('train-TL.csv', 'train-ZH.csv', test_split=0.1)

    # 計算測試集準確率
    test_acc = calculate_test_accuracy(model, src_test, tgt_test, src_vocab, tgt_vocab, device)

    # 最終結果
    print("\n" + "="*60)
    print("🎯 最終測試集結果")
    print("="*60)
    print(f"測試樣本數: {test_acc['num_samples']}")
    print(f"\n完全匹配準確率: {test_acc['exact_match_acc']:.2f}%")
    print(f"詞級別準確率:   {test_acc['word_acc']:.2f}%")
    print(f"字符級別準確率: {test_acc['char_acc']:.2f}%")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
