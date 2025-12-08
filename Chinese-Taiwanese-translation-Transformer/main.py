# 閩南語vs中文翻譯 (閩南語→中文) - 防過擬合版本 + 計算準確率 (移除 Early Stopping)
# Train 80% / Validation 10% / Test 10%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import math
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 設置隨機種子
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. 數據處理 ====================

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

        print(f"詞彙表大小: {len(self.itos)}")
        return frequencies

    def numericalize(self, text):
        """將文本轉換為數字序列"""
        return [self.stoi.get(token, self.stoi['<unk>']) for token in str(text).split()]


class TranslationDataset(Dataset):
    """翻譯數據集 (閩南語→中文)"""
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]

        # 轉換為數字序列
        src_numericalized = [self.src_vocab.stoi['<sos>']]
        src_numericalized += self.src_vocab.numericalize(src_text)
        src_numericalized.append(self.src_vocab.stoi['<eos>'])

        tgt_numericalized = [self.tgt_vocab.stoi['<sos>']]
        tgt_numericalized += self.tgt_vocab.numericalize(tgt_text)
        tgt_numericalized.append(self.tgt_vocab.stoi['<eos>'])

        return torch.tensor(src_numericalized), torch.tensor(tgt_numericalized)


def load_and_split_data_three_way(src_file, tgt_file, val_split=0.1, test_split=0.1, freq_threshold=1):
    """載入數據並分割訓練集/驗證集/測試集 (80%/10%/10%)"""
    print(f"正在讀取數據...")

    # 讀取數據
    src_df = pd.read_csv(src_file)  # train-TL.csv (閩南語)
    tgt_df = pd.read_csv(tgt_file)  # train-ZH.csv (中文)

    # 檢測欄位名稱
    print(f"源語言檔案欄位: {src_df.columns.tolist()}")
    print(f"目標語言檔案欄位: {tgt_df.columns.tolist()}")

    # 自動選擇正確的欄位
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

    print(f"使用源語言欄位: '{src_col}' (閩南語)")
    print(f"使用目標語言欄位: '{tgt_col}' (中文)")

    src_data = src_df[src_col].tolist()
    tgt_data = tgt_df[tgt_col].tolist()

    # 過濾空值
    valid_pairs = [(s, t) for s, t in zip(src_data, tgt_data)
                   if pd.notna(s) and pd.notna(t)]
    src_data = [pair[0] for pair in valid_pairs]
    tgt_data = [pair[1] for pair in valid_pairs]

    print(f"有效數據筆數: {len(src_data)}")

    # 🔥 三次分割: Train 80% / Val 10% / Test 10%
    # 第一次分割: 80% train, 20% temp
    src_train, src_temp, tgt_train, tgt_temp = train_test_split(
        src_data, tgt_data, test_size=(val_split + test_split), random_state=42
    )

    # 第二次分割: 將 temp 分成 val 和 test (各佔總數據的 10%)
    val_ratio = val_split / (val_split + test_split)  # 0.5 (在 temp 中各佔一半)
    src_val, src_test, tgt_val, tgt_test = train_test_split(
        src_temp, tgt_temp, test_size=(1 - val_ratio), random_state=42
    )

    print(f"\n數據分割:")
    print(f"  訓練集: {len(src_train)} ({len(src_train)/len(src_data)*100:.1f}%)")
    print(f"  驗證集: {len(src_val)} ({len(src_val)/len(src_data)*100:.1f}%)")
    print(f"  測試集: {len(src_test)} ({len(src_test)/len(src_data)*100:.1f}%)")

    # 構建詞彙表 (只用訓練集)
    print("\n構建源語言詞彙表 (閩南語)...")
    src_vocab = Vocabulary(freq_threshold)
    src_freq = src_vocab.build_vocabulary(src_train)

    print("構建目標語言詞彙表 (中文)...")
    tgt_vocab = Vocabulary(freq_threshold)
    tgt_freq = tgt_vocab.build_vocabulary(tgt_train)

    # 統計句子長度
    src_lengths = [len(str(s).split()) for s in src_data]
    tgt_lengths = [len(str(t).split()) for t in tgt_data]

    # 顯示範例
    print(f"\n數據範例:")
    for i in range(min(3, len(src_train))):
        print(f"  [{i+1}] 閩南語: {src_train[i]}")
        print(f"      中文:   {tgt_train[i]}")

    return (src_train, src_val, src_test,
            tgt_train, tgt_val, tgt_test,
            src_vocab, tgt_vocab, src_freq, tgt_freq,
            src_lengths, tgt_lengths)


class MyCollate:
    """自定義 collate 函數"""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_batch = [item[0] for item in batch]
        tgt_batch = [item[1] for item in batch]

        src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True,
                                              padding_value=self.pad_idx)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True,
                                              padding_value=self.pad_idx)

        return src_batch, tgt_batch


# ==================== 2. 模型定義 ====================

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


# ==================== 3. 訓練和評估 ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    batch_losses = []

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_losses.append(loss.item())

    return total_loss / len(dataloader), batch_losses


def validate_epoch(model, dataloader, criterion, device):
    """驗證一個 epoch"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)


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


# ==================== 4. 準確率計算 ====================

def calculate_accuracy(model, src_data, tgt_data, src_vocab, tgt_vocab, device, dataset_name="數據集", num_samples=None):
    """計算翻譯準確率"""
    print("\n" + "="*60)
    print(f"計算 {dataset_name} 翻譯準確率")
    print("="*60)

    if num_samples is None:
        num_samples = len(src_data)
    else:
        num_samples = min(num_samples, len(src_data))

    # 不同層級的準確率
    exact_match = 0  # 完全匹配
    word_correct = 0  # 詞級別正確數
    word_total = 0    # 總詞數
    char_correct = 0  # 字符級別正確數
    char_total = 0    # 總字符數

    model.eval()

    print(f"\n評估樣本數: {num_samples}")
    print("開始計算...")

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

        # 計算詞級別的正確數
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
        if (i + 1) % 100 == 0:
            print(f"  已完成: {i + 1}/{num_samples}")

    # 計算準確率
    exact_match_acc = (exact_match / num_samples) * 100
    word_acc = (word_correct / word_total) * 100 if word_total > 0 else 0
    char_acc = (char_correct / char_total) * 100 if char_total > 0 else 0

    print("\n" + "="*60)
    print(f"{dataset_name} 準確率統計")
    print("="*60)
    print(f"評估樣本數: {num_samples}")
    print(f"\n1. 完全匹配準確率 (Exact Match Accuracy):")
    print(f"   {exact_match}/{num_samples} = {exact_match_acc:.2f}%")
    print(f"\n2. 詞級別準確率 (Word-level Accuracy):")
    print(f"   {word_correct}/{word_total} = {word_acc:.2f}%")
    print(f"\n3. 字符級別準確率 (Character-level Accuracy):")
    print(f"   {char_correct}/{char_total} = {char_acc:.2f}%")

    # 顯示一些翻譯範例
    print("\n" + "="*60)
    print(f"{dataset_name} 翻譯範例 (前 10 個):")
    print("="*60)
    for i in range(min(10, num_samples)):
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


# ==================== 5. 視覺化和分析 ====================

def plot_training_curves(train_losses, val_losses, save_dir='./results'):
    """繪製訓練和驗證曲線"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Training & Validation Loss
    axes[0].plot(train_losses, linewidth=2, color='#2E86AB', label='Training Loss', marker='o', markersize=4)
    axes[0].plot(val_losses, linewidth=2, color='#F18F01', label='Validation Loss', marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 2. Loss 變化率
    if len(train_losses) > 1:
        train_changes = [train_losses[i] - train_losses[i-1] for i in range(1, len(train_losses))]
        val_changes = [val_losses[i] - val_losses[i-1] for i in range(1, len(val_losses))]

        axes[1].plot(train_changes, linewidth=2, color='#2E86AB', label='Train Loss Change', marker='o', markersize=4)
        axes[1].plot(val_changes, linewidth=2, color='#F18F01', label='Val Loss Change', marker='s', markersize=4)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss Change', fontsize=12)
        axes[1].set_title('Loss Change Rate', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

    # 3. Train-Val Gap
    if len(train_losses) == len(val_losses):
        gap = [val_losses[i] - train_losses[i] for i in range(len(train_losses))]
        axes[2].plot(gap, linewidth=2, color='#A23B72', marker='D', markersize=4)
        axes[2].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='No Overfitting')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Val Loss - Train Loss', fontsize=12)
        axes[2].set_title('Overfitting Monitor', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

        # 標註過擬合區域
        if any(g > 0 for g in gap):
            axes[2].fill_between(range(len(gap)), 0, gap, where=[g > 0 for g in gap],
                                alpha=0.3, color='red', label='Overfitting Zone')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✅ 訓練曲線已保存至: {save_dir}/training_curves.png")
    plt.show()


def plot_data_statistics(src_lengths, tgt_lengths, src_vocab, tgt_vocab,
                         src_freq, tgt_freq, save_dir='./results'):
    """繪製數據統計圖"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 句子長度分布
    axes[0, 0].hist(src_lengths, bins=30, alpha=0.7, label='Minnan', color='#2E86AB', edgecolor='black')
    axes[0, 0].hist(tgt_lengths, bins=30, alpha=0.7, label='Chinese', color='#A23B72', edgecolor='black')
    axes[0, 0].set_xlabel('Sentence Length', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Sentence Length Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 詞彙量對比
    vocab_sizes = [len(src_vocab), len(tgt_vocab)]
    axes[0, 1].bar(['Minnan', 'Chinese'], vocab_sizes, color=['#2E86AB', '#A23B72'], edgecolor='black')
    axes[0, 1].set_ylabel('Vocabulary Size', fontsize=12)
    axes[0, 1].set_title('Vocabulary Size Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(vocab_sizes):
        axes[0, 1].text(i, v + 50, str(v), ha='center', fontweight='bold')

    # 3. 詞頻 Top 20 (閩南語)
    if src_freq:
        top_words = sorted(src_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        words, freqs = zip(*top_words)
        axes[1, 0].barh(range(len(words)), freqs, color='#2E86AB', edgecolor='black')
        axes[1, 0].set_yticks(range(len(words)))
        axes[1, 0].set_yticklabels(words, fontsize=10)
        axes[1, 0].set_xlabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Top 20 Minnan Words', fontsize=14, fontweight='bold')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. 詞頻 Top 20 (中文)
    if tgt_freq:
        top_words = sorted(tgt_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        words, freqs = zip(*top_words)
        axes[1, 1].barh(range(len(words)), freqs, color='#A23B72', edgecolor='black')
        axes[1, 1].set_yticks(range(len(words)))
        axes[1, 1].set_yticklabels(words, fontsize=10)
        axes[1, 1].set_xlabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Top 20 Chinese Words', fontsize=14, fontweight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/data_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✅ 數據統計圖已保存至: {save_dir}/data_statistics.png")
    plt.show()


def plot_accuracy_comparison(train_acc, val_acc, test_acc, save_dir='./results'):
    """繪製 Train/Val/Test 準確率對比圖"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    datasets = ['Train', 'Validation', 'Test']
    exact_match = [train_acc['exact_match_acc'], val_acc['exact_match_acc'], test_acc['exact_match_acc']]
    word_acc = [train_acc['word_acc'], val_acc['word_acc'], test_acc['word_acc']]
    char_acc = [train_acc['char_acc'], val_acc['char_acc'], test_acc['char_acc']]

    colors = ['#2E86AB', '#F18F01', '#A23B72']

    # 1. 完全匹配準確率
    axes[0].bar(datasets, exact_match, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Exact Match Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(exact_match):
        axes[0].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')

    # 2. 詞級別準確率
    axes[1].bar(datasets, word_acc, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Word-level Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(word_acc):
        axes[1].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')

    # 3. 字符級別準確率
    axes[2].bar(datasets, char_acc, color=colors, edgecolor='black', alpha=0.8)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].set_title('Character-level Accuracy', fontsize=14, fontweight='bold')
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(char_acc):
        axes[2].text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 準確率對比圖已保存至: {save_dir}/accuracy_comparison.png")
    plt.show()


def save_comprehensive_report(train_losses, val_losses, src_lengths, tgt_lengths,
                              src_vocab, tgt_vocab, model, best_epoch,
                              train_acc, val_acc, test_acc, save_dir='./results'):
    """保存完整訓練報告 (包含 Test 結果)"""
    os.makedirs(save_dir, exist_ok=True)

    # 計算過擬合指標
    final_gap = val_losses[-1] - train_losses[-1]
    max_gap = max([val_losses[i] - train_losses[i] for i in range(len(train_losses))])

    report = f"""
{'='*60}
Transformer 閩南語→中文翻譯系統 - 完整訓練報告
{'='*60}
訓練時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【數據統計】
- 總樣本數: {len(src_lengths)}
- 訓練集: {train_acc['num_samples']} (80%)
- 驗證集: {val_acc['num_samples']} (10%)
- 測試集: {test_acc['num_samples']} (10%)
- 閩南語詞彙量: {len(src_vocab)}
- 中文詞彙量: {len(tgt_vocab)}
- 平均句子長度 (閩南語): {np.mean(src_lengths):.2f} 詞
- 平均句子長度 (中文): {np.mean(tgt_lengths):.2f} 詞
- 最長句子 (閩南語): {max(src_lengths)} 詞
- 最長句子 (中文): {max(tgt_lengths)} 詞

【模型配置】
- 模型參數量: {sum(p.numel() for p in model.parameters()):,}
- Embedding 維度: {model.encoder_embedding.embedding_dim}
- 注意力頭數: {model.encoder_layers[0].self_attn.num_heads}
- 編碼器層數: {len(model.encoder_layers)}
- 解碼器層數: {len(model.decoder_layers)}

【訓練結果】
- 訓練輪數: {len(train_losses)}
- 最佳模型 Epoch: {best_epoch + 1}
- 初始 Train Loss: {train_losses[0]:.4f}
- 最終 Train Loss: {train_losses[-1]:.4f}
- 最佳 Train Loss: {min(train_losses):.4f}
- Train Loss 降低: {train_losses[0] - train_losses[-1]:.4f} ({(train_losses[0] - train_losses[-1])/train_losses[0]*100:.2f}%)

【驗證結果】
- 初始 Val Loss: {val_losses[0]:.4f}
- 最終 Val Loss: {val_losses[-1]:.4f}
- 最佳 Val Loss: {min(val_losses):.4f}
- Val Loss 降低: {val_losses[0] - val_losses[-1]:.4f} ({(val_losses[0] - val_losses[-1])/val_losses[0]*100:.2f}%)

【過擬合分析】
- 最終 Train-Val Gap: {final_gap:.4f}
- 最大 Train-Val Gap: {max_gap:.4f}
- 過擬合狀態: {'⚠️ 有過擬合跡象' if final_gap > 0.5 else '✅ 良好'}

【準確率評估 - 訓練集】
- 樣本數: {train_acc['num_samples']}
- 完全匹配: {train_acc['exact_match_acc']:.2f}% ({train_acc['exact_match']}/{train_acc['num_samples']})
- 詞級別: {train_acc['word_acc']:.2f}%
- 字符級別: {train_acc['char_acc']:.2f}%

【準確率評估 - 驗證集】
- 樣本數: {val_acc['num_samples']}
- 完全匹配: {val_acc['exact_match_acc']:.2f}% ({val_acc['exact_match']}/{val_acc['num_samples']})
- 詞級別: {val_acc['word_acc']:.2f}%
- 字符級別: {val_acc['char_acc']:.2f}%

【準確率評估 - 測試集】🔥 最終評估
- 樣本數: {test_acc['num_samples']}
- 完全匹配: {test_acc['exact_match_acc']:.2f}% ({test_acc['exact_match']}/{test_acc['num_samples']})
- 詞級別: {test_acc['word_acc']:.2f}%
- 字符級別: {test_acc['char_acc']:.2f}%

【Loss 統計】
訓練集:
  - 平均 Loss: {np.mean(train_losses):.4f}
  - Loss 標準差: {np.std(train_losses):.4f}
  - Loss 中位數: {np.median(train_losses):.4f}

驗證集:
  - 平均 Loss: {np.mean(val_losses):.4f}
  - Loss 標準差: {np.std(val_losses):.4f}
  - Loss 中位數: {np.median(val_losses):.4f}

{'='*60}
"""

    with open(f'{save_dir}/comprehensive_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 完整報告已保存至: {save_dir}/comprehensive_report.txt")
    print(report)


# ==================== 6. 主程序 ====================

def main():
    print("="*60)
    print("Transformer 閩南語→中文翻譯系統")
    print("Train 80% / Validation 10% / Test 10%")
    print("="*60)

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")
    if torch.cuda.is_available():
        print(f"GPU 型號: {torch.cuda.get_device_name(0)}")

    # 超參數
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.0001
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    D_FF = 1024
    DROPOUT = 0.5
    VAL_SPLIT = 0.10  # 10%
    TEST_SPLIT = 0.10  # 10%
    SAVE_DIR = './transformer_results'

    # 🔥 創建保存目錄
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n【超參數設置】")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  數據分割: Train 80% / Val 10% / Test 10%")

    # 🔥 載入數據並三分割
    print("\n【步驟 1】載入數據並分割 Train/Val/Test (80%/10%/10%)")
    try:
        (src_train, src_val, src_test,
         tgt_train, tgt_val, tgt_test,
         src_vocab, tgt_vocab, src_freq, tgt_freq,
         src_lengths, tgt_lengths) = load_and_split_data_three_way(
            'train-TL.csv', 'train-ZH.csv',
            val_split=VAL_SPLIT, test_split=TEST_SPLIT, freq_threshold=1
        )
    except FileNotFoundError:
        print("\n❌ 錯誤: 找不到檔案")
        print("請確保 train-TL.csv 和 train-ZH.csv 存在")
        return

    # 繪製數據統計圖
    print("\n【步驟 2】數據分析")
    plot_data_statistics(src_lengths, tgt_lengths, src_vocab, tgt_vocab,
                        src_freq, tgt_freq, SAVE_DIR)

    # 🔥 創建三個 DataLoader
    print("\n【步驟 3】創建 DataLoader")
    train_dataset = TranslationDataset(src_train, tgt_train, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(src_val, tgt_val, src_vocab, tgt_vocab)
    test_dataset = TranslationDataset(src_test, tgt_test, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=MyCollate(pad_idx=src_vocab.stoi['<pad>'])
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=MyCollate(pad_idx=src_vocab.stoi['<pad>'])
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=MyCollate(pad_idx=src_vocab.stoi['<pad>'])
    )

    print(f"訓練批次數: {len(train_loader)}")
    print(f"驗證批次數: {len(val_loader)}")
    print(f"測試批次數: {len(test_loader)}")

    # 創建模型
    print("\n【步驟 4】創建模型")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.stoi['<pad>'])

    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    # 訓練
    print("\n【步驟 5】開始訓練")
    print("="*60)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss, _ = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        end_time = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        lr_changed = ""
        if old_lr != current_lr:
            lr_changed = f" → LR 降低至 {current_lr:.6f}"

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {end_time - start_time:.2f}s{lr_changed}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'config': {
                    'd_model': D_MODEL,
                    'num_heads': NUM_HEADS,
                    'num_layers': NUM_LAYERS,
                    'd_ff': D_FF,
                    'dropout': DROPOUT
                }
            }, f'{SAVE_DIR}/best_model.pth')
            print(f"  ✅ 保存最佳模型 (Val Loss: {val_loss:.4f})")

    # 繪製訓練曲線
    print("\n【步驟 6】生成視覺化結果")
    plot_training_curves(train_losses, val_losses, SAVE_DIR)

    # 載入最佳模型
    print("\n【步驟 7】載入最佳模型")
    checkpoint = torch.load(f'{SAVE_DIR}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 載入最佳模型 (Epoch {checkpoint['epoch']+1}, Val Loss: {checkpoint['val_loss']:.4f})")

    # 🔥 計算三個數據集的準確率
    print("\n【步驟 8】計算 Train/Val/Test 準確率")
    print("="*60)

    train_acc = calculate_accuracy(model, src_train, tgt_train, src_vocab, tgt_vocab, device, "訓練集")
    val_acc = calculate_accuracy(model, src_val, tgt_val, src_vocab, tgt_vocab, device, "驗證集")
    test_acc = calculate_accuracy(model, src_test, tgt_test, src_vocab, tgt_vocab, device, "測試集")

    # 繪製準確率對比圖
    print("\n【步驟 9】生成準確率對比圖")
    plot_accuracy_comparison(train_acc, val_acc, test_acc, SAVE_DIR)

    # 保存完整報告
    print("\n【步驟 10】保存完整報告")
    save_comprehensive_report(train_losses, val_losses, src_lengths, tgt_lengths,
                             src_vocab, tgt_vocab, model, best_epoch,
                             train_acc, val_acc, test_acc, SAVE_DIR)

    # 最終結果摘要
    print("\n" + "="*60)
    print("🎯 最終結果摘要")
    print("="*60)
    print(f"\n【訓練集】 ({train_acc['num_samples']} 樣本)")
    print(f"  完全匹配: {train_acc['exact_match_acc']:.2f}%")
    print(f"  詞級別:   {train_acc['word_acc']:.2f}%")
    print(f"  字符級別: {train_acc['char_acc']:.2f}%")

    print(f"\n【驗證集】 ({val_acc['num_samples']} 樣本)")
    print(f"  完全匹配: {val_acc['exact_match_acc']:.2f}%")
    print(f"  詞級別:   {val_acc['word_acc']:.2f}%")
    print(f"  字符級別: {val_acc['char_acc']:.2f}%")

    print(f"\n【測試集】 ({test_acc['num_samples']} 樣本) ← 🔥 最終評估")
    print(f"  完全匹配: {test_acc['exact_match_acc']:.2f}%")
    print(f"  詞級別:   {test_acc['word_acc']:.2f}%")
    print(f"  字符級別: {test_acc['char_acc']:.2f}%")

    print("\n" + "="*60)
    print("✅ 訓練完成！所有結果已保存至:", SAVE_DIR)
    print("="*60)
    print("\n保存的檔案:")
    print(f"  📊 training_curves.png - 訓練/驗證曲線圖")
    print(f"  📊 data_statistics.png - 數據統計圖")
    print(f"  📊 accuracy_comparison.png - 準確率對比圖")
    print(f"  📄 comprehensive_report.txt - 完整訓練報告")
    print(f"  💾 best_model.pth - 最佳模型")


if __name__ == "__main__":
    main()
