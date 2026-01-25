"""
CRFライブラリ比較・テストスクリプト

利用可能なCRFライブラリ：
1. pytorch-crf (kmkurn/pytorch-crf)
2. TorchCRF (s14t/TorchCRF)

それぞれの特徴と使用方法を比較
"""

import torch
import torch.nn as nn
import numpy as np

print("🔍 CRF Libraries Comparison")
print("=" * 50)

# 1. pytorch-crf
try:
    from pytorch_crf import CRF as PyTorchCRF
    print("✅ pytorch-crf (0.7.2) - Available")
    
    # Basic usage example
    vocab_size = 10
    num_tags = 5
    seq_length = 20
    batch_size = 2
    
    # Create a simple CRF layer
    crf_pytorch = PyTorchCRF(num_tags, batch_first=True)
    
    # Sample emissions (batch_size, seq_length, num_tags)
    emissions = torch.randn(batch_size, seq_length, num_tags)
    # Sample tags (batch_size, seq_length)  
    tags = torch.randint(0, num_tags, (batch_size, seq_length))
    # Sample mask (batch_size, seq_length)
    mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    
    # Calculate log-likelihood
    log_likelihood = crf_pytorch(emissions, tags, mask=mask)
    print(f"  - Log likelihood: {log_likelihood.item():.4f}")
    
    # Decode (Viterbi)
    decoded = crf_pytorch.decode(emissions, mask=mask)
    print(f"  - Decoded sequence length: {len(decoded[0])}")
    
    print("  📋 pytorch-crf Features:")
    print("    - Batch-first support ✅")
    print("    - Mask support ✅") 
    print("    - Viterbi decoding ✅")
    print("    - Well documented ✅")
    print("    - AllenNLP compatible ✅")
    
except ImportError as e:
    print(f"❌ pytorch-crf not available: {e}")

print()

# 2. TorchCRF
try:
    from TorchCRF import CRF as TorchCRF
    print("✅ TorchCRF (1.1.0) - Available")
    
    # Create a TorchCRF layer
    crf_torch = TorchCRF(num_tags)
    
    # Sample data (TorchCRF uses seq_len, batch_size format by default)
    emissions_torch = torch.randn(seq_length, batch_size, num_tags)
    tags_torch = torch.randint(0, num_tags, (seq_length, batch_size))
    
    # Calculate negative log-likelihood
    loss = crf_torch(emissions_torch, tags_torch)
    print(f"  - Loss (negative log-likelihood): {loss.item():.4f}")
    
    # Decode (Viterbi)
    decoded_torch = crf_torch.decode(emissions_torch)
    print(f"  - Decoded sequence length: {len(decoded_torch[0])}")
    
    print("  📋 TorchCRF Features:")
    print("    - Simple API ✅")
    print("    - Sequence-first by default ✅")
    print("    - Viterbi decoding ✅")
    print("    - Lightweight ✅")
    
except ImportError as e:
    print(f"❌ TorchCRF not available: {e}")

print()

# 3. Additional CRF options to consider
print("🔧 Additional CRF Implementation Options:")
print("=" * 50)

print("3. 📦 AllenNLP CRF:")
print("   - pip install allennlp")
print("   - from allennlp.modules import ConditionalRandomField")
print("   - Enterprise-grade, well-tested")
print("   - More heavyweight dependency")

print()
print("4. 🛠️ Custom CRF Implementation:")
print("   - Implement from scratch using PyTorch")
print("   - Full control over implementation")
print("   - Educational value")

print()
print("5. 📚 Flair CRF:")
print("   - pip install flair")
print("   - from flair.models import SequenceTagger")
print("   - NLP-focused, ready-to-use models")

print()

# Recommendation
print("💡 Recommendations for v0.3 Implementation:")
print("=" * 50)

print("🥇 1st Choice: pytorch-crf")
print("   - ✅ Most popular and well-maintained")
print("   - ✅ AllenNLP heritage (proven in research)")
print("   - ✅ Batch-first support (matches our data flow)")
print("   - ✅ Comprehensive mask support")
print("   - ✅ Clear documentation and examples")
print("   - ✅ Compatible with transformers workflow")

print()
print("🥈 2nd Choice: TorchCRF") 
print("   - ✅ Simpler, more lightweight")
print("   - ✅ Good for quick prototyping")
print("   - ❌ Less documentation")
print("   - ❌ Sequence-first (requires data reshaping)")

print()
print("🥉 3rd Choice: Custom Implementation")
print("   - ✅ Full control and customization")
print("   - ✅ No external dependencies")
print("   - ❌ Time-consuming to implement correctly")
print("   - ❌ Potential bugs and edge cases")

print()

# Integration example for v0.3
print("🔧 Integration Code Example (pytorch-crf):")
print("=" * 50)

integration_code = '''
# v0.3 Enhanced Model with pytorch-crf
from pytorch_crf import CRF

class BiLSTMCRFHead(nn.Module):
    def __init__(self, hidden_size=768, lstm_hidden=256, num_labels=3):
        super().__init__()
        
        # BiLSTM
        self.bilstm = nn.LSTM(hidden_size, lstm_hidden, 
                             batch_first=True, bidirectional=True)
        
        # CRF emission layer
        self.emission = nn.Linear(lstm_hidden * 2, num_labels)
        
        # CRF layer (pytorch-crf)
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, sequence_output, attention_mask, labels=None):
        # BiLSTM processing
        lstm_out, _ = self.bilstm(sequence_output)
        
        # CRF emissions
        emissions = self.emission(lstm_out)
        
        if labels is not None:
            # Training: compute loss
            return -self.crf(emissions, labels, mask=attention_mask)
        else:
            # Inference: decode best path
            return self.crf.decode(emissions, mask=attention_mask)
'''

print(integration_code)

print("🎯 Next Steps for v0.3 Implementation:")
print("=" * 50)
print("1. ✅ Install pytorch-crf (completed)")
print("2. 🔄 Update fine_tuning_v3.py to use pytorch-crf")
print("3. 🔄 Create BIO label generation for training data")
print("4. 🔄 Integrate CRF loss with weighted position loss")
print("5. 🔄 Test CRF-enhanced model training")
print("6. 📊 Evaluate End Position accuracy improvement")