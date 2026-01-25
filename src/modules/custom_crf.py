"""
Custom CRF Implementation for v0.3 End Position Enhancement
シンプルで効果的なConditional Random Field実装

依存関係の問題を避けるために、PyTorchのみを使用したCRF実装
End Position精度向上に特化した設計
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleCRF(nn.Module):
    """
    Simple CRF implementation optimized for Question Answering tasks
    
    Features:
    - Batch-first processing
    - Mask support for variable-length sequences
    - Viterbi decoding
    - Compatible with transformers workflow
    """
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        """
        Args:
            num_tags: Number of tags (e.g., 3 for O, B-ANSWER, I-ANSWER)
            batch_first: Whether input is batch-first (True) or seq-first (False)
        """
        super().__init__()
        
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition parameters: transitions[i][j] is transition score from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Special tags for start and end
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        # Initialize transitions
        self._init_transitions()
    
    def _init_transitions(self):
        """Initialize transition parameters with reasonable values"""
        # Small random initialization
        nn.init.xavier_normal_(self.transitions)
        nn.init.xavier_normal_(self.start_transitions.unsqueeze(0))
        nn.init.xavier_normal_(self.end_transitions.unsqueeze(0))
    
    def forward(self, 
                emissions: torch.Tensor, 
                tags: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
        """
        Compute CRF negative log-likelihood loss
        
        Args:
            emissions: [batch_size, seq_len, num_tags] emission scores
            tags: [batch_size, seq_len] ground truth tags
            mask: [batch_size, seq_len] mask for variable length sequences
            reduction: 'none', 'sum', or 'mean'
        
        Returns:
            Negative log-likelihood loss
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)  # seq_len x batch_size x num_tags
            tags = tags.transpose(0, 1)  # seq_len x batch_size
            if mask is not None:
                mask = mask.transpose(0, 1)  # seq_len x batch_size
        
        batch_size, seq_len = tags.shape
        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # Compute log partition function (normalizing constant)
        log_partition = self._compute_log_partition(emissions, mask)
        
        # Compute log probability of the gold sequence
        log_prob = self._compute_sequence_log_prob(emissions, tags, mask)
        
        # Loss is negative log-likelihood
        loss = log_partition - log_prob
        
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        else:  # mean
            return loss.mean()
    
    def _compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute log partition function using forward algorithm
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize forward variables
        # alpha[t][tag] = log probability of being in state 'tag' at time t
        alpha = emissions[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_tags]
        
        for t in range(1, seq_len):
            # Expand alpha for transition computation
            alpha_expanded = alpha.unsqueeze(2)  # [batch_size, num_tags, 1]
            transitions_expanded = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            emissions_t = emissions[:, t].unsqueeze(1)  # [batch_size, 1, num_tags]
            
            # Compute forward scores for time t
            next_alpha = alpha_expanded + transitions_expanded + emissions_t
            next_alpha = torch.logsumexp(next_alpha, dim=1)  # [batch_size, num_tags]
            
            # Apply mask
            mask_t = mask[:, t].unsqueeze(1)  # [batch_size, 1]
            alpha = torch.where(mask_t, next_alpha, alpha)
        
        # Add end transitions
        final_alpha = alpha + self.end_transitions.unsqueeze(0)
        
        # Sum over all final states
        log_partition = torch.logsumexp(final_alpha, dim=1)  # [batch_size]
        
        return log_partition
    
    def _compute_sequence_log_prob(self, 
                                  emissions: torch.Tensor, 
                                  tags: torch.Tensor, 
                                  mask: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of a specific tag sequence
        """
        batch_size, seq_len = tags.shape
        
        # Start transition scores
        log_prob = self.start_transitions[tags[:, 0]]  # [batch_size]
        
        # Emission scores for first timestep
        log_prob = log_prob + emissions[torch.arange(batch_size), 0, tags[:, 0]]
        
        # Transition and emission scores for remaining timesteps
        for t in range(1, seq_len):
            # Transition scores
            transition_scores = self.transitions[tags[:, t-1], tags[:, t]]
            
            # Emission scores
            emission_scores = emissions[torch.arange(batch_size), t, tags[:, t]]
            
            # Add scores with mask
            mask_t = mask[:, t]
            step_scores = transition_scores + emission_scores
            log_prob = log_prob + step_scores * mask_t
        
        # End transition scores (only for sequences that end)
        # Find the last valid position for each sequence
        seq_ends = mask.long().sum(dim=1) - 1  # [batch_size]
        end_tags = tags[torch.arange(batch_size), seq_ends]
        end_scores = self.end_transitions[end_tags]
        log_prob = log_prob + end_scores
        
        return log_prob
    
    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Viterbi decoding to find the best tag sequence
        
        Args:
            emissions: [batch_size, seq_len, num_tags] emission scores
            mask: [batch_size, seq_len] mask for variable length sequences
        
        Returns:
            List of decoded tag sequences for each batch item
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        batch_size, seq_len, num_tags = emissions.shape
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=emissions.device)
        
        # Find sequence lengths
        seq_lengths = mask.long().sum(dim=1)  # [batch_size]
        
        best_sequences = []
        
        for b in range(batch_size):
            seq_len_b = seq_lengths[b].item()
            if seq_len_b == 0:
                best_sequences.append([])
                continue
            
            # Viterbi variables
            viterbi_vars = emissions[b, 0] + self.start_transitions  # [num_tags]
            viterbi_path = []
            
            # Forward pass
            for t in range(1, seq_len_b):
                next_viterbi_vars = []
                viterbi_path_t = []
                
                for next_tag in range(num_tags):
                    # Compute scores for transitioning to next_tag
                    transition_scores = viterbi_vars + self.transitions[:, next_tag]
                    best_prev_tag = torch.argmax(transition_scores)
                    next_viterbi_vars.append(transition_scores[best_prev_tag] + emissions[b, t, next_tag])
                    viterbi_path_t.append(best_prev_tag.item())
                
                viterbi_vars = torch.stack(next_viterbi_vars)
                viterbi_path.append(viterbi_path_t)
            
            # Add end transitions and find best final tag
            final_scores = viterbi_vars + self.end_transitions
            best_final_tag = torch.argmax(final_scores).item()
            
            # Backward pass to construct best path
            best_path = [best_final_tag]
            for t in range(seq_len_b - 2, -1, -1):
                best_final_tag = viterbi_path[t][best_final_tag]
                best_path.append(best_final_tag)
            
            best_path.reverse()
            best_sequences.append(best_path)
        
        return best_sequences


class BiLSTMCRFHead(nn.Module):
    """
    BiLSTM + Custom CRF Head for enhanced span boundary detection
    
    BERT出力 → BiLSTM → CRF → Position Predictions
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 lstm_hidden_size: int = 256,
                 num_labels: int = 3,  # O, B-ANSWER, I-ANSWER
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_labels = num_labels
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # CRF emission layer
        self.emission = nn.Linear(lstm_hidden_size * 2, num_labels)
        
        # Custom CRF layer
        self.crf = SimpleCRF(num_labels, batch_first=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Traditional QA heads for compatibility
        self.qa_outputs = nn.Linear(lstm_hidden_size * 2, 2)  # start/end logits
        
    def forward(self, sequence_output, attention_mask=None, crf_labels=None):
        """
        Args:
            sequence_output: BERT output [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            crf_labels: [batch_size, seq_len] BIO labels for CRF training
        
        Returns:
            start_logits, end_logits, crf_loss (if crf_labels provided)
        """
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # BiLSTM processing
        lstm_output, _ = self.bilstm(sequence_output)  # [batch, seq_len, lstm_hidden*2]
        lstm_output = self.dropout(lstm_output)
        
        # CRF emissions
        emissions = self.emission(lstm_output)  # [batch, seq_len, num_labels]
        
        # Traditional QA logits
        logits = self.qa_outputs(lstm_output)  # [batch, seq_len, 2]
        start_logits = logits[:, :, 0]  # [batch, seq_len]
        end_logits = logits[:, :, 1]    # [batch, seq_len]
        
        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'crf_emissions': emissions
        }
        
        # CRF loss computation during training
        if crf_labels is not None:
            crf_loss = self.crf(emissions, crf_labels, mask=attention_mask)
            outputs['crf_loss'] = crf_loss
        
        return outputs
    
    def crf_decode(self, emissions, mask=None):
        """CRF decoding to get best label sequence"""
        return self.crf.decode(emissions, mask=mask)


def create_bio_labels(context: str, answer: str, tokenizer, max_length: int = 512) -> List[int]:
    """
    Create BIO labels for CRF training
    
    Args:
        context: Context text
        answer: Answer text within context
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
    
    Returns:
        BIO labels: 0=O (Outside), 1=B-ANSWER (Begin), 2=I-ANSWER (Inside)
    """
    # Tokenize context
    encoding = tokenizer(context, max_length=max_length, truncation=True, return_offsets_mapping=True)
    
    # Find answer span in context
    answer_start = context.find(answer)
    if answer_start == -1:
        # Answer not found, return all O labels
        return [0] * len(encoding['input_ids'])
    
    answer_end = answer_start + len(answer)
    
    # Create BIO labels based on token offsets
    labels = []
    answer_started = False
    
    for i, (start_offset, end_offset) in enumerate(encoding.offset_mapping):
        if start_offset is None or end_offset is None:
            # Special tokens
            labels.append(0)  # O
        elif start_offset >= answer_start and end_offset <= answer_end:
            # Token is within answer span
            if not answer_started:
                labels.append(1)  # B-ANSWER
                answer_started = True
            else:
                labels.append(2)  # I-ANSWER
        else:
            # Token is outside answer span
            labels.append(0)  # O
            answer_started = False
    
    # Pad or truncate to max_length
    while len(labels) < max_length:
        labels.append(0)
    labels = labels[:max_length]
    
    return labels


def test_custom_crf():
    """Test the custom CRF implementation"""
    print("🧪 Testing Custom CRF Implementation")
    print("=" * 50)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    num_tags = 3
    
    # Create test data
    emissions = torch.randn(batch_size, seq_len, num_tags)
    tags = torch.randint(0, num_tags, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Initialize CRF
    crf = SimpleCRF(num_tags, batch_first=True)
    
    print(f"✅ CRF initialized with {num_tags} tags")
    
    # Test forward pass (loss computation)
    loss = crf(emissions, tags, mask)
    print(f"✅ Forward pass successful, loss: {loss.item():.4f}")
    
    # Test decoding
    decoded = crf.decode(emissions, mask)
    print(f"✅ Decoding successful, decoded length: {len(decoded[0])}")
    print(f"   Sample decoded sequence: {decoded[0][:5]}...")
    
    # Test BiLSTM + CRF head
    bilstm_crf = BiLSTMCRFHead(hidden_size=768, lstm_hidden_size=256, num_labels=num_tags)
    
    # Test input
    bert_output = torch.randn(batch_size, seq_len, 768)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    crf_labels = torch.randint(0, num_tags, (batch_size, seq_len))
    
    # Forward pass
    outputs = bilstm_crf(bert_output, attention_mask, crf_labels)
    
    print(f"✅ BiLSTM+CRF Head test successful")
    print(f"   Start logits shape: {outputs['start_logits'].shape}")
    print(f"   End logits shape: {outputs['end_logits'].shape}")
    print(f"   CRF loss: {outputs['crf_loss'].item():.4f}")
    
    print()
    print("🎯 Custom CRF Implementation Ready for v0.3!")
    print("   - No external dependencies ✅")
    print("   - Batch-first processing ✅")
    print("   - Mask support ✅")
    print("   - Viterbi decoding ✅")
    print("   - BiLSTM integration ✅")


if __name__ == '__main__':
    test_custom_crf()