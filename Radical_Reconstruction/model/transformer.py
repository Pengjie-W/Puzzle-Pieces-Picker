import copy
import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, 
                 dropout, normalize_before, pad_token_id, num_classes, max_position_embeddings, 
                 return_intermediate_dec, eos_index, activation="relu",beam_search=False,beam_size=100,beam_search_max_length=24,use_length_penalty=False,length_penalty=0.7):
        super(Transformer, self).__init__()
        self.embedding = DecoderEmbeddings(num_classes, d_model, pad_token_id, max_position_embeddings, dropout)
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.nhead = nhead
        self.d_model = d_model
        self.eos_index = eos_index
        self.pad_token_id = pad_token_id
        self.num_encoder_layers = num_encoder_layers
        self.max_position_embeddings = max_position_embeddings
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.beam_search_max_length = beam_search_max_length
        self.use_length_penalty = use_length_penalty
        self.length_penalty = length_penalty

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed, seq, vocab_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed.half())
        else:
            memory = src

        query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, bs, 1)
        if self.training:
            tgt = self.embedding(seq).permute(1, 0, 2)
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed[:len(tgt)],
                          tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
            # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
            return vocab_embed(hs[-1].transpose(0, 1))
        else:
            if self.beam_search==False:
                probs = []
                for i in range(self.max_position_embeddings):
                    tgt = self.embedding(seq).permute(1, 0, 2)
                    hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                            pos=pos_embed, query_pos=query_embed[:len(tgt)],
                            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
                    out = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])
                    out = out.softmax(-1)

                    prob, extra_seq = out.topk(dim=-1, k=1)
                    seq = torch.cat([seq, extra_seq], dim=-1)
                    probs.append(prob)                
                    if extra_seq[0] == self.eos_index or i>24:
                        break
                
                seq = seq[:, 1:] # remove start index
                return seq, torch.cat(probs, dim=-1)
            # The block below is only used during testing and is newly added.
            # The paper uses the previous (greedy) decoding for evaluation.
            else:
                num_beams= self.beam_size
                max_length = self.beam_search_max_length
                length_penalty = self.length_penalty
                use_length_penalty = self.use_length_penalty
                batch_size=1
                generated_hyps = [
                    BeamHypotheses(num_beams, max_length, length_penalty, use_length_penalty)
                    for _ in range(batch_size)
                ]
                # Scores for each beam container; total of batch_size * num_beams
                beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=seq.device)
                beam_scores = beam_scores.view(-1)

                # Whether each sample has finished generation; total of batch_size flags
                done = [False for _ in range(batch_size)]
                # input_ids = seq.expand(num_beams, -1)  # [beam_size, seq_len]
                input_ids=seq
                for cur_len in range(1,self.beam_search_max_length+1):
                    if cur_len==1:
                        tgt = self.embedding(seq).permute(1, 0, 2)
                        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                pos=pos_embed, query_pos=query_embed[:len(tgt)],
                                tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
                        logits = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])
                        log_probs = F.log_softmax(logits, dim=-1)  # use log-probs for easy accumulation; [beam_size, vocab_size]
                        vocab_size=log_probs.size(-1)
                        next_scores = log_probs
                        next_scores = next_scores.view(
                            batch_size,  vocab_size
                        )  # (batch_size, vocab_size)
                    else:
                        tgt = self.embedding(input_ids).permute(1, 0, 2)  # [seq_len, beam_size, embed_dim]
                        hs = self.decoder(
                            tgt, memory.repeat(1, num_beams, 1),  # expand memory to match beam_size
                            memory_key_padding_mask=mask.repeat(num_beams, 1),
                            pos=pos_embed.repeat(1, num_beams, 1),
                            query_pos=query_embed[:len(tgt)].repeat(1, num_beams, 1),
                            tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device)
                        )  # [seq_len, beam_size, embed_dim]
                        logits = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])
                        log_probs = F.log_softmax(logits, dim=-1)  # use log-probs for easy accumulation; [beam_size, vocab_size]
                        vocab_size=log_probs.size(-1)
                        next_scores = log_probs + beam_scores[:, None].expand_as(log_probs) 
                        next_scores = next_scores.view(
                                batch_size, num_beams * vocab_size
                            )  # (batch_size, num_beams * vocab_size)
                        
                    next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=
                                                        True, sorted=True)
                    # Beam list for the entire batch at the next time step.
                    # Each element is a triplet: (score, token_id, beam_id)
                    next_batch_beam = []

                    # Expand beams for each sample
                    for batch_idx in range(batch_size):
                        # If this sample has already finished generation
                        if done[batch_idx]:
                            # For finished sentences, pad with pad token
                            next_batch_beam.extend([(0, self.pad_token_id, 0)] * num_beams)
                            continue
                        # Next-step beam list for the current sample
                        # We need the top num_beams expansions.
                        # next_scores and next_tokens are aligned and already sorted.
                        next_sent_beam = []
                        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                            zip(next_tokens[batch_idx], next_scores[batch_idx])
                        ):
                            # get beam and word IDs
                            beam_id = beam_token_id // vocab_size 
                            token_id = beam_token_id % vocab_size
                            effective_beam_id = batch_idx * num_beams + beam_id

                            # If EOS appears, we have a complete hypothesis
                            if (self.eos_index is not None) and (token_id.item() == self.eos_index):
                                # If this token is worse than the top num_beams, skip adding it
                                is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                                if is_beam_token_worse_than_top_num_beams:
                                    continue
                                # Add this sequence to the hypotheses container
                                generated_hyps[batch_idx].add(
                                    input_ids[effective_beam_id].clone(), beam_token_score.item(),
                                )
                            else:
                                # Add next predicted token if it is not EOS
                                next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                            # Stop once we have num_beams expansions
                            if len(next_sent_beam) == num_beams:
                                break

                        # Check if this sample is done. Two cases:
                        # 1) we've already marked it done; 2) new results don't improve the best score.
                        done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                            next_scores[batch_idx].max().item(), cur_len=cur_len
                        )

                        # Append current sample results to the batch results
                        next_batch_beam.extend(next_sent_beam)

                    # If all samples are done, we can exit early
                    if all(done):
                        break
                    
                    # Convert the triplet list back to separate lists
                    beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
                    beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
                    beam_idx = input_ids.new([x[2] for x in next_batch_beam])

                    # Prepare decoder input for the next step
                    # Select the actually expanded beams
                    input_ids = input_ids[beam_idx, :]
                    # Append the newly generated tokens to those beams
                    input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
                # Select the best hypotheses as the final output
                # Number of sequences to return per sample
                output_num_return_sequences_per_batch = 100
                # Record the length of each returned sentence for padding
                output_batch_size=output_num_return_sequences_per_batch*batch_size
                sent_lengths = input_ids.new(output_batch_size)
                best_sequences = []
                best_scores = []
                # For each sample, take the best output_num_return_sequences_per_batch sequences
                for i, hypotheses in enumerate(generated_hyps):
                    sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
                    for j in range(output_num_return_sequences_per_batch):
                        effective_batch_idx = output_num_return_sequences_per_batch * i + j
                        # best_hyp = sorted_hyps.pop()[1]
                        score,best_hyp = sorted_hyps.pop()
                        sent_lengths[effective_batch_idx] = len(best_hyp)
                        best_sequences.append(best_hyp)
                        best_scores.append(score)

                # If sequences have different lengths, pad them to the same length
                if sent_lengths.min().item() != sent_lengths.max().item():
                    sent_max_len = min(sent_lengths.max().item() + 1, max_length)
                    # Fill the output matrix with PAD tokens
                    decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.pad_token_id)

                    # Write actual content
                    for i, hypo in enumerate(best_sequences):
                        decoded[i, : sent_lengths[i]] = hypo
                        # Fill EOS token
                        if sent_lengths[i] < max_length:
                            decoded[i, sent_lengths[i]] = self.eos_index
                else:
                    # If none of the sequences finished with EOS, just stack them
                    decoded = torch.stack(best_sequences).type(torch.long).to(next(self.parameters()).device)
                # Convert scores back to probabilities (from log-probabilities)
                best_probs = torch.tensor(best_scores, device=decoded.device).exp()
                # Returned results include the BOS token
                return decoded, best_probs

class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
        super(DecoderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

        self.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    max_position_embeddings = (2 + 25) * args.max_num_text_ins + 1
    return Transformer(
        d_model=args.tfm_hidden_dim,
        nhead=args.tfm_nheads,
        num_encoder_layers=args.tfm_enc_layers,
        num_decoder_layers=args.tfm_dec_layers,
        dim_feedforward=args.tfm_dim_feedforward,
        dropout=args.tfm_dropout,
        normalize_before=args.tfm_pre_norm,
        pad_token_id=args.padding_index,
        num_classes=args.num_classes,
        max_position_embeddings=max_position_embeddings,
        return_intermediate_dec=False,
        eos_index=args.eos_index,
        beam_search=args.beam_search,
        beam_size=args.beam_size,
        beam_search_max_length=args.beam_search_max_length,
        use_length_penalty=args.use_length_penalty,
        length_penalty=args.length_penalty,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, use_length_penalty):
        """
        Initialize an n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
        self.length_penalty= length_penalty
        self.use_length_penalty = use_length_penalty

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        if self.use_length_penalty:
            score = sum_logprobs / len(hyp) ** self.length_penalty
        else:
            score = sum_logprobs
        if len(self) < self.num_beams or score > self.worst_score:
            # Updatable: either not saturated yet, or better than the current worst score
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                # If saturated, remove the worst one
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        Whether the sample is finished.
        best_sum_logprobs is the highest score among new candidate sequences.
        """
        if len(self) < self.num_beams:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_length
            if self.use_length_penalty:
                cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            else:
                cur_score = best_sum_logprobs
            # Done if the best possible score is still worse than the current worst saved score
            ret = self.worst_score >= cur_score
            return ret