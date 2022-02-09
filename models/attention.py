import math
import torch
import torch.nn as nn
import copy


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.word_embeddings_2 = nn.Linear(config.embedding_size, config.hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, input_size, num_attention_heads=None):
        super(SelfAttention, self).__init__()
        self.all_head_size = input_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        else:
            self.num_attention_heads = self.all_head_size
        self.attention_head_size = int(input_size / self.num_attention_heads)
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input, attention_mask=None):
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        mixed_value_layer = self.value(input)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SingleSelfAttention(nn.Module):
    def __init__(self, input_size, num_attention_heads=None):
        super(SingleSelfAttention, self).__init__()
        self.all_head_size = input_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        else:
            self.num_attention_heads = self.all_head_size
        self.attention_head_size = int(input_size / self.num_attention_heads)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input):
        mixed_layer = self.transpose_for_scores(input)
        attention_scores = torch.matmul(mixed_layer, mixed_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, mixed_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return (context_layer, attention_probs)


class LstmAttention(nn.Module):
    def __init__(self, input_size, num_attention_heads=None):
        super(LstmAttention, self).__init__()
        self.all_head_size = input_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        else:
            self.num_attention_heads = self.all_head_size
        self.attention_head_size = int(input_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.LSTM(input_size, input_size)
        self.key = nn.LSTM(input_size, input_size)
        self.value = nn.LSTM(input_size, input_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        mixed_value_layer = self.value(input)
        query_layer = self.transpose_for_scores(mixed_query_layer[0])
        key_layer = self.transpose_for_scores(mixed_key_layer[0])
        value_layer = self.transpose_for_scores(mixed_value_layer[0])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return (context_layer, attention_probs)


class SuperSelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super(SuperSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SuperAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(SuperAttention, self).__init__()
        self.self = SelfAttention(hidden_size,
                                  num_attention_heads)
        self.output = SuperSelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = self.prune_linear_layer(self.self.query, index)
        self.self.key = self.prune_linear_layer(self.self.key, index)
        self.self.value = self.prune_linear_layer(self.self.value, index)
        self.output.dense = self.prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class SuperIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(SuperIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, input_tensor):
        outputs = self.dense(input_tensor)
        return self.gelu(outputs)


class SuperOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super(SuperOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SuperLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads=None):
        super(SuperLayer, self).__init__()
        self.attention = SuperAttention(hidden_size, num_attention_heads)
        self.intermediate = SuperIntermediate(hidden_size, intermediate_size)
        self.output = SuperOutput(intermediate_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output_pre = attention_output
        intermediate_output = self.intermediate(attention_output_pre)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SuperPooler(nn.Module):
    def __init__(self, hidden_size):
        super(SuperPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

     
class SuperEncoder(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads=None):
        super(SuperEncoder, self).__init__()
        layer = SuperLayer(hidden_size, intermediate_size, num_attention_heads)
        self.layer_shared = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])

    def forward(self, hidden_state, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_state)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for layer_module in self.layer_shared:
            hidden_state = layer_module(hidden_state, extended_attention_mask)
        return hidden_state


class DecoderSelfAttention(nn.Module):
    def __init__(self, input_size, num_attention_heads=None):
        super(DecoderSelfAttention, self).__init__()
        self.all_head_size = input_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        else:
            self.num_attention_heads = self.all_head_size
        self.attention_head_size = int(input_size / self.num_attention_heads)
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 2, 3)

    def forward(self, input, attention_mask=None):
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        mixed_value_layer = self.value(input)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_mask = self.transpose_for_scores(attention_mask)
            attention_mask = torch.matmul(attention_mask, attention_mask.transpose(-1, -2))
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 2, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(DecoderAttention, self).__init__()
        self.self = DecoderSelfAttention(hidden_size,
                                         num_attention_heads)
        self.output = SuperSelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = self.prune_linear_layer(self.self.query, index)
        self.self.key = self.prune_linear_layer(self.self.key, index)
        self.self.value = self.prune_linear_layer(self.self.value, index)
        self.output.dense = self.prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor):
        self_outputs = self.self(input_tensor)
        attention_output = self.output(self_outputs[0], input_tensor)
        return (attention_output,) + self_outputs[1:]


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads=None):
        super(DecoderLayer, self).__init__()
        self.attention = DecoderAttention(hidden_size, num_attention_heads)
        self.intermediate = SuperIntermediate(hidden_size, intermediate_size)
        self.output = SuperOutput(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        attention_outputs = self.attention(hidden_states)
        attention_output = attention_outputs[0]
        attention_output_pre = attention_output
        intermediate_output = self.intermediate(attention_output_pre)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,) + attention_outputs[1:]


class SuperDecoder(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads=None):
        super(SuperDecoder, self).__init__()
        layer = DecoderLayer(hidden_size, intermediate_size, num_attention_heads)
        self.layer_shared = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layer_shared:
            layer_output = layer_module(hidden_states)
            hidden_states = layer_output[0]
        return hidden_states


class DualSelfAttention(nn.Module):
    def __init__(self, input_size, input_size2, num_attention_heads=None, num_attention_heads2=None):
        super(DualSelfAttention, self).__init__()
        self.all_head_size = input_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        else:
            self.num_attention_heads = self.all_head_size
        self.attention_head_size = int(input_size / self.num_attention_heads)
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

        self.all_head_size2 = input_size2
        if num_attention_heads2 is not None:
            self.num_attention_heads2 = num_attention_heads2
        else:
            self.num_attention_heads2 = self.all_head_size2
        self.attention_head_size2 = int(input_size2 / self.num_attention_heads2)
        self.query2 = nn.Linear(input_size2, input_size2)
        self.key2 = nn.Linear(input_size2, input_size2)
        self.value2 = nn.Linear(input_size2, input_size2)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 2, 3)

    def transpose_for_scores2(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads2, self.attention_head_size2)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 2, 3)

    def forward(self, context_layer, attention_mask=None):
        context_layer = context_layer.permute(0, 2, 1)
        mixed_query_layer = self.query2(context_layer)
        mixed_key_layer = self.key2(context_layer)
        mixed_value_layer = self.value2(context_layer)
        query_layer = self.transpose_for_scores2(mixed_query_layer)
        key_layer = self.transpose_for_scores2(mixed_key_layer)
        value_layer = self.transpose_for_scores2(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size2)
        if attention_mask is not None:
            attention_mask = self.transpose_for_scores2(attention_mask)
            attention_mask = torch.matmul(attention_mask, attention_mask.transpose(-1, -2))
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 2, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size2,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.permute(0, 2, 1)

        mixed_query_layer = self.query(context_layer)
        mixed_key_layer = self.key(context_layer)
        mixed_value_layer = self.value(context_layer)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 2, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class DualAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(DualAttention, self).__init__()
        self.self = DecoderSelfAttention(hidden_size,
                                      num_attention_heads)
        self.output = SuperSelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = self.prune_linear_layer(self.self.query, index)
        self.self.key = self.prune_linear_layer(self.self.key, index)
        self.self.value = self.prune_linear_layer(self.self.value, index)
        self.output.dense = self.prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs, input_tensor)
        return attention_output


class DecoderSelfAttention2(nn.Module):
    def __init__(self, input_size, num_attention_heads=None):
        super(DecoderSelfAttention2, self).__init__()
        self.all_head_size = input_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        else:
            self.num_attention_heads = self.all_head_size
        self.attention_head_size = int(input_size / self.num_attention_heads)
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input, attention_mask=None):
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        mixed_value_layer = self.value(input)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class DualAttention2(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(DualAttention2, self).__init__()
        self.self = DecoderSelfAttention2(hidden_size,
                                          num_attention_heads)
        self.output = SuperSelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = self.prune_linear_layer(self.self.query, index)
        self.self.key = self.prune_linear_layer(self.self.key, index)
        self.self.value = self.prune_linear_layer(self.self.value, index)
        self.output.dense = self.prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs, input_tensor)
        return attention_output


class DualLayer(nn.Module):
    def __init__(self, hidden_size, hidden_size2, num_attention_heads=None, num_attention_heads2=None):
        super(DualLayer, self).__init__()
        self.attention = DualAttention2(hidden_size, num_attention_heads)
        self.intermediate = SuperIntermediate(hidden_size, hidden_size*4)
        self.output = SuperOutput(hidden_size*4, hidden_size)
        self.attention2 = DualAttention2(hidden_size2, num_attention_heads2)
        self.intermediate2 = SuperIntermediate(hidden_size2, hidden_size2 * 4)
        self.output2 = SuperOutput(hidden_size2 * 4, hidden_size2)

    def forward(self, layer_output, attention_mask=None):
        layer_output = layer_output.permute(0, 2, 1)
        layer_output = self.attention2(layer_output, None)
        intermediate_output = self.intermediate2(layer_output)
        layer_output = self.output2(intermediate_output, layer_output)
        layer_output = layer_output.permute(0, 2, 1)

        layer_output = self.attention(layer_output, attention_mask)
        intermediate_output = self.intermediate(layer_output)
        layer_output = self.output(intermediate_output, layer_output)
        return layer_output


class DualDecoder(nn.Module):
    def __init__(self, hidden_size, hidden_size2,
                 num_hidden_layers, num_attention_heads=None, num_attention_heads2=None):
        super(DualDecoder, self).__init__()
        layer = DualLayer(hidden_size, hidden_size2,
                          num_attention_heads, num_attention_heads2)
        self.layer_shared = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_states)
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for layer_module in self.layer_shared:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states


class DualSelfAttention(nn.Module):
    def __init__(self, input_size, input_size2, num_attention_heads=None, num_attention_heads2=None):
        super(DualSelfAttention, self).__init__()
        self.all_head_size = input_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        else:
            self.num_attention_heads = self.all_head_size
        self.attention_head_size = int(input_size / self.num_attention_heads)
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

        self.all_head_size2 = input_size2
        if num_attention_heads2 is not None:
            self.num_attention_heads2 = num_attention_heads2
        else:
            self.num_attention_heads2 = self.all_head_size2
        self.attention_head_size2 = int(input_size2 / self.num_attention_heads2)
        self.query2 = nn.Linear(input_size2, input_size2)
        self.key2 = nn.Linear(input_size2, input_size2)
        self.value2 = nn.Linear(input_size2, input_size2)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 2, 3)

    def transpose_for_scores2(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads2, self.attention_head_size2)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 2, 3)

    def forward(self, context_layer, attention_mask=None):
        context_layer = context_layer.permute(0, 2, 1)
        mixed_query_layer = self.query2(context_layer)
        mixed_key_layer = self.key2(context_layer)
        mixed_value_layer = self.value2(context_layer)
        query_layer = self.transpose_for_scores2(mixed_query_layer)
        key_layer = self.transpose_for_scores2(mixed_key_layer)
        value_layer = self.transpose_for_scores2(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size2)
        if attention_mask is not None:
            attention_mask = self.transpose_for_scores2(attention_mask)
            attention_mask = torch.matmul(attention_mask, attention_mask.transpose(-1, -2))
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 2, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size2,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.permute(0, 2, 1)

        mixed_query_layer = self.query(context_layer)
        mixed_key_layer = self.key(context_layer)
        mixed_value_layer = self.value(context_layer)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 2, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class DualAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(DualAttention, self).__init__()
        self.self = DecoderSelfAttention(hidden_size,
                                      num_attention_heads)
        self.output = SuperSelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = self.prune_linear_layer(self.self.query, index)
        self.self.key = self.prune_linear_layer(self.self.key, index)
        self.self.value = self.prune_linear_layer(self.self.value, index)
        self.output.dense = self.prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs, input_tensor)
        return attention_output


class BiSelfAttention(nn.Module):
    def __init__(self, input_size, num_attention_heads=None):
        super(BiSelfAttention, self).__init__()
        self.all_head_size = input_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        else:
            self.num_attention_heads = self.all_head_size
        self.attention_head_size = int(input_size / self.num_attention_heads)
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 2, 3)

    def forward(self, input, attention_mask=None):
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        mixed_value_layer = self.value(input)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 1, 2, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BiAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(BiAttention, self).__init__()
        self.self = BiSelfAttention(hidden_size,
                                    num_attention_heads)
        self.output = SuperSelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = self.prune_linear_layer(self.self.query, index)
        self.self.key = self.prune_linear_layer(self.self.key, index)
        self.self.value = self.prune_linear_layer(self.self.value, index)
        self.output.dense = self.prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs, input_tensor)
        return attention_output


class BiLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=None):
        super(BiLayer, self).__init__()
        self.attention = BiAttention(hidden_size, num_attention_heads)
        self.intermediate = SuperIntermediate(hidden_size, hidden_size*4)
        self.output = SuperOutput(hidden_size*4, hidden_size)

    def forward(self, layer_output, attention_mask=None):
        layer_output = self.attention(layer_output, attention_mask)
        intermediate_output = self.intermediate(layer_output)
        layer_output = self.output(intermediate_output, layer_output)
        return layer_output


class BiDecoder(nn.Module):
    def __init__(self, hidden_size, hidden_size2,
                 num_hidden_layers, num_attention_heads=None, num_attention_heads2=None):
        super(BiDecoder, self).__init__()
        layer1 = SuperLayer(hidden_size,
                            hidden_size*4,
                            num_attention_heads)
        self.layer_shared1 = nn.ModuleList([copy.deepcopy(layer1) for _ in range(num_hidden_layers)])
        layer2 = BiLayer(hidden_size2,
                         num_attention_heads2)
        self.layer_shared2 = nn.ModuleList([copy.deepcopy(layer2) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_states)
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states1 = hidden_states.permute(0, 2, 1)
        for layer_module in self.layer_shared2:
            hidden_states1 = layer_module(hidden_states1, None)
        hidden_states1 = hidden_states1.permute(0, 2, 1)
        for layer_module in self.layer_shared1:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        hidden_states = torch.cat([hidden_states, hidden_states1], dim=-1)
        return hidden_states


class SuperDualDecoder(nn.Module):
    def __init__(self, hidden_size, hidden_size2,
                 num_hidden_layers, num_attention_heads=None, num_attention_heads2=None):
        super(SuperDualDecoder, self).__init__()
        layer1 = SuperLayer(hidden_size, hidden_size*4, num_attention_heads)
        self.layer_shared1 = nn.ModuleList([copy.deepcopy(layer1) for _ in range(num_hidden_layers)])
        layer2 = SuperLayer(hidden_size2, hidden_size2*4, num_attention_heads2)
        self.layer_shared2 = nn.ModuleList([copy.deepcopy(layer2) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states.permute(0, 2, 1)
        for layer_module in self.layer_shared2:
            layer_output = layer_module(hidden_states)
            hidden_states = layer_output[0]
        hidden_states = hidden_states.permute(0, 2, 1)
        for layer_module in self.layer_shared1:
            layer_output = layer_module(hidden_states)
            hidden_states = layer_output[0]
        return hidden_states


class NestSelfAttention(nn.Module):
    def __init__(self, input_size, num_attention_heads=None):
        super(NestSelfAttention, self).__init__()
        self.all_head_size = input_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        else:
            self.num_attention_heads = self.all_head_size
        self.attention_head_size = int(input_size / self.num_attention_heads)
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input, attention_mask=None):
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        mixed_value_layer = self.value(input)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class NestAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(NestAttention, self).__init__()
        self.self = NestSelfAttention(hidden_size,
                                  num_attention_heads)
        self.output = SuperSelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = self.prune_linear_layer(self.self.query, index)
        self.self.key = self.prune_linear_layer(self.self.key, index)
        self.self.value = self.prune_linear_layer(self.self.value, index)
        self.dense = self.prune_linear_layer(self.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class NestLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads=None):
        super(NestLayer, self).__init__()
        self.attention = NestAttention(hidden_size, num_attention_heads)
        self.intermediate = SuperIntermediate(hidden_size, intermediate_size)
        self.output = SuperOutput(intermediate_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class NestEncoder(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads=None):
        super(NestEncoder, self).__init__()
        layer = NestLayer(hidden_size, intermediate_size, num_attention_heads)
        self.layer_shared = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_states)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        for layer_module in self.layer_shared:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states


class LstmAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(LstmAttention, self).__init__()
        self.self = LstmAttention(hidden_size,
                                  num_attention_heads)
        self.output = SuperSelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        self.self.query = self.prune_linear_layer(self.self.query, index)
        self.self.key = self.prune_linear_layer(self.self.key, index)
        self.self.value = self.prune_linear_layer(self.self.value, index)
        self.dense = self.prune_linear_layer(self.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        return (attention_output,) + self_outputs[1:]


class LstmLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads=None):
        super(LstmLayer, self).__init__()
        self.attention = LstmAttention(hidden_size, num_attention_heads)
        self.intermediate = SuperIntermediate(hidden_size, intermediate_size)
        self.output = SuperOutput(intermediate_size, hidden_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        attention_output_pre = attention_output
        intermediate_output = self.intermediate(attention_output_pre)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,) + attention_outputs[1:]


class LstmEncoder(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads=None):
        super(LstmEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layer_shared = LstmLayer(hidden_size, intermediate_size, num_attention_heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_states)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.num_hidden_layers
        for i in range(self.num_hidden_layers):
            layer_output = self.layer_shared(hidden_states, extended_attention_mask, head_mask[i])
            hidden_states = layer_output[0]
        return hidden_states


try:
    from albert_master.modeling_albert_bright import AlbertPreTrainedModel


    class SuperAlbert(AlbertPreTrainedModel):
        def __init__(self, config):
            super(SuperAlbert, self).__init__(config)
            self.embeddings = Embedding(config)
            self.encoder = SuperEncoder(config.hidden_size,
                                        config.num_hidden_layers,
                                        config.intermediate_size,
                                        config.num_attention_heads)
            self.init_weights()

        def _resize_token_embeddings(self, new_num_tokens):
            old_embeddings = self.embeddings.word_embeddings
            new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
            self.embeddings.word_embeddings = new_embeddings
            return self.embeddings.word_embeddings

        def _prune_heads(self, heads_to_prune):
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        def forward(self, x, attention_mask=None):
            x = self.embeddings(x, None, torch.zeros_like(x))
            x = self.encoder(x, attention_mask, None)
            return x


    class NestAlbert(AlbertPreTrainedModel):
        def __init__(self, config):
            super(NestAlbert, self).__init__(config)
            self.embeddings = Embedding(config)
            self.encoder = NestEncoder(config.hidden_size,
                                       config.num_hidden_layers,
                                       config.intermediate_size,
                                       config.num_attention_heads)
            self.init_weights()

        def _resize_token_embeddings(self, new_num_tokens):
            old_embeddings = self.embeddings.word_embeddings
            new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
            self.embeddings.word_embeddings = new_embeddings
            return self.embeddings.word_embeddings

        def _prune_heads(self, heads_to_prune):
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        def forward(self, x, attention_mask=None):
            x = self.embeddings(x, None, torch.zeros_like(x))
            x = self.encoder(x, attention_mask, None)
            return x


    class LstmAlbert(AlbertPreTrainedModel):
        def __init__(self, config):
            super(LstmAlbert, self).__init__(config)
            self.embeddings = Embedding(config)
            self.encoder = LstmEncoder(config.hidden_size,
                                       config.num_hidden_layers,
                                       config.intermediate_size,
                                       config.num_attention_heads)
            self.init_weights()

        def _resize_token_embeddings(self, new_num_tokens):
            old_embeddings = self.embeddings.word_embeddings
            new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
            self.embeddings.word_embeddings = new_embeddings
            return self.embeddings.word_embeddings

        def _prune_heads(self, heads_to_prune):
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        def forward(self, x, attention_mask=None):
            x = self.embeddings(x, None, torch.zeros_like(x))
            x = self.encoder(x, attention_mask, None)
            return x


except ImportError:
    class SuperAlbert(nn.Module):
        def __init__(self, config):
            super(SuperAlbert, self).__init__(config)
            self.embeddings = Embedding(config)
            self.encoder = SuperEncoder(config.hidden_size,
                                        config.num_hidden_layers,
                                        config.intermediate_size,
                                        config.num_attention_heads)

        def _resize_token_embeddings(self, new_num_tokens):
            old_embeddings = self.embeddings.word_embeddings
            new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
            self.embeddings.word_embeddings = new_embeddings
            return self.embeddings.word_embeddings

        def _prune_heads(self, heads_to_prune):
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        def forward(self, x, attention_mask=None):
            x = self.embeddings(x, None, torch.zeros_like(x))
            x = self.encoder(x, attention_mask, None)
            return x


    class NestAlbert(nn.Module):
        def __init__(self, config):
            super(NestAlbert, self).__init__(config)
            self.embeddings = Embedding(config)
            self.encoder = NestEncoder(config.hidden_size,
                                       config.num_hidden_layers,
                                       config.intermediate_size,
                                       config.num_attention_heads)

        def _resize_token_embeddings(self, new_num_tokens):
            old_embeddings = self.embeddings.word_embeddings
            new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
            self.embeddings.word_embeddings = new_embeddings
            return self.embeddings.word_embeddings

        def _prune_heads(self, heads_to_prune):
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        def forward(self, x, attention_mask=None):
            x = self.embeddings(x, None, torch.zeros_like(x))
            x = self.encoder(x, attention_mask, None)
            return x


    class LstmAlbert(nn.Module):
        def __init__(self, config):
            super(LstmAlbert, self).__init__(config)
            self.embeddings = Embedding(config)
            self.encoder = LstmEncoder(config.hidden_size,
                                       config.num_hidden_layers,
                                       config.intermediate_size,
                                       config.num_attention_heads)

        def _resize_token_embeddings(self, new_num_tokens):
            old_embeddings = self.embeddings.word_embeddings
            new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
            self.embeddings.word_embeddings = new_embeddings
            return self.embeddings.word_embeddings

        def _prune_heads(self, heads_to_prune):
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        def forward(self, x, attention_mask=None):
            x = self.embeddings(x, None, torch.zeros_like(x))
            x = self.encoder(x, attention_mask, None)
            return x