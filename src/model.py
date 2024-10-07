import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass



@dataclass
class Configuration:
    n_encoder_layers: int = 8
    n_decoder_layers: int = 16
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_length: int = 1500
    vocab_size: int = 51876
    n_conv_channels: list = (80, 256, 512, 1024)
    dropout: float = 0.1

class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class EncoderLayer(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.Q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.V = nn.Linear(config.d_model, config.d_model, bias=False)
        self.K = nn.Linear(config.d_model, config.d_model, bias=False)
        self.Out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_layer_norm = nn.LayerNorm(config.d_model, eps=1e-4)
        self.activation = nn.SiLU()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.ffn_layer_norm = nn.LayerNorm(config.d_model, eps=1e-4)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        q, k, v = self.Q(x), self.K(x), self.V(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(x.size(-1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.dropout(self.Out(attn_output))
        x = self.attn_layer_norm(x + attn_output)

        # Feed-forward network
        ffn_output = self.fc2(self.dropout(self.activation(self.fc1(x))))
        x = self.ffn_layer_norm(x + ffn_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout)
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.n_heads, dropout=config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention Link 
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, key_padding_mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward network
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class SONATA_Encoder(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.conv1 = nn.Conv1d(config.n_conv_channels[0], config.n_conv_channels[1], 3, 1, 1)
        self.conv2 = nn.Conv1d(config.n_conv_channels[1], config.n_conv_channels[2], 3, 1, 1)
        self.conv3 = nn.Conv1d(config.n_conv_channels[2], config.n_conv_channels[3], 3, 2, 1)
        self.projection = nn.Linear(config.n_conv_channels[3], config.d_model)
        self.pos_embed = PositionalEncoding(config.max_seq_length, config.d_model)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)

class SONATA_Decoder(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = PositionalEncoding(config.max_seq_length, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-4)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embed_tokens(x)
        x = self.pos_embed(x)
        x = x.transpose(0, 1)  # Change to (seq_len, batch_size, d_model)
        encoder_output = encoder_output.transpose(0, 1)  # Change to (seq_len, batch_size, d_model)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        x = x.transpose(0, 1)  # Change back to (batch_size, seq_len, d_model)
        return self.layer_norm(x)

class SONATA(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        self.Encoder = SONATA_Encoder(self.config)
        self.Decoder = SONATA_Decoder(self.config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        encoder_output = self.Encoder(src)
        decoder_output = self.Decoder(tgt, encoder_output, tgt_mask, src_mask)
        return self.proj_out(decoder_output)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

 
config = Configuration()
model = SONATA(config)
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device) 


print(f"The model has {count_parameters(model):,} trainable parameters")

# Dummy input
src = torch.randn(32, 80, 1000)  # (batch_size, features, time_steps)
tgt = torch.randint(0, config.vocab_size, (32, 100))  # (batch_size, seq_len)
src, tgt = src.to(device), tgt.to(device)
output = model(src=src, tgt=tgt)
print(output.shape)  # Should be (32, 100, 51876)










To address the problem statement using Neo4j, we'll need to utilize various pathfinding algorithms, as mentioned in the procedure. Here's a breakdown of how each of the tasks can be tackled using Neo4j's query language, Cypher.

### 1. **Find the shortest path between 'TomC' and 'ReneeZ' using the shortest Path function.**

```cypher
MATCH (start:Person {name: 'TomC'}), (end:Person {name: 'ReneeZ'})
CALL algo.shortestPath.stream(start, end, null)
YIELD nodeId, cost
RETURN algo.getNodeById(nodeId).name AS node, cost
```

### 2. **Find the shortest path between 'ReginaK' and 'JohnC' and calculate the weight of the path using the A* algorithm.**

```cypher
MATCH (start:Person {name: 'ReginaK'}), (end:Person {name: 'JohnC'})
CALL algo.shortestPath.stream(start, end, null, {algorithm: 'astar'})
YIELD nodeId, cost
RETURN algo.getNodeById(nodeId).name AS node, cost
```

### 3. **Find all possible paths between 'JackN' and 'Helen' with any relationship type.**

```cypher
MATCH p = (start:Person {name: 'JackN'})-[*]-(end:Person {name: 'Helen'})
RETURN p
```

### 4. **Find all shortest paths between 'GregK' and 'CubaI' where the path only includes 'ACTED_IN' relationships.**

```cypher
MATCH p = allShortestPaths((start:Person {name: 'GregK'})-[:ACTED_IN*]-(end:Person {name: 'CubaI'}))
RETURN p
```

### 5. **Find the shortest path from 'TomC' to 'JamesC' with a maximum depth of 5 relationships.**

```cypher
MATCH p = shortestPath((start:Person {name: 'TomC'})-[*..5]-(end:Person {name: 'JamesC'}))
RETURN p
```

### 6. **Find the shortest path between 'EthanH' and 'JohnC' with a maximum of 3 relationships.**

```cypher
MATCH p = shortestPath((start:Person {name: 'EthanH'})-[*..3]-(end:Person {name: 'JohnC'}))
RETURN p
```

### 7. **Perform a random walk starting from 'GregK' and traverse up to 3 relationships, returning the path.**

```cypher
MATCH (start:Person {name: 'GregK'})
CALL algo.randomWalk.stream(start, 3)
YIELD node
RETURN node
```

### 8. **Perform a random walk starting from 'TomH' with a constraint to only include 'ACTED_IN' relationships, traversing up to 5 steps.**

```cypher
MATCH (start:Person {name: 'TomH'})
CALL algo.randomWalk.stream(start, 5, {relationshipQuery: 'ACTED_IN'})
YIELD node
RETURN node
```

### 9. **Perform a spanning tree DFS starting from 'ReneeZ' and include relationships up to a depth of 5.**

```cypher
MATCH p = (start:Person {name: 'ReneeZ'})-[:*..5]->(end)
RETURN p
```

### 10. **Perform a DFS starting from 'Nathan' and expand up to a maximum of 5 levels, returning the paths found.**

```cypher
MATCH p = (start:Person {name: 'Nathan'})-[:*..5]->(end)
RETURN p
```

### 11. **Perform a BFS starting from 'BillyC' and expand up to a maximum of 2 levels, returning the paths.**

```cypher
MATCH p = (start:Person {name: 'BillyC'})-[:*..2]->(end)
RETURN p
```

### 12. **Perform a BFS starting from 'TomH' with a maximum of 3 levels of expansion, returning the paths.**

```cypher
MATCH p = (start:Person {name: 'TomH'})-[:*..3]->(end)
RETURN p
```

### 13. **Generate a minimum spanning tree for nodes labeled 'Person' using the 'ACTED_IN' relationship and return the edges and weights.**

```cypher
CALL algo.spanningTree('Person', 'ACTED_IN')
YIELD node1, node2, weight
RETURN algo.getNodeById(node1).name AS from, algo.getNodeById(node2).name AS to, weight
```

### 14. **Generate a minimum spanning tree for nodes labeled 'Movie' using the 'DIRECTED_IN' relationship and return the edges and weights.**

```cypher
CALL algo.spanningTree('Movie', 'DIRECTED_IN')
YIELD node1, node2, weight
RETURN algo.getNodeById(node1).name AS from, algo.getNodeById(node2).name AS to, weight
```

### 15. **Find the shortest path between 'The Matrix' and 'That Thing You Do' using any relationship type.**

```cypher
MATCH p = shortestPath((start:Movie {title: 'The Matrix'})-[*]-(end:Movie {title: 'That Thing You Do'}))
RETURN p
```

### 16. **Find the shortest path from 'The Replacements' to 'Orlando' using the 'ACTED_IN' relationship.**

```cypher
MATCH p = shortestPath((start:Movie {title: 'The Replacements'})-[:ACTED_IN*]-(end:Movie {title: 'Orlando'}))
RETURN p
```

### 17. **Find the shortest path between 'The Replacements' and 'The Birdcage' based on the 'REVIEWED' relationship.**

```cypher
MATCH p = shortestPath((start:Movie {title: 'The Replacements'})-[:REVIEWED*]-(end:Movie {title: 'The Birdcage'}))
RETURN p
```

### 18. **Find the shortest path between 'MikeN' and 'RobertI' using any relationship type.**

```cypher
MATCH p = shortestPath((start:Person {name: 'MikeN'})-[*]-(end:Person {name: 'RobertI'}))
RETURN p
```

### 19. **Find the shortest path from 'TomH' to 'CloudAtlas' with a maximum depth of 5 relationships.**

```cypher
MATCH p = shortestPath((start:Person {name: 'TomH'})-[*..5]-(end:Movie {title: 'CloudAtlas'}))
RETURN p
```

### 20. **Find the shortest path between 'TomH' and 'TomC' and calculate the weight of the path.**

```cypher
MATCH (start:Person {name: 'TomH'}), (end:Person {name: 'TomC'})
CALL algo.shortestPath.stream(start, end, null)
YIELD nodeId, cost
RETURN algo.getNodeById(nodeId).name AS node, cost
```

This structured set of queries should provide the desired results using Neo4j's algorithms for pathfinding, spanning trees, and random walks. Be sure to adapt any syntax if there are updates in Neo4j libraries or custom requirements in your database setup.