import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, 
                 device,
                 dim, 
                 max_position):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim
        self.max_position = max_position

        # Position Embedding을 계산하는 함수
        self.position_embeddings = self._get_position_embeddings().to(device)

    def _get_position_embeddings(self):
        """
        sinusoidal 임베딩을 사인과 코사인을 결합하여 생성
        """
        # 각 position에 대해 임베딩을 계산
        position = torch.arange(self.max_position, dtype=torch.float).unsqueeze(1)  # shape: [max_position, 1]
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * -(math.log(10000.0) / self.dim))  # shape: [dim/2]

        # 사인과 코사인 값 계산
        embeddings = torch.zeros(self.max_position, self.dim)  # [max_position, dim]
        embeddings[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스: 사인
        embeddings[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스: 코사인

        return embeddings

    def forward(self, position_ids):
        """
        주어진 위치에 대해 임베딩을 반환
        position_ids: (batch_size, seq_len) 위치 인덱스 텐서
        """
        return self.position_embeddings[position_ids]
