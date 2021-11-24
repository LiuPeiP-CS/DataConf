import torch
import torch.nn as nn


class CRF(nn.Module):

    def __init__(self, num_entities, pad_idx, bos_idx, eos_idx, device):
        super().__init__()
        self.num_entities = num_entities
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.cuda = device == 'cuda'

        self.transitions = nn.Parameter(torch.empty(self.num_entities, self.num_entities))  # treated as Parameter

        self._init_transitions()

    def forward(self, emissions, entities, mask=None):
        """Forward logic that computes the negative logarithm likelihood

        Args:
            emissions (torch.Tensor): emission matrix, which should be the output of previous layer (B, T, num_entities)
            entities (torch.LongTensor): given entities sequence (B, T)
            mask (torch.BoolTensor): indicates valid positions within each sequence in the batch (B, T)

        Returns:
            (torch.Tensor): neg-log-likelihood as loss, mean over batch (1,)

        """

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float, device='cuda' if self.cuda else 'cpu').bool()

        score = self._score(emissions, entities, mask)  # (B,)　# 最优序列的得分
        partition = self._log_partition(emissions, mask)  # (B,) # 所有可能序列的得分
        return -torch.mean(score - partition)  # (1,) # 对batch内所有的最优以及可能序列的loss进行融合

    def _init_transitions(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # Enforce contraints with a big negative number (from 'row' to 'column')
        # so exp(-10000) will tend to zero
        self.transitions.data[:, self.bos_idx] = -10000  # no transition to BOS
        self.transitions.data[self.eos_idx, :] = -10000  # no transition from eos (except to PAD)
        self.transitions.data[self.pad_idx, :] = -10000  # no transition from pad (except to PAD)
        self.transitions.data[:, self.pad_idx] = -10000  # no transition to pad (except from PAD)
        self.transitions.data[self.pad_idx, self.pad_idx] = 0
        self.transitions.data[self.pad_idx, self.eos_idx] = 0

    def _score(self, emissions, entities, mask):
        """

        Args:
            emissions (torch.Tensor): emission matrix, which should be the output of previous layer (B, T, num_entities)
            entities (torch.LongTensor): given entities sequence (B, T)
            mask (torch.BoolTensor): indicates valid positions within each sequence in the batch (B, T)

        Returns:
            torch.Tensor: scores of each entity sequence in current batch: (B,)

        """
        batch_size, seq_len = entities.shape # ground-truth tags in every sequence

        scores = torch.zeros(batch_size, dtype=torch.float, device='cuda' if self.cuda else 'cpu')

        first_ents = entities[:, 0]  # first entities (B,1) # 每个序列的第一个实体标记。
        last_valid_idx = mask.sum(1) - 1  # (B,1) # 每个序列的实际长度
        last_ents = entities.gather(1, last_valid_idx.unsqueeze(1)).squeeze() # last entities (B,1) # 根据序列长度获取每个序列的最后一个实体标记

        t_scores = self.transitions[self.bos_idx, first_ents] # size=(B,) # 开始符到第一个实体标记的trans_score.对于[x]和[1,Y]的索引,X会自动repeat复制成为[1,Y]大小

        e_scores = emissions[:, 0].gather(1, first_ents.unsqueeze(1)).squeeze() # (B,)# 根据序列的第一个实体标记，找到其对应的emission　score

        scores += e_scores + t_scores # 加法emission和transition

        # For each remaining entities(words)
        for i in range(1, seq_len):
            prev_ents = entities[:, i - 1]  # previous entities
            curr_ents = entities[:, i]  # current entities

            # Calculate emission and transition scores
            e_scores = emissions[:, i].gather(1, curr_ents.unsqueeze(1)).squeeze()  # (B,) # 寻找当前实体标签的emission值
            t_scores = self.transitions[prev_ents, curr_ents]  # (B,1) # 寻找相邻标签间的transition_score，这里的prev_ents和curr_ents以元组对的形式构造索引

            # Apply masking --- If position is PAD, then the contributing score should be 0
            e_scores = e_scores * mask[:, i] # 构造有效值：mask掉填充的外部信息
            t_scores = t_scores * mask[:, i] # 构造有效值：mask掉填充的外部信息

            scores += e_scores + t_scores # (B,1) 当前位置(第i位置的)的打分情况

        # Transition score from last entity to EOS
        scores += self.transitions[last_ents, self.eos_idx] # 添加到结束符的trans_score

        return scores  # (B,)

    # 该函数主要是为了计算所有可能的标签序列，并获取其对数似然
    def _log_partition(self, emissions, mask):
        """

        Args:
            emissions (torch.Tensor): emission matrix, which should be the output of previous layer (B, T, num_entities)
            mask (torch.BoolTensor): indicates valid positions within each sequence in the batch (B, T)

        Returns:
            (torch.Tensor): partition scores for current batch (B,)
        """

        batch_size, seq_len, num_ents = emissions.shape

        # Antilog of partition part of score, which will be finally applied a logsumexp()
        alphas = self.transitions[self.bos_idx, :].unsqueeze(0) + emissions[:, 0]  # (B, num_ents) # 任何可能开始的标签，其emission_score+trans_score

        for i in range(1, seq_len):
            alpha_t = []

            for ent in range(num_ents):
                # Emission scores for current entity, broadcast to entity-wise
                e_scores = emissions[:, i, ent].unsqueeze(1) # (B, 1) # seq第i个位置的第ent个标签的emission_score

                # Transition scores for current entity, broadcast to batch-wise
                t_scores = self.transitions[:, ent].unsqueeze(0) # (1, num_entities) # 前一时刻可到当前时刻第ent标签的所有标签trans_score

                # Combine current scores with previous alphas (in log space: add instead of multiply)
                scores = e_scores + t_scores + alphas # (B, num_entities) # 前一时刻可到时刻seq第i个位置的第ent个标签的score

                alpha_t.append(torch.logsumexp(scores, dim=1)) # inner (B,) #在tag_num维度，对各维度进行logsumexp.

            new_alphas = torch.stack(alpha_t, dim=0).t() # (B, num_entities) # stack后，alpha_t的维度为(num_entities,B) ;经过t()进行转置

            masking = mask[:, i].int().unsqueeze(1) # (B, 1)
            alphas = masking * new_alphas + (1 - masking) * alphas # 此处没有理解，也是为了计算序列得分

        # Transition scores from last entity to EOS
        end_scores = alphas + self.transitions[:, self.eos_idx].unsqueeze(0)  # (B, num_entities)# 所有标注以结束标记符终止，因此最后都指向结束标签的trans_score

        return torch.logsumexp(end_scores, dim=1) # 全部序列标注的最终得分

    def viterbi_decode(self, emissions, mask):
        """Find the most probable entity sequence by applying viterbi decoding

        Args:
            emissions (torch.Tensor): emission matrix, which should be the output of previous layer (B, T, num_entities)
            mask (torch.BoolTensor): indicates valid positions within each sequence in the batch (B, T)

        Returns:
            (tuple): tuple containing:
                (torch.Tensor): viterbi score for each sequence in current batch (B,).
                (list[list[int]]): best sequences of entities of this batch, representing in indexes (B, *)

        """
        batch_size, seq_len, num_ents = emissions.shape

        alphas = self.transitions[self.bos_idx, :].unsqueeze(0) + emissions[:, 0]  # (B, num_ents)　如上

        backpointers = []

        for i in range(1, seq_len):
            alpha_t = []
            backpointers_t = []

            for ent in range(num_ents):
                # Emission scores for current entity, broadcast to entity-wise
                e_scores = emissions[:, i, ent].unsqueeze(1)  # (B, 1) 如上

                # Transition scores for current entity, broadcast to batch-wise
                t_scores = self.transitions[:, ent].unsqueeze(0)  # (1, num_entities) 如上

                # Combine current scores with previous alphas (in log space: add instead of multiply)
                scores = e_scores + t_scores + alphas # (B, num_entities) 如上:获取到第i时刻上，第ent标签(由所有可能的标签间tag2tag trans_score相加)的score

                # Find the highest score and the entity associated with it
                max_scores, max_ents = torch.max(scores, dim=-1) # (B,1) # 在tag_num维度查看score矩阵的最大值，并记录相应的tag_num位置(即tag2tag转移到本标签时，引起产生最大socore的前一个标签的位置编号)

                # Add the max score for current entity
                alpha_t.append(max_scores)  # inner: (B,)# 当前i时刻，所有标签的最优得分进行聚集。循环结束后，alpha_t大小为(num_tag,Batch_size)

                # Add the corresponding entity to backpointers
                backpointers_t.append(max_ents)  # backpointers_t finally (num_entities, B) # 解释同alpha_t,不同的是alpha_t记录的是score,而backpointers_t记录的是哪个标签转移到该标签会使socre最大，记录其编号。

            new_alphas = torch.stack(alpha_t, dim=0).t()  # (B, num_entities) # 同上

            masking = mask[:, i].int().unsqueeze(1)  # (B, 1)
            alphas = masking * new_alphas + (1 - masking) * alphas # 同上　

            backpointers.append(backpointers_t)  # finally (T-1, num_entities, B) # 将所有时刻的backpointers_t进行汇合

        # Transition scores from last entity to EOS
        end_scores = alphas + self.transitions[:, self.eos_idx].unsqueeze(0)  # (B, num_entities) # 计算score直到序列结束。结束符不代表真实标注意义，因此最后一个标注还是序列的最后一个token，已经被包含在了backpointers中

        # Final most probable score and corresponding entity
        max_final_scores, max_final_ents = torch.max(end_scores, dim=-1)  # (B,1) # 序列计算结束后，找到每个序列的最大score及其标签位置，该score就是最优标注的score

        # Decode the best sequence for current batch
        # Follow the backpointers to find the best sequence of entities
        best_seqs = []
        emission_lens = mask.sum(dim=1)  # (B,) # 计算所有句子的长度

        for i in range(batch_size):
            sample_len = emission_lens[i].item() # 第i个句子的长度

            sample_final_entity = max_final_ents[i].item() # 第i个句子的最后标签

            sample_backpointers = backpointers[: sample_len - 1] # 因为第i个句子的长度问题，对所有实际标注路径都以此为节点截断

            best_path = [sample_final_entity] # 以第i句子的最后标签为初始化，建立列表

            best_entity = sample_final_entity  # 第i个句子最后标签的简单赋值

            for backpointers_t in reversed(sample_backpointers): # 从后向前倒推，每个backpointers_t的size为(num_tag,Batch_size)
                best_entity = backpointers_t[best_entity][i].item() # 前面我们说过，backpointers_t的当前位置，记录的是前一时刻哪个标签可以转移到本标签得到最大score，因此倒推到前一个标签的位置标记。
                best_path.insert(0, best_entity) # 插入到前面，即倒叙插入

            best_seqs.append(best_path)  # best_seqs finally (B, *) # 获取所有的最优标注序列

        return max_final_scores, best_seqs

