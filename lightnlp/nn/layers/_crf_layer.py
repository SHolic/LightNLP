import torch
import torch.nn as nn


class CrfLayer(nn.Module):
    def __init__(self, start="[SOS]", end="[END]", label2idx=None):
        super(CrfLayer, self).__init__()
        self.start = start
        self.end = end
        self.label2idx = label2idx
        self.label_num = len(label2idx.keys())

        print(self.label2idx)

        self.transitions = nn.Parameter(
            torch.randn(self.label_num, self.label_num))
        self.transitions.data[self.label2idx[self.start], :] = -10000
        self.transitions.data[:, self.label2idx[self.end]] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.label_num], -10000.)
        init_alphas[:, self.label2idx[self.start]] = 0.

        forward_var_list = [init_alphas]
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.label2idx[self.end]].repeat([feats.shape[0], 1])
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _score_sentence(self, feats, tags):
        score = torch.zeros(tags.shape[0])
        tags = torch.cat([torch.full([tags.shape[0], 1], self.label2idx[self.start]).long(), tags], dim=1)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(feat.shape[0]), tags[:, i + 1]]
        score = score + self.transitions[self.label2idx[self.end], tags[:, -1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.label2idx[self.start]] = 0

        forward_var_list = [init_vvars]
        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions

            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        terminal_var = forward_var_list[-1] + self.transitions[self.label2idx[self.stop]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.label2idx[self.start]
        best_path.reverse()
        return path_score, best_path

    def loss(self, emission, label_ids):
        forward_score = self._forward_alg(emission)
        gold_score = self._score_sentence(emission, label_ids)
        return torch.sum(forward_score - gold_score)

    def forward(self, emission, label_ids=None, only_loss=False):
        loss = None
        if label_ids is not None:
            loss = self.loss(emission=emission, label_ids=label_ids)
        if only_loss:
            return loss
        score, tag_seq = self._viterbi_decode_new(emission)
        return score, tag_seq, loss
