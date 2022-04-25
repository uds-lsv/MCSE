import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)   # non-linear activation
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.bert = BertModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.roberta = RobertaModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )



class ResNetVisnModel(nn.Module):
    def __init__(self, feature_dim,  proj_dim):
        super().__init__()
        self.mlp = MLPLayer(feature_dim, proj_dim)  # visual features -> grounding space

    def forward(self, x):
        x = self.mlp(x)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x


class MCSE(nn.Module):
    def __init__(self, lang_model, visn_model, args):
        super().__init__()
        self.args = args
        self.lang_model = lang_model
        self.visn_model = visn_model
        self.grounding = MLPLayer(args.hidden_size, args.proj_dim) # sent embeddings -> grounding space
        self.sim = Similarity(temp=self.args.temp)
        self.sim_vl = Similarity(temp=self.args.temp_vl)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, batch):
        lang_output = self.lang_model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      token_type_ids=batch['token_type_ids'] if 'position_ids' in batch.keys() else None,
                                      position_ids=batch['position_ids'] if 'position_ids' in batch.keys() else None)

        batch_size = batch['input_ids'].size(0)
        num_sent = batch['input_ids'].size(1)

        # [bs*2, hidden] -> [bs, 2, hidden]
        lang_pooled_output = lang_output.last_hidden_state[:, 0].view((batch_size, num_sent, -1))
        lang_projection = lang_output.pooler_output.view((batch_size, num_sent, -1))  # [bs, 2,  hidden],  output of additional MLP layer

        return lang_pooled_output, lang_projection

    def compute_loss(self, batch, cal_inter=False):
        l_pool, l_proj = self.forward(batch)

        # Separate representation
        z1, z2 = l_proj[:, 0], l_proj[:, 1]  # (bs, hidden)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)

        labels = torch.arange(cos_sim.size(0)).long().to(self.args.device)  # [0, 1, bs-1]  (bs)
        loss = self.loss_fct(cos_sim, labels)  # unsup: bs-1 negatives

        if not cal_inter:
            return loss

        else:
            v = self.visn_model(batch['img'])  # [bs, proj_dim]
            l2v_proj = self.grounding(l_pool)  # [bs, 2, proj_dim],  output for vision grounding
            l2v_proj = l2v_proj / l2v_proj.norm(2, dim=-1, keepdim=True)

            p1, p2 = l2v_proj[:, 0], l2v_proj[:, 1]  # (bs, proj)
            cos_sim_p0 = self.sim_vl(p1.unsqueeze(1), v.unsqueeze(0))  # (bs, bs)
            cos_sim_p1 = self.sim_vl(p2.unsqueeze(1), v.unsqueeze(0))
            inter_loss = (self.loss_fct(cos_sim_p0, labels) + self.loss_fct(cos_sim_p1, labels)) / 2

            return loss, inter_loss
