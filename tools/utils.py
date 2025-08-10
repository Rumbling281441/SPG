import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import math



    
def register_attn_control(unet, controller, cache=None):
    def attn_forward(self):
        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
        ):
            residual = hidden_states
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            q = self.to_q(hidden_states)
            is_self = encoder_hidden_states is None

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            inner_dim = k.shape[-1]
            head_dim = inner_dim // self.heads

            q = q.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
           
            if cache.do_spg and cache.injection and not cache.do_cfg:
                q,q_con=q.chunk(2)
                k,k_con=k.chunk(2)
                v,v_con=v.chunk(2)
            if cache.do_spg and cache.injection and cache.do_cfg and not cache.do_spg2:
                q,q_con,q_uncon=q.chunk(3)
                k,k_con,k_uncon=k.chunk(3)
                v,v_con,v_uncon=v.chunk(3)
            if cache.do_spg and cache.injection and cache.do_cfg and cache.do_spg2:
                q,q_s,q_con,q_uncon=q.chunk(4)
                k,k_s,k_con,k_uncon=k.chunk(4)
                v,v_s,v_con,v_uncon=v.chunk(4)
            if is_self and cache.injection and controller.cur_self_layer in controller.self_layers:
                # q=cache.get_q()
                
                k=cache.get_k()
                v=cache.get_v()
            if cache.do_spg2 and controller.cur_self_layer in cache.self_layers and is_self and cache.injection:
                q_s=cache.get_q_s()
                    # k_s=k
                    # v_s=v
                
            hidden_states = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            if cache.do_spg2:
                hidden_states_s = F.scaled_dot_product_attention(
                q_s, k_s, v_s, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            if cache.do_spg and not cache.do_cfg:
                hidden_states_con=F.scaled_dot_product_attention(
                    q_con, k_con, v_con, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
            if cache.do_spg and cache.do_cfg:
                hidden_states_con=F.scaled_dot_product_attention(
                    q_con, k_con, v_con, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states_uncon=F.scaled_dot_product_attention(
                    q_uncon, k_uncon, v_uncon, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
            if is_self and controller.cur_self_layer in controller.self_layers:
                if cache.use_cache:  
                    cache.add(q, k, v, hidden_states)
            if is_self and controller.cur_self_layer in cache.self_layers:
                if cache.use_cache2:
                    cache.add_s(q,k,v,hidden_states)
            if cache.do_spg and not cache.do_cfg:
                hidden_states=torch.cat([hidden_states,hidden_states_con],dim=0)
            if cache.do_spg and cache.do_cfg and not cache.do_spg2:
                hidden_states=torch.cat([hidden_states,hidden_states_con,hidden_states_uncon],dim=0)
            if cache.do_spg and cache.do_cfg and cache.do_spg2:
                hidden_states=torch.cat([hidden_states,hidden_states_s,hidden_states_con,hidden_states_uncon],dim=0)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            )
            hidden_states = hidden_states.to(q.dtype)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            if is_self:
                controller.cur_self_layer += 1

            return hidden_states

        return forward

    def modify_forward(net, count):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":  # spatial Transformer layer
                net.forward = attn_forward(net)
                return count + 1
            elif hasattr(net, "children"):
                count = modify_forward(subnet, count)
        return count

    cross_att_count = 0
    for net_name, net in unet.named_children():
        cross_att_count += modify_forward(net, 0)
    controller.num_self_layers = cross_att_count // 2


def load_image(image_path, size=None, mode="RGB"):
    img = Image.open(image_path).convert(mode)
    if size is None:
        width, height = img.size
        new_width = (width // 64) * 64
        new_height = (height // 64) * 64
        size = (new_width, new_height)
    img = img.resize(size, Image.BICUBIC)
    return ToTensor()(img).unsqueeze(0)


def adain(source, target, eps=1e-6):
    source_mean, source_std = torch.mean(source, dim=(2, 3), keepdim=True), torch.std(
        source, dim=(2, 3), keepdim=True
    )
    target_mean, target_std = torch.mean(
        target, dim=(0, 2, 3), keepdim=True
    ), torch.std(target, dim=(0, 2, 3), keepdim=True)
    normalized_source = (source - source_mean) / (source_std + eps)
    transferred_source = normalized_source * target_std + target_mean

    return transferred_source


class Controller:
    def step(self):
        self.cur_self_layer = 0

    def __init__(self, self_layers=(0, 16)):
        self.num_self_layers = -1
        self.cur_self_layer = 0
        self.self_layers = list(range(*self_layers))


class DataCache:
    def __init__(self):
        self.q = []
        self.k = []
        self.v = []
        self.out = []
        self.q_s = []
        self.k_s = []
        self.v_s = []
        self.out_s = []
        self.injection=False
        self.use_cache=True
        self.do_spg=False
        self.do_cfg=False
        self.do_spg2=False
        self.use_cache2=False
        self.s_start_layer=0
        self.s_end_layer=0
        self.self_layers=[]
    def clear(self):
        self.q.clear()
        self.k.clear()
        self.v.clear()
        self.out.clear()
        self.q_s.clear()
        self.k_s.clear()
        self.v_s.clear()
        self.out_s.clear()

    def add(self, q, k, v, out):
        self.q.append(q)
        self.k.append(k)
        self.v.append(v)
        self.out.append(out)
    def add_s(self,q,k,v,out):
        self.q_s.append(q)
        self.k_s.append(k)
        self.v_s.append(v)
        self.out_s.append(out)
    def get(self):
        
        return self.q.copy(), self.k.copy(), self.v.copy(), self.out.copy()
    def get_q(self):
        return self.q.pop(0)
    def get_k(self):
        return self.k.pop(0)
    def get_v(self):
        return self.v.pop(0)
    def get_o(self):
        return self.out.pop(0)
    def get_q_s(self):
        return self.q_s.pop(0)
    def get_k_s(self):
        return self.k_s.pop(0)
    def get_v_s(self):
        return self.v_s.pop(0)
    def get_o_s(self):
        return self.out_s.pop(0)
    
def show_image(path, title, display_height=3, title_fontsize=12):
    img = Image.open(path)
    img_width, img_height = img.size

    aspect_ratio = img_width / img_height
    display_width = display_height * aspect_ratio

    plt.figure(figsize=(display_width, display_height))
    plt.imshow(img)
    plt.title(title, 
             fontsize=title_fontsize, 
             fontweight='bold', 
             pad=20) 
    plt.axis('off')    
    plt.tight_layout() 
    plt.show()
