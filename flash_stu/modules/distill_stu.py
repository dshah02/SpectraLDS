import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from flash_stu.utils.lds import LDS


class DistillSTU(nn.Module):
    def __init__(self, stu, lds_path = None, state_dim = 100, dtype = torch.float32) -> None:
        super(DistillSTU, self).__init__()
        
        self.K = stu.config.num_eigh
        self.d_in = stu.config.n_embd if hasattr(stu.config, "n_embd") else stu.config.dim
        self.d_out = stu.config.n_embd if hasattr(stu.config, "n_embd") else stu.config.dim
        self.use_hankel_L = stu.config.use_hankel_L
        self.use_approx = stu.config.use_approx
        self.output_dtype = stu.config.torch_dtype
        
        self.lds = self.get_lds(lds_path, state_dim, dtype, self.d_in)
        
        if self.use_approx:
            self.M_inputs = nn.Parameter(
                stu.M_inputs.data.to(self.lds.dtype)
            )
            
            self.M_filters = stu.M_filters.data.to(self.lds.dtype)
        else:
            self.M_phi_plus = nn.Parameter(
                stu.M_phi_plus.data.to(self.lds.dtype)
            )
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    stu.M_phi_minus.data.to(self.lds.dtype)
                )
    
    # @torch.compile
    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        if self.use_approx:
            x = x.to(self.lds.dtype) @ self.M_inputs  # [B, L, d_in] => [B, L, K_total]
            
            bsz, seq_len, d_in = x.shape
            
            # Flatten (B * d_in, seq_len, 1)
            x_reshaped = x.transpose(1, 2)  # [B, d_in, L]
            x_reshaped = x_reshaped.reshape(bsz * d_in, seq_len, 1)

            U_reshaped = self.lds(x_reshaped)  # [B*d_in, seq_len, state_dim]
            
            # Reshape back to [B, seq_len, state_dim, d_in]
            U = U_reshaped.view(bsz, d_in, seq_len, -1)  # [B, d_in, L, state_dim]
            U = U.permute(0, 2, 3, 1).contiguous()       # [B, L, state_dim, d_in]
            
            spectral_plus = (U[:, :, self.K:, :] * self.M_filters[None, None, :, :]).sum(dim=2)
            spectral_minus = (U[:, :, :self.K, :] * self.M_filters[None, None, :, :]).sum(dim=2)
            
            ret = spectral_plus if self.use_hankel_L else (spectral_plus + spectral_minus)
            return ret.to(self.output_dtype)
            
        else:
            x = x.double()
            bsz = x.shape[0]
            x_reshaped = x.permute(0, 2, 1).reshape(-1, x.shape[1], 1)
            U_reshaped = self.lds(x_reshaped)
            U = U_reshaped.reshape(bsz, x.shape[2], x.shape[1], -1).permute(0, 2, 3, 1)
            U_plus, U_minus = U[:, :, :self.K, :], U[:, :, self.K:, :]
            
            spectral_plus = torch.tensordot(U_plus, self.M_phi_plus, dims=([2, 3], [0, 1]))
            if not self.use_hankel_L:
                spectral_minus = torch.tensordot(U_minus, self.M_phi_minus, dims=([2, 3], [0, 1]))
            ret = spectral_plus if self.use_hankel_L else (spectral_plus + spectral_minus)
            return ret.to(self.output_dtype)

    def loss(self, inputs, targets):
        pred = self.forward(inputs, input_pos=None)
        return F.mse_loss(pred, targets)

    def get_lds(self, checkpoint_path=None, state_dim = 100, dtype = torch.float64, bsz_dim = 896):
        # print(f"INITIALIZED WITH SD {state_dim}")
        if checkpoint_path is None:
            lds_phi = LDS( 
                state_dim=state_dim,
                input_dim=1,
                output_dim=2 * self.K,
                dtype=dtype,
            )
            
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            lds_phi = LDS( 
                state_dim=state_dim,
                input_dim=1,
                output_dim=2 * self.K,
                dtype=dtype,
                bsz_dim = bsz_dim, 
            )
            state_dict = checkpoint['model_state_dict']
            state_dict = {k: v for k, v in state_dict.items() if k not in ['h0', 'M']}
            lds_phi.load_state_dict(state_dict, strict=False)
                
        return lds_phi.cuda()
        