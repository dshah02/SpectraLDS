import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flashfftconv import FlashFFTConv

def future_fill(v, w):
    """
    Implements FutureFill operation using FFT for batched inputs.
    This version is more efficient for longer sequences.
    
    Args:
        v: Tensor of shape (B, t1) - the input sequence
        w: Tensor of shape (B, t2) - the filter
        
    Returns:
        Tensor of shape (B, t2-1) containing the FutureFill result
    """
    B, t1 = v.shape
    _, t2 = w.shape
    input_dtype = v.dtype
    
    # Pad to next power of 2 for efficient FFT
    n_fft = 2 ** int(torch.ceil(torch.log2(torch.tensor(t1 + t2 - 1))))
    
    v_padded = F.pad(v, (0, n_fft - t1)).float() #fft and ifft pytorch can't handle bf16
    w_padded = F.pad(w, (0, n_fft - t2)).float()
    
    v_fft = torch.fft.rfft(v_padded)
    w_fft = torch.fft.rfft(w_padded)
    result_fft = v_fft * w_fft
    result = torch.fft.irfft(result_fft, n=n_fft)
    
    return result[:, t1:t2].to(input_dtype)

class EpochedFutureFill(nn.Module):

    def __init__(self, filter, bsz = 1,  epoch_length=None, device = torch.device("cuda"), fftconv = None):
        
        super(EpochedFutureFill, self).__init__()
                
        if not isinstance(filter, torch.Tensor):
            self.filter = torch.tensor(filter, dtype=torch.float32, device = device)
        else:
            self.filter = filter.float().to(device)
            
        self.L = self.filter.shape[-1]
        
        if epoch_length is None:
            self.K = int(math.sqrt(2*self.L * math.log2(self.L)))
            # print(f"None: Epoch K for FutureFill is {self.K}")
        elif isinstance(epoch_length, str) and  epoch_length == "None":
            self.K = int(math.sqrt(2*self.L * math.log2(self.L)))
            # print(f"str: Epoch K for FutureFill is {self.K}")
        else:
            self.K = epoch_length
            # print(f"Given: Epoch K for FutureFill is {self.K}")
        
        self.bsz = bsz
        
        self.tau = 1  # Current position within epoch
        self.cache = torch.zeros(bsz, self.K, dtype=torch.float32, device = device)
        self.register_buffer('buffer', torch.zeros(self.bsz , 0, dtype=torch.float32,device =device))
        self.device = device
        self.prefill_idx = -1 #prefill is not used

        self.fftconv = fftconv #If None, use regular Pytorch FFT; else FlashFFT

    def reset_cache(self):
        self.cache = torch.zeros(self.bsz, self.K, dtype=torch.float32, device =self.device)
        self.buffer = torch.zeros(self.bsz , 0, dtype=torch.float32,device =self.device)
        self.tau = 1

    def prefill(self, x, length):
        print("WARNING using prefill")

        self.prefill_cache = future_fill(x, self.filter[: x.shape[1] + length])
        self.prefill_idx = 1

        #next power of 2 for fft
        n_fft = 2 ** int(torch.ceil(torch.log2(torch.tensor(x.shape[1]))))
        bsz = x.shape[0]

        x_reshaped = x.unsqueeze(1)  # Shape: [bsz, 1, seq_len]
        padded_input = F.pad(x_reshaped, (self.filter.shape[1]-1, 0))  # Pad along the sequence dimension

        # Flip the filters for all batches at once
        flipped_filter = self.filter.flip(1).unsqueeze(1)  # Shape: [bsz, 1, filter_len]
        padded_input = padded_input.transpose(0, 1)  # Shape: [1, bsz, seq_len+pad]
        standard_output = F.conv1d( #maybe replace for speed?
            padded_input.to(flipped_filter.dtype),
            flipped_filter,
            groups=bsz  # Each filter only applies to its corresponding batch element
        )[0]  # Shape: [bsz, seq_len]

        return standard_output

    def flash_future_fill(self, v,w):
        B, t1 = v.shape
        _, t2 = w.shape
        input_dtype = v.dtype

        n_fft = 2 ** int(torch.ceil(torch.log2(torch.tensor(t1 + t2 - 1))))
    
        v_padded = F.pad(v, (0, n_fft - t1)).unsqueeze(dim = 0).to(torch.bfloat16)

        out = self.fftconv(v_padded, w)
        return out.squeeze(dim = 0)[:, t1:t2].to(input_dtype)

    def process(self, u_t, filter, *args, **kwargs):
        """Process a single input token and return the convolution output.
        
        Args:
            u_t: The input token at the current time step as a tensor or scalar
            
        Returns:
            The convolution result y_t at the current time step
        """
        init_dtype = u_t.dtype
            
        self.buffer = torch.cat([self.buffer, u_t.unsqueeze(dim = -1)], dim  = -1)
        t = self.buffer.size(-1)
        
        buffer_slice = self.buffer[:, t-min(self.tau, t):t].flip(dims=[1])
        filter_slice = filter[:, :min(self.tau, t)]
        y_hat = torch.sum(buffer_slice * filter_slice, dim=1) +  self.cache[:,self.tau-1]
        
        if 0 < self.prefill_idx <= self.prefill_cache.shape[-1]:
            y_hat += self.prefill_cache[:, self.prefill_idx - 1]
            self.prefill_idx += 1
 
        if self.tau == self.K:
            if self.fftconv:
                # print("Flash futurefill")
                self.cache = self.flash_future_fill(self.buffer, self.filter[:, : self.buffer.shape[-1] + self.K])
            else: 
                # print("Regular futurefill")
                self.cache = future_fill(self.buffer, self.filter[:, : self.buffer.shape[-1] + self.K])

            self.tau = 1
        else:
            
            self.tau += 1      
        
        return y_hat

    def forward(self, x, *args, **kwargs):
        bsz, seq_len = x.shape
        epoched_output = []
        for t in range(seq_len):
            y_t = self.process(x[:, t], *args, **kwargs)
            epoched_output.append(y_t)

        epoched_output = torch.stack(epoched_output, dim = -1)
        return epoched_output #[B, t2 - t1]