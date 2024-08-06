import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


def hz2erb(hz):
    """
    Converts frequency value in Hz to human-defined ERB band index,
    using the formula of Glasberg and Moore.
    """
    return 9.265 * np.log(1 + hz / (24.7 * 9.265))

def erb2hz(erb):
    """
    Converts human-defined ERB band index to frequency value in Hz,
    using the formula of Glasberg and Moore.
    """
    return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)


class ERB:

    def __init__(self, samplerate: int, fftsize: int, erbsize: int, minwidth: int, alpha: float):

        self.samplerate = samplerate
        self.fftsize = fftsize
        self.erbsize = erbsize
        self.minwidth = minwidth
        self.alpha = alpha

        self.widths = ERB.get_band_widths(samplerate, fftsize, erbsize, minwidth)
        self.weights = ERB.get_band_weights(samplerate, self.widths)
        self.mean = np.zeros(erbsize)
        self.first_step = True

    def __call__(self, dfts):

        x = np.abs(dfts) # TODO try np.absolute with 10*log10 instead
        y = np.matmul(x, self.weights)
        y = 20 * np.log10(y + np.finfo(dfts.dtype).eps)

        # TODO ISSUE #100
        # mean = np.full(y.shape[-1], y[..., 0, :])
        if(self.first_step):
            self.mean = y[..., 0, :].copy()
            self.first_step = False
        # mean = np.mean(y)
        for i in range(y.shape[-2]):
            self.mean = y[..., i, :] * (1 - self.alpha) + self.mean * self.alpha
            y[..., i, :] -= self.mean
        y /= 40

        return y

    @staticmethod
    def get_band_widths(samplerate: int, fftsize: int, erbsize: int, minwidth: int):

        dftsize = fftsize / 2 + 1
        nyquist = samplerate / 2
        bandwidth = samplerate / fftsize

        erbmin = hz2erb(0)
        erbmax = hz2erb(nyquist)
        erbinc = (erbmax - erbmin) / erbsize

        bands = np.arange(1, erbsize + 1)
        freqs = erb2hz(erbmin + erbinc * bands)
        widths = np.round(freqs / bandwidth).astype(int)

        prev = 0
        over = 0

        for i in range(erbsize):

            next = widths[i]
            width = next - prev - over
            prev = next

            over = max(minwidth - width, 0)
            width = max(minwidth, width)

            widths[i] = width

        widths[erbsize - 1] += 1
        assert np.sum(widths) == dftsize

        return widths

    @staticmethod
    def get_band_weights(samplerate: int, widths: np.ndarray, normalized: bool = True, inverse: bool = False):

        n_freqs = int(np.sum(widths))
        all_freqs = np.linspace(0, samplerate // 2, n_freqs + 1)[:-1]

        b_pts = np.cumsum([0] + widths.tolist()).astype(int)[:-1]

        fb = np.zeros((all_freqs.shape[0], b_pts.shape[0]))

        for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
            fb[b : b + w, i] = 1

        if inverse:
            fb = fb.t()
            if not normalized:
                fb /= np.sum(fb, axis=1, keepdim=True)
        else:
            if normalized:
                fb /= np.sum(fb, axis=0)

        return fb



if __name__=="__main__":
    from libdf import DF, erb, erb_norm, unit_norm
    import torch
    import numpy as np

    # origin df
    df_state = DF(
        sr=48000,
        fft_size=960,
        hop_size=480,
        nb_bands=32,
        min_nb_erb_freqs=1,
    )
    df_erb_fb = df_state.erb_widths()
    

    # my erb
    my_erb = ERB(
        samplerate=48000,
        fftsize=960,
        erbsize=32,
        minwidth=1,
        alpha=0.99)


    spec = torch.randn((1, 1, 1, 481), dtype=torch.complex64)

    # spec, erb_feat, spec_feat = df_features(torch.from_numpy(x.astype(np.float32)), state, param_deep_filter_bins, device=cpu)

    df_erb_res = erb(spec.numpy(), df_erb_fb).squeeze(1)
    df_erb_feat = torch.as_tensor(erb_norm(df_erb_res, 0.99)).unsqueeze(1)
    

    my_erb = my_erb(spec.numpy())
    
    xxx = 1

# from typing import List, Tuple

# import numpy as np


# def hz2erb(hz):
#     """
#     Converts frequency value in Hz to human-defined ERB band index,
#     using the formula of Glasberg and Moore.
#     """
#     return 9.265 * np.log(1 + hz / (24.7 * 9.265))

# def erb2hz(erb):
#     """
#     Converts human-defined ERB band index to frequency value in Hz,
#     using the formula of Glasberg and Moore.
#     """
#     return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)


# class ERB:

#     def __init__(self, samplerate: int, fftsize: int, erbsize: int, minwidth: int, alpha: float):

#         self.samplerate = samplerate
#         self.fftsize = fftsize
#         self.erbsize = erbsize
#         self.minwidth = minwidth
#         self.alpha = alpha

#         self.widths = ERB.get_band_widths(samplerate, fftsize, erbsize, minwidth)
#         self.weights = ERB.get_band_weights(samplerate, self.widths)

#         xxx = 1

#     def __call__(self, dfts):

#         x = np.abs(dfts) # TODO try np.absolute with 10*log10 instead
#         y = np.matmul(x, self.weights)
#         y = 10 * np.log10(y + 1e-10)

#         # TODO ISSUE #100
#         mean = np.full(y.shape[-1], y[..., 0, :])
#         alpha = self.alpha
#         for i in range(y.shape[-2]):
#             mean = y[..., i, :] * (1 - alpha) + mean * alpha
#             y[..., i, :] -= mean
#         y /= 40

#         return y

#     @staticmethod
#     def get_band_widths(samplerate: int, fftsize: int, erbsize: int, minwidth: int):

#         dftsize = fftsize / 2 + 1
#         nyquist = samplerate / 2
#         bandwidth = samplerate / fftsize

#         # erbmin = hz2erb(0)
#         erbmin = hz2erb(0)
#         erbmax = hz2erb(nyquist)
#         erbinc = (erbmax - erbmin) / erbsize

#         bands = np.arange(1, erbsize + 1)
#         freqs = erb2hz(erbmin + erbinc * bands)
#         widths = np.round(freqs / bandwidth).astype(int)

#         prev = 0
#         over = 0

#         for i in range(erbsize):

#             next = widths[i]
#             width = next - prev - over
#             prev = next

#             over = max(minwidth - width, 0)
#             width = max(minwidth, width)

#             widths[i] = width

#         widths[erbsize - 1] += 1
#         assert (np.sum(widths) - dftsize) < 1e-5

#         return widths

#     @staticmethod
#     def get_band_weights(samplerate: int, widths: np.ndarray, normalized: bool = True, inverse: bool = False):

#         n_freqs = int(np.sum(widths))
#         all_freqs = np.linspace(0, samplerate // 2, n_freqs + 1)[:-1]

#         b_pts = np.cumsum([0] + widths.tolist()).astype(int)[:-1]

#         fb = np.zeros((all_freqs.shape[0], b_pts.shape[0]))

#         for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
#             fb[b : b + w, i] = 1

#         if inverse:
#             fb = fb.t()
#             if not normalized:
#                 fb /= np.sum(fb, axis=1, keepdim=True)
#         else:
#             if normalized:
#                 fb /= np.sum(fb, axis=0)

#         return fb
    

# if __name__=="__main__":
#     my_erb_calc = ERB(
#         samplerate=48000,
#         fftsize=960,
#         erbsize=32,
#         minwidth=1,
#         alpha=0.99)
#     widths = my_erb_calc.get_band_widths(48000,960,32,1)
#     weights = my_erb_calc.get_band_weights(48000,widths)


#     xxx = 1