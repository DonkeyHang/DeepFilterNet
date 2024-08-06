import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import glob
import os
import time
import warnings
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

from audio import read, write
from cpx import CPX
from erb import ERB
from stft import STFT
import numpy as np


import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio as ta

from df.deepfilternet3 import init_model
from df.checkpoint import load_model as load_model_cp
from df.config import config
from df.io import load_audio, resample, save_audio
from df.logger import init_logger
from df.model import ModelParams
from df.modules import get_device
from df.utils import as_complex, as_real, download_file, get_cache_dir, get_norm_alpha
from df.version import version
from libdf import DF, erb, erb_norm, unit_norm

PRETRAINED_MODELS = ("DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3")
DEFAULT_MODEL = "DeepFilterNet3"
FFT_SIZE = 960


# class AudioDataset(Dataset):
#     def __init__(self, files: List[str], sr: int) -> None:
#         super().__init__()
#         self.files = []
#         for file in files:
#             if not os.path.isfile(file):
#                 logger.warning(f"File not found: {file}. Skipping...")
#             self.files.append(file)
#         self.sr = sr

#     def __getitem__(self, index) -> Tuple[str, Tensor, int]:
#         fn = self.files[index]
#         audio, meta = load_audio(fn, self.sr, "cpu")
#         return fn, audio, meta.sample_rate

#     def __len__(self):
#         return len(self.files)


# def main(args):
#     model, df_state, suffix, epoch = init_df(
#         args.model_base_dir,
#         post_filter=args.pf,
#         log_level=args.log_level,
#         config_allow_defaults=True,
#         epoch=args.epoch,
#         mask_only=args.no_df_stage,
#     )
#     suffix = suffix if args.suffix else None
#     if args.output_dir is None:
#         args.output_dir = "."
#     elif not os.path.isdir(args.output_dir):
#         os.mkdir(args.output_dir)
#     df_sr = ModelParams().sr
#     if args.noisy_dir is not None:
#         if len(args.noisy_audio_files) > 0:
#             logger.error("Only one of `noisy_audio_files` or `noisy_dir` arguments are supported.")
#             exit(1)
#         input_files = glob.glob(args.noisy_dir + "/*")
#     else:
#         assert len(args.noisy_audio_files) > 0, "No audio files provided"
#         input_files = args.noisy_audio_files
#     ds = AudioDataset(input_files, df_sr)
#     loader = DataLoader(ds, num_workers=2, pin_memory=True)
#     n_samples = len(ds)
#     for i, (file, audio, audio_sr) in enumerate(loader):
#         file = file[0]
#         audio = audio.squeeze(0)
#         progress = (i + 1) / n_samples * 100
#         t0 = time.time()
#         audio = enhance(
#             model, df_state, audio, pad=args.compensate_delay, atten_lim_db=args.atten_lim
#         )
#         t1 = time.time()
#         t_audio = audio.shape[-1] / df_sr
#         t = t1 - t0
#         rtf = t / t_audio
#         fn = os.path.basename(file)
#         p_str = f"{progress:2.0f}% | " if n_samples > 1 else ""
#         logger.info(f"{p_str}Enhanced noisy audio file '{fn}' in {t:.2f}s (RT factor: {rtf:.3f})")
#         audio = resample(audio.to("cpu"), df_sr, audio_sr)
#         save_audio(file, audio, sr=audio_sr, output_dir=args.output_dir, suffix=suffix, log=False)


# def get_model_basedir(m: Optional[str]) -> str:
#     if m is None:
#         m = DEFAULT_MODEL
#     is_default_model = m in PRETRAINED_MODELS
#     if is_default_model:
#         return maybe_download_model(m)
#     return m


def init_df(
    model_base_dir: Optional[str] = None,
    post_filter: bool = False,
    log_level: str = "INFO",
    log_file: Optional[str] = "enhance.log",
    config_allow_defaults: bool = True,
    epoch: Union[str, int, None] = "best",
    default_model: str = DEFAULT_MODEL,
    mask_only: bool = False,
) -> Tuple[nn.Module, DF, str, int]:
    """Initializes and loads config, model and deep filtering state.

    Args:
        model_base_dir (str): Path to the model directory containing checkpoint and config. If None,
            load the pretrained DeepFilterNet2 model.
        post_filter (bool): Enable post filter for some minor, extra noise reduction.
        log_level (str): Control amount of logging. Defaults to `INFO`.
        log_file (str): Optional log file name. None disables it. Defaults to `enhance.log`.
        config_allow_defaults (bool): Whether to allow initializing new config values with defaults.
        epoch (str): Checkpoint epoch to load. Options are `best`, `latest`, `<int>`, and `none`.
            `none` disables checkpoint loading. Defaults to `best`.

    Returns:
        model (nn.Modules): Intialized model, moved to GPU if available.
        df_state (DF): Deep filtering state for stft/istft/erb
        suffix (str): Suffix based on the model name. This can be used for saving the enhanced
            audio.
        epoch (int): Epoch number of the loaded checkpoint.
    """
    # try:
    #     from icecream import ic, install

    #     ic.configureOutput(includeContext=True)
    #     install()
    # except ImportError:
    #     pass
    model_base_dir = '/Users/donkeyddddd/Library/Caches/DeepFilterNet/DeepFilterNet3'
    # use_default_model = model_base_dir is None or model_base_dir in PRETRAINED_MODELS
    # model_base_dir = get_model_basedir(model_base_dir or default_model)

    # if not os.path.isdir(model_base_dir):
        # raise NotADirectoryError("Base directory not found at {}".format(model_base_dir))
    # log_file = os.path.join(model_base_dir, log_file) if log_file is not None else None
    # init_logger(file=log_file, level=log_level, model=model_base_dir)
    # if use_default_model:
    #     logger.info(f"Using {default_model} model at {model_base_dir}")
    
    config.load(
        os.path.join(model_base_dir, "config.ini"),
        config_must_exist=True,
        allow_defaults=config_allow_defaults,
        allow_reload=True,
    )
    
    # if post_filter:
        # config.set("mask_pf", True, bool, ModelParams().section)
        # try:
        #     beta = config.get("pf_beta", float, ModelParams().section)
        #     beta = f"(beta: {beta})"
        # except KeyError:
        #     beta = ""
        # logger.info(f"Running with post-filter {beta}")
    # p = ModelParams()
    # df_state = DF(
    #     sr=p.sr,
    #     fft_size=p.fft_size,
    #     hop_size=p.hop_size,
    #     nb_bands=p.nb_erb,
    #     min_nb_erb_freqs=p.min_nb_freqs,
    # )
    df_state = DF(
        sr=48000,
        fft_size=960,
        hop_size=480,
        nb_bands=32,
        min_nb_erb_freqs=1,
    )
    # checkpoint_dir = os.path.join(model_base_dir, "checkpoints")
    # load_cp = epoch is not None and not (isinstance(epoch, str) and epoch.lower() == "none")
    # if not load_cp:
    #     checkpoint_dir = None
    # mask_only = mask_only or config(
    #     "mask_only", cast=bool, section="train", default=False, save=False
    # )
    # model, epoch = load_model_cp(checkpoint_dir, df_state, epoch=epoch, mask_only=mask_only)
    # model, epoch = load_model_cp(checkpoint_dir, df_state, epoch=epoch, mask_only=False)
    model = init_model(df_state)
    latest = torch.load("/Users/donkeyddddd/Library/Caches/DeepFilterNet/DeepFilterNet3/checkpoints/model_120.ckpt.best", map_location="cpu")
    latest = {k.replace("clc", "df"): v for k, v in latest.items()}
    model.load_state_dict(latest, strict=False)
    model.to(memory_format=torch.channels_last)
    
    # if (epoch is None or epoch == 0) and load_cp:
    #     logger.error("Could not find a checkpoint")
    #     exit(1)
    # logger.debug(f"Loaded checkpoint from epoch {epoch}")
    model = model.to(get_device())
    # Set suffix to model name
    # suffix = os.path.basename(os.path.abspath(model_base_dir))
    # if post_filter:
    #     suffix += "_pf"
    # logger.info("Running on device {}".format(get_device()))
    # logger.info("Model loaded")
    # return model, df_state, suffix, epoch
    return model, df_state


def erb_fb_function(widths: np.ndarray, sr: int, normalized: bool = True, inverse: bool = False) -> Tensor:
    n_freqs = int(np.sum(widths))
    all_freqs = torch.linspace(0, sr // 2, n_freqs + 1)[:-1]

    b_pts = np.cumsum([0] + widths.tolist()).astype(int)[:-1]

    fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
    for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
        fb[b : b + w, i] = 1
    # Normalize to constant energy per resulting band
    if inverse:
        fb = fb.t()
        if not normalized:
            fb /= fb.sum(dim=1, keepdim=True)
    else:
        if normalized:
            fb /= fb.sum(dim=0)
    return fb.to(device=get_device())



def df_features(audio: Tensor, df: DF, nb_df: int, device=None) -> Tuple[Tensor, Tensor, Tensor]:
    spec = df.analysis(audio.numpy())  # [C, Tf] -> [C, Tf, F]
    a = get_norm_alpha(False)
    erb_fb = df.erb_widths()

    # =====
    # my_erb_calc = ERB(
    #     samplerate=48000,
    #     fftsize=960,
    #     erbsize=32,
    #     minwidth=1,
    #     alpha=0.99)
    # widths = my_erb_calc.get_band_widths(48000,960,32,1)
    # weights = my_erb_calc.get_band_weights(48000,widths)

    # =====


    # ======
    # x = np.abs(spec) # TODO try np.absolute with 10*log10 instead
    # y = np.matmul(x, weights)
    # y = 10 * np.log10(y + np.finfo(spec.dtype).eps)

    # TODO ISSUE #100
    # mean = np.full(y.shape[-1], y[..., 0, :])
    # alpha = 0.99
    # for i in range(y.shape[-2]):
    #     mean = y[..., i, :] * (1 - alpha) + mean * alpha
    #     y[..., i, :] -= mean
    # y /= 40
    # ======


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        erb_feat = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1)

    spec_feat = as_real(torch.as_tensor(unit_norm(spec[..., :nb_df], a)).unsqueeze(1))
    spec = as_real(torch.as_tensor(spec).unsqueeze(1))
    if device is not None:
        spec = spec.to(device)
        erb_feat = erb_feat.to(device)
        spec_feat = spec_feat.to(device)
    # print("spec :",spec.shape)
    # print("erb_feat :",erb_feat.shape)
    # print("spec_feat :",spec_feat.shape)
    return spec, erb_feat, spec_feat


@torch.no_grad()
def enhance(
    model: nn.Module, df_state: DF, audio: Tensor, pad=True, atten_lim_db: Optional[float] = None
):
    """Enhance a single audio given a preloaded model and DF state.

    Args:
        model (nn.Module): A DeepFilterNet model.
        df_state (DF): DF state for STFT/ISTFT and feature calculation.
        audio (Tensor): Time domain audio of shape [C, T]. Sampling rate needs to match to `model` and `df_state`.
        pad (bool): Pad the audio to compensate for delay due to STFT/ISTFT.
        atten_lim_db (float): An optional noise attenuation limit in dB. E.g. an attenuation limit of
            12 dB only suppresses 12 dB and keeps the remaining noise in the resulting audio.

    Returns:
        enhanced audio (Tensor): If `pad` was `False` of shape [C, T'] where T'<T slightly delayed due to STFT.
            If `pad` was `True` it has the same shape as the input.
    """
    model.eval()
    # bs = audio.shape[0]
    # if hasattr(model, "reset_h0"):
    #     model.reset_h0(batch_size=bs, device=get_device())
    orig_len = audio.shape[-1]
    n_fft, hop = 0, 0
    if pad:
        n_fft, hop = df_state.fft_size(), df_state.hop_size()
        # Pad audio to compensate for the delay due to the real-time STFT implementation
        audio = F.pad(audio, (0, n_fft))
    # nb_df = getattr(model, "nb_df", getattr(model, "df_bins", ModelParams().nb_df))
    nb_df = 96
    spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=get_device())
    enhanced = model(spec.clone(), erb_feat, spec_feat)[0].cpu()
    enhanced = as_complex(enhanced.squeeze(1))
    # if atten_lim_db is not None and abs(atten_lim_db) > 0:
    #     lim = 10 ** (-abs(atten_lim_db) / 20)
    #     enhanced = as_complex(spec.squeeze(1).cpu()) * lim + enhanced * (1 - lim)
    audio = torch.as_tensor(df_state.synthesis(enhanced.numpy()))
    if pad:
        # The frame size is equal to p.hop_size. Given a new frame, the STFT loop requires e.g.
        # ceil((n_fft-hop)/hop). I.e. for 50% overlap, then hop=n_fft//2
        # requires 1 additional frame lookahead; 75% requires 3 additional frames lookahead.
        # Thus, the STFT/ISTFT loop introduces an algorithmic delay of n_fft - hop.
        assert n_fft % hop == 0  # This is only tested for 50% and 75% overlap
        d = n_fft - hop
        audio = audio[:, d : orig_len + d]
    return audio


@torch.no_grad
def enhance_diy(
    model: nn.Module, df_state: DF, audio: Tensor
):
    model.eval()

    nb_df = 96
    spec, erb_feat, spec_feat = df_features(audio, df_state, nb_df, device=get_device())

    enhanced = model(spec.clone(), erb_feat, spec_feat)[0].cpu()
    enhanced = as_complex(enhanced.squeeze(1))

    res = torch.as_tensor(df_state.synthesis(enhanced.numpy()))
    
    return res
    



# def maybe_download_model(name: str = DEFAULT_MODEL) -> str:
#     """Download a DeepFilterNet model.

#     Args:
#         - name (str): Model name. Currently needs to one of `[DeepFilterNet, DeepFilterNet2]`.

#     Returns:
#         - base_dir: Return the model base directory as string.
#     """
#     cache_dir = get_cache_dir()
#     if name.endswith(".zip"):
#         name = name.removesuffix(".zip")
#     model_dir = os.path.join(cache_dir, name)
#     if os.path.isfile(os.path.join(model_dir, "config.ini")) or os.path.isdir(
#         os.path.join(model_dir, "checkpoints")
#     ):
#         return model_dir
#     os.makedirs(cache_dir, exist_ok=True)
#     url = f"https://github.com/Rikorose/DeepFilterNet/raw/main/models/{name}"
#     download_file(url + ".zip", cache_dir, extract=True)
#     return model_dir


# def parse_epoch_type(value: str) -> Union[int, str]:
#     try:
#         return int(value)
#     except ValueError:
#         assert value in ("best", "latest")
#         return value


# class PrintVersion(argparse.Action):
#     def __init__(self, option_strings, dest):
#         super().__init__(
#             option_strings=option_strings,
#             dest=dest,
#             nargs=0,
#             required=False,
#             help="Print DeepFilterNet version information",
#         )

#     def __call__(self, *args):
#         print("DeepFilterNet", version)
#         exit(0)


# def setup_df_argument_parser(
#     default_log_level: str = "INFO", parser=None
# ) -> argparse.ArgumentParser:
#     if parser is None:
#         parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model-base-dir",
#         "-m",
#         type=str,
#         default=None,
#         help="Model directory containing checkpoints and config. "
#         "To load a pretrained model, you may just provide the model name, e.g. `DeepFilterNet`. "
#         "By default, the pretrained DeepFilterNet2 model is loaded.",
#     )
#     parser.add_argument(
#         "--pf",
#         help="Post-filter that slightly over-attenuates very noisy sections.",
#         action="store_true",
#     )
#     parser.add_argument(
#         "--output-dir",
#         "-o",
#         type=str,
#         default=None,
#         help="Directory in which the enhanced audio files will be stored.",
#     )
#     parser.add_argument(
#         "--log-level",
#         type=str,
#         default=default_log_level,
#         help="Logger verbosity. Can be one of (debug, info, error, none)",
#     )
#     parser.add_argument("--debug", "-d", action="store_const", const="DEBUG", dest="log_level")
#     parser.add_argument(
#         "--epoch",
#         "-e",
#         default="best",
#         type=parse_epoch_type,
#         help="Epoch for checkpoint loading. Can be one of ['best', 'latest', <int>].",
#     )
#     parser.add_argument("-v", "--version", action=PrintVersion)
#     return parser


# def run():
#     parser = setup_df_argument_parser()
#     parser.add_argument(
#         "--no-delay-compensation",
#         dest="compensate_delay",
#         action="store_false",
#         help="Dont't add some paddig to compensate the delay introduced by the real-time STFT/ISTFT implementation.",
#     )
#     parser.add_argument(
#         "--atten-lim",
#         "-a",
#         type=int,
#         default=None,
#         help="Attenuation limit in dB by mixing the enhanced signal with the noisy signal.",
#     )
#     parser.add_argument(
#         "noisy_audio_files",
#         type=str,
#         nargs="*",
#         help="List of noise files to mix with the clean speech file.",
#     )
#     parser.add_argument(
#         "--noisy-dir",
#         "-i",
#         type=str,
#         default=None,
#         help="Input directory containing noisy audio files. Use instead of `noisy_audio_files`.",
#     )
#     parser.add_argument(
#         "--no-suffix",
#         action="store_false",
#         dest="suffix",
#         help="Don't add the model suffix to the enhanced audio files",
#     )
#     parser.add_argument("--no-df-stage", action="store_true")
#     args = parser.parse_args()
#     main(args)


def ut():
    # initial model
    model, df_state = init_df()
    # print(model_name, "||", model_epoch)
    
    
    # load reference wav
    audio_path = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/input_noise_car_talk.wav"
    # audio_path = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/sine_20ms_1234hz.wav"
    # audio, _ = load_audio(audio_path, sr=48000)
    audio,sr = ta.load(audio_path)

    audio = F.pad(audio,(480,480),"constant",0)

    audio_len = audio.shape[1]
    block_len = 480*2
    audio_num = audio_len // 480 - 2

    overlap_cache = torch.zeros(1,480)

    res_tensor = torch.tensor([])

    # for idx in tqdm(range(audio_num)):
    #     tmp_audio = audio[0,idx*480:idx*480+block_len].unsqueeze(0)
        
    #     res_enhanced = enhance_diy(model, df_state, tmp_audio)
        
    #     output_res = (overlap_cache[:,0:480] + res_enhanced[:,0:480])
    #     res_tensor = torch.cat((res_tensor,output_res),dim=1)
    #     overlap_cache = overlap_cache[:] + res_enhanced[:,480:]
        # print(res_tensor.shape)

    

    # Denoise the audio
    idx=0
    block_len=480
    tmp_audio = audio[0,idx*480:idx*480+block_len].unsqueeze(0)
    tmp_audio = tmp_audio*0+1
    tmp_audio = F.pad(tmp_audio,(480,0),"constant",0)
    enhanced = enhance(model, df_state, tmp_audio,pad=False)
    
    

    # save result wav
    # save_audio("/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/result_noise_car_talk.wav", enhanced, sr)
    save_audio("/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/result_noise_car_talk.wav", res_tensor, sr)

    xxx = 1


# def make_sine():
#     import numpy as np
#     import soundfile as sf
#     freq = 1234.0
#     res = []
#     phase = 0
#     for idx in range(480*2):
#         res.append(np.sin(2.0*np.pi*freq*idx/48000.0))
    
#     sf.write("/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/sine_20ms_1234hz.wav",res,48000)
    
#     import matplotlib.pyplot as plt
#     plt.figure(1)
#     plt.plot(res)
#     plt.show()

#     xxx = 1


def vorbis_window(n):
    indices = torch.arange(n, dtype=torch.float64) + 0.5
    n_double = torch.tensor(n, dtype=torch.float64)
    window = torch.sin((torch.pi / 2.0) * torch.pow(torch.sin(indices / n_double * torch.pi), 2.0))
    
    return window


def analysis(audio, n_fft, hop):
        stft_norm = np.float32(1.0 / n_fft)
        spec_float = torch.stft(
                    audio, n_fft=n_fft, hop_length=hop, window=vorbis_window(n_fft),
                    return_complex=False, normalized=False, center=False
                ).transpose(1, 2) * stft_norm
        # spec_float = torch.stft(
        #     audio, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft),
        #     return_complex=False, normalized=False, center=False
        # ).transpose(1, 2) * stft_norm
        spec_cplx = torch.view_as_complex(spec_float)
        spec_float = spec_float.unsqueeze(0).to(torch.float32)
        return spec_float,spec_cplx


def analysis_fft(audio, vorbis_win):
    assert(max(audio.shape)==FFT_SIZE)
    fft_norm = np.float32(1.0 / FFT_SIZE)
    tmp_torch_cplx = torch.fft.rfft( torch.as_tensor(audio) * vorbis_win ,n=FFT_SIZE ) * fft_norm
    real_part = tmp_torch_cplx.real
    imag_part = tmp_torch_cplx.imag
    tmp_torch_float = torch.cat((real_part.unsqueeze(-1), imag_part.unsqueeze(-1)), dim=-1).unsqueeze(0).unsqueeze(0)
    tmp_torch_cplx = tmp_torch_cplx.unsqueeze(0)
    return tmp_torch_float.to(torch.float32), tmp_torch_cplx


def erb_function(spec, erb_m, db=True):
    output = np.abs(spec)
    output = np.matmul(output, erb_m)
    if db:
        output = 20 * np.log10(output + 1e-10)

    return output


def ut_analysis_spec(audio, df):
    # import matplotlib.pyplot as plt

    tmp_audio = F.pad(audio,(480,0),"constant",0)

    spec = df.analysis(audio.numpy()) 
    spec_ = analysis(tmp_audio, 960, 480)
    
    # fig, (ax1, ax2) = plt.subplots(1,2)
    # ax1.imshow(as_real(torch.as_tensor(spec[0, :, :32].T))[..., 0])  # lower spectrum part
    # ax2.imshow(as_real(spec_[0, :, :32].T)[..., 0])
    # plt.show()

    xxx = 1


def band_mean_norm_erb( xs, state, alpha):
    state = xs * (1. - alpha) + state * alpha
    xs = (xs - state) / 40.0
    return xs, state
    
def my_erb_norm(erb_log, alpha=0.99):
    b = erb_log.shape[2]
    state_ch0 = np.linspace(-60, -90, b)
    state = state_ch0

    norm_erb = np.zeros(erb_log.shape)
    for f_channel in range(erb_log.shape[0]):
        for in_step in range(erb_log.shape[1]):
            norm_erb[f_channel, in_step, :], state = band_mean_norm_erb(erb_log[f_channel, in_step, :], state, alpha)
    norm_erb = torch.as_tensor(norm_erb).unsqueeze(0).to(torch.float32)
    return norm_erb


def band_unit_norm(xs, state, alpha=0.99):
        xs_real = np.zeros(96)
        xs_imag = np.zeros(96)
        for idx in range(96):
            xs_real[idx] = xs[idx].real
            xs_imag[idx] = xs[idx].imag
        mag_abs = np.sqrt(xs_real ** 2 + xs_imag ** 2)
        state = np.sqrt(mag_abs * (1. - alpha) + state * alpha)
        xs_real = xs_real / state
        xs_imag = xs_imag / state
        xs = np.stack([xs_real, xs_imag],axis=1)

        return xs, state

def my_spec_norm(spec,alpha=0.99):

    f = spec.shape[-1]
    state_ch0 = np.linspace(0.001, 0.0001, 96, dtype=np.float32)
    state = state_ch0
    norm_unit = np.zeros((481,2), dtype=np.float32)
    for f_channel in range(spec.shape[0]):
        for in_step in range(spec.shape[1]):
            norm_unit, state = band_unit_norm(spec[f_channel, in_step, :], state, alpha)
    norm_unit = torch.as_tensor(norm_unit).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)
    return norm_unit




def ut_analysis_erb(audio,df):
    tmp_audio = F.pad(audio,(480,0),"constant",0)

    spec = df.analysis(audio.numpy()) 
    spec_ = analysis(tmp_audio, 960, 480)


    #origin df
    a = get_norm_alpha(False)
    erb_fb = df.erb_widths()
    erb_feat1 = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1)




    erb_self = ERB(
    samplerate=48000,
    fftsize=960,
    erbsize=32,
    minwidth=2,
    alpha=0.99)

    # x2 = torch.view_as_complex(spec).numpy()
    x2 = spec
    widths = erb_self.get_band_widths(48000,960,32,2)
    weights = erb_self.get_band_weights(48000,widths)
    erb_res = erb_function(spec,weights)


    print("df erb: \n",erb(spec, erb_fb))
    print("my erb: \n",erb_res)
    

    # mean = np.full(erb_res.shape[-1], erb_res[..., 0, :])
    alpha = 0.99
    erb_res = my_erb_norm(erb_res,alpha)
    print("df erb feat: \n",erb_feat1)
    print("my erb_feat: \n",erb_res)
    


    xxx = 1



def ut_erb_and_spec_feat(audio,df):
    tmp_audio = F.pad(audio,(480,0),"constant",0)

    spec = df.analysis(audio.numpy()) 
    # audio -> spec
    spec_,spec_cplx = analysis(tmp_audio, 960, 480)
    print("==========PART A START==================\n")
    print("origin spec: \n",spec[:,:,0:10])
    print("my spec: \n",spec_[:,:,0:10])
    print("==========PART A END==================\n")

    #origin df
    a = get_norm_alpha(False)
    erb_fb = df.erb_widths()
    erb_feat1 = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1)
    print("==========PART B START==================\n")
    print("origin df erb_feat: \n",erb_feat1)

    # erb feat
    erb_self = ERB(
        samplerate=48000,
        fftsize=960,
        erbsize=32,
        minwidth=2,
        alpha=0.99)

    widths = erb_self.get_band_widths(48000,960,32,2)
    weights = erb_self.get_band_weights(48000,widths)
    erb_feat = erb_function(spec_,weights)
    erb_feat = my_erb_norm(erb_feat,0.99)
    print("my erb feat: \n",erb_feat)
    print("==========PART B END==================\n")

    #spec feat
    df_spec_feat = as_real(torch.as_tensor(unit_norm(spec[..., :96], a)).unsqueeze(1))
    print("==========PART C START==================\n")
    print("origin df spec feat: \n",df_spec_feat[:,:,:,0:10,:])

    my_spec_feat = my_spec_norm(spec)
    print("my spec feat: \n",my_spec_feat[0:10,:])
    print("==========PART C END==================\n")


    xxx = 1


def ut_fft_ifft_():
    import matplotlib.pyplot as plt

    audio = np.ones((1,480))
    tmp_audio = np.concatenate((np.zeros((1,480)), audio),axis=1)

    vorbis_win = vorbis_window(960)
    FFT_SIZE = 960

    #numpy rfft and irfft
    tmp_np_fft = np.fft.rfft(tmp_audio*vorbis_win.numpy(),n=FFT_SIZE)/FFT_SIZE
    tmp_np_ifft = np.fft.irfft(tmp_np_fft,n=FFT_SIZE)*FFT_SIZE


    #torch rfft and irfft
    tmp_torch_fft = torch.fft.rfft( torch.as_tensor(tmp_audio) * vorbis_win ,n=FFT_SIZE ) / FFT_SIZE
    tmp_torch_ifft = torch.fft.irfft( tmp_torch_fft, n=FFT_SIZE ) * FFT_SIZE


    #torch stft and istft
    tmp_torch_stft = torch.stft(
                    torch.as_tensor(tmp_audio), n_fft=FFT_SIZE, hop_length=480, window=vorbis_window(FFT_SIZE),
                    return_complex=True, normalized=False, center=False
                ).transpose(1, 2) / FFT_SIZE
    tmp_torch_stft_ifft = torch.fft.irfft( tmp_torch_stft,n=FFT_SIZE ) * FFT_SIZE


    #df analysis
    model,state = init_df()
    fft = state.analysis(np.float32(tmp_audio))


    plt.figure(1)
    # plt.plot(tmp_np_ifft[0,:])
    plt.plot(tmp_torch_ifft[0,:].numpy())
    plt.plot(vorbis_win.numpy())
    # plt.plot(tmp_np_ifft[0,:] - tmp_torch_stft_ifft[0,0,:].numpy())

    plt.plot()

    plt.show()


    xxx = 1



def ut2():
    #test rt_filter

    model, state = init_df()

    model.eval()

    # x, sr = read('/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/x.wav', state.sr())
    # x = x[:,0:480*1]*0+1
    # load reference wav
    audio_path = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/input_noise_car_talk.wav"
    # audio_path = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/sine_20ms_1234hz.wav"
    # audio, _ = load_audio(audio_path, sr=48000)
    audio,sr = ta.load(audio_path)
    audio = audio[:,0:480]*0+1



    # ut_analysis_spec(audio,state) #ok
    # ut_analysis_erb(audio,state) #ok
    ut_erb_and_spec_feat(audio,state)



    # write('/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/y.wav', sr, y)



    xxx = 1







def ut3():
    # test my feature push in model and result

    #define variable
    NB_DF = 96
    FFT_SIZE = 960

    # init model
    model, state = init_df()
    model.eval()

    # load wav
    audio_path = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/input_noise_car_talk.wav"
    # audio_path = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/sine_20ms_1234hz.wav"
    # audio, _ = load_audio(audio_path, sr=48000)
    audio,sr = ta.load(audio_path)
    

    audio_num = 1

    # init my_feat
    my_erb_calc = ERB(
        samplerate=48000,
        fftsize=960,
        erbsize=32,
        minwidth=1,
        alpha=0.99)
    widths = my_erb_calc.get_band_widths(48000,960,32,1)
    erb_matrix = erb_fb_function(widths,48000)
    weights = my_erb_calc.get_band_weights(48000,widths)


    # processBlock
    for idx in range(audio_num):
        tmp_audio = audio[:,idx*480:(idx+1)*480]
        tmp_pad_audio = F.pad(tmp_audio,(480,0),"constant",0)

        # origin df erb&spec feature
        df_spec = state.analysis(tmp_audio.numpy()) 
        df_spec_res, df_erb_feat, df_spec_feat = df_features(tmp_audio, state, NB_DF, device=get_device())
        # print("df_spec_res.shape:",df_spec_res.shape)
        # print("df_erb_feat.shape:",df_erb_feat.shape)
        # print("df_spec_feat.shape:",df_spec_feat.shape)
        df_enhanced = model(df_spec_res.clone(), df_erb_feat, df_spec_feat)[0].cpu()
        df_enhanced_cplx = torch.view_as_complex(df_enhanced)
        df_enhanced_ifft = torch.fft.irfft(df_enhanced_cplx,n=FFT_SIZE)*FFT_SIZE
        


        # my erb&spec feature
        # my_spec_float, my_spec_cplx = analysis(tmp_pad_audio, 960, 480)
        spec = torch.as_tensor(state.analysis(tmp_audio.numpy())).unsqueeze(0).to(torch.float32)
        my_erb_feat = torch.matmul(spec.abs(), erb_matrix)
        my_spec_feat = my_spec_norm(my_spec_cplx)
        # my_erb_feat = erb_function(my_spec_cplx, weights)
        # my_erb_feat = my_erb_norm(my_erb_feat,0.99)
        # my_erb_feat = torch.matmul(my_spec_cplx.unsqueeze(0).abs().square().to(torch.float32), erb_matrix)
        # my_spec_feat = my_spec_norm(my_spec_cplx)
        # print("my_spec_float.shape:",my_spec_float.shape)
        # print("my_erb_feat.shape:",my_erb_feat.shape)
        # print("my_spec_feat.shape:",my_spec_feat.shape)
        my_enhanced = model(my_spec_float, my_erb_feat, my_spec_feat)[0].cpu()
        my_enhanced_cplx = torch.view_as_complex(my_enhanced)
        my_enhanced_ifft = torch.fft.irfft(my_enhanced_cplx,n=FFT_SIZE)*FFT_SIZE
        

        print("========================================\n")
        # ok
        print("origin spec: \n",df_spec[:,:,0:10])
        # print("origin spec view_as_real:\n",df_spec_res[:,:,:,0:10])
        print("my spec: \n",my_spec_float[:,:,:,0:10])
        # print("========================================\n")
        # had a little diff
        print("origin df erb_feat: \n",df_erb_feat)
        print("my erb feat: \n",my_erb_feat)
        # print("========================================\n")
        # ok
        print("origin df spec feat: \n",df_spec_feat[:,:,:,0:10,:])
        print("my spec feat: \n",my_spec_feat[:,:,:,0:10,:])
        print("========================================\n")
        # ok
        print("df_enhanced: \n",df_enhanced[:,:,:,0:10,:])
        print("my_enhanced: \n",my_enhanced[:,:,:,0:10,:])
        print("========================================\n")

        print("df_enhanced_cplx: \n",df_enhanced_cplx[:,:,:,0:20])
        print("my_enhanced_cplx: \n",my_enhanced_cplx[:,:,:,0:20])
        print("========================================\n")
        
        print("df_enhanced_ifft: \n",df_enhanced_ifft[:,:,:,0:20])
        print("my_enhanced_ifft: \n",my_enhanced_ifft[:,:,:,0:20])
        print("========================================\n")

        xxx = 1





def ut4():
    # test overlap mode and run model in a little diff in erb
    import matplotlib.pyplot as plt

    # define variable
    NB_DF           = 96
    AUDIO_SIZE_10ms = 480
    FFT_SIZE        = 960
    HOP_SIZE        = 480
    ERB_SIZE        = 32
    MIN_WIDTH       = 2
    SAMPLERATE      = 48000
    ALPHA           = 0.99

    # load audio
    audio_path = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/input_noise_car_talk.wav"
    audio,sr = ta.load(audio_path)
    # audio = audio[:,0:SAMPLERATE*5]*0+1

    AUDIO_NUM = max(audio.shape)//480-1
    output = torch.zeros((1,AUDIO_NUM*AUDIO_SIZE_10ms))
    output_df = torch.zeros((1,AUDIO_NUM*AUDIO_SIZE_10ms))

    # init model
    model, state = init_df()
    model.eval()

    # init erb
    my_erb_calc = ERB(
        samplerate=SAMPLERATE,
        fftsize=FFT_SIZE,
        erbsize=ERB_SIZE,
        minwidth=MIN_WIDTH,
        alpha=ALPHA)
    widths = my_erb_calc.get_band_widths(SAMPLERATE,FFT_SIZE,ERB_SIZE,MIN_WIDTH)
    weights = my_erb_calc.get_band_weights(SAMPLERATE,widths)

    # init overlap buffer
    process_buffer_960 = torch.zeros((1,FFT_SIZE))
    tmp_audio = torch.zeros((1,AUDIO_SIZE_10ms))
    cache_buffer = torch.zeros((1,AUDIO_SIZE_10ms))
    cache_buffer_df = torch.zeros((1,AUDIO_SIZE_10ms))

    # init FIR window
    vorbis_win = vorbis_window(FFT_SIZE)

    # df result
    # df_enhanced = enhance(model, state, audio, pad=False)
    df_spec, df_erb_feat, df_spec_feat = df_features(audio[:,:AUDIO_NUM*AUDIO_SIZE_10ms], state, NB_DF, device=get_device())
    # output_df[:,idx*AUDIO_SIZE_10ms:(idx+1)*AUDIO_SIZE_10ms] = df_enhanced


    for idx in tqdm(range(AUDIO_NUM)):
        tmp_audio = audio[:,idx*AUDIO_SIZE_10ms:(idx+1)*AUDIO_SIZE_10ms].clone()
        process_buffer_960[:,:AUDIO_SIZE_10ms] = process_buffer_960[:,AUDIO_SIZE_10ms:].clone()
        process_buffer_960[:,AUDIO_SIZE_10ms:AUDIO_SIZE_10ms*2] = tmp_audio.clone()

        # my_spec_float, my_spec_cplx = analysis(process_buffer_960, FFT_SIZE, HOP_SIZE)
        my_spec_float, my_spec_cplx = analysis_fft(process_buffer_960, vorbis_win)
        my_erb_feat = erb_function(my_spec_cplx, weights)
        my_erb_feat = my_erb_norm(my_erb_feat,ALPHA)
        my_spec_feat = my_spec_norm(my_spec_cplx)

        df_spec_float_tmp = df_spec[:,:,idx,:,:].unsqueeze(0)
        df_erb_feat_tmp = df_erb_feat[:,:,idx,:].unsqueeze(0)
        df_spec_feat_tmp = df_spec_feat[:,:,idx,:,:].unsqueeze(0)

        # print("my_spec_float:\n",my_spec_float[:,:,:,0:10,:])
        # print("df_spec_float:\n",df_spec_float_tmp[:,:,:,0:10,:])
        print("my_erb_feat:\n",my_erb_feat[:,:,:,0:10])
        print("df_erb_feat:\n",df_erb_feat_tmp[:,:,:,0:10])
        print("my_spec_feat:\n",my_spec_feat[:,:,:,0:10,:])
        print("df_spec_feat:\n",df_spec_feat_tmp[:,:,:,0:10,:])

        # model process
        my_enhanced = model(my_spec_float, my_erb_feat, my_spec_feat)[0].cpu()
        df_enhanced = model(df_spec_float_tmp, df_erb_feat_tmp, df_spec_feat_tmp)[0].cpu()
        
        my_enhanced_cplx = torch.view_as_complex(my_enhanced)
        my_enhanced_ifft = torch.fft.irfft(my_enhanced_cplx,n=FFT_SIZE)*FFT_SIZE
        # my_enhanced_ifft = torch.fft.irfft(my_spec_cplx,n=FFT_SIZE)*FFT_SIZE
        df_enhanced_cplx = torch.view_as_complex(df_enhanced)
        df_enhanced_ifft = torch.fft.irfft(df_enhanced_cplx,n=FFT_SIZE)*FFT_SIZE

        # output: result[0:480]+cahce
        output[:,idx*AUDIO_SIZE_10ms:(idx+1)*AUDIO_SIZE_10ms] = cache_buffer + my_enhanced_ifft[0,0,:,0:AUDIO_SIZE_10ms]
        # output[:,idx*AUDIO_SIZE_10ms:(idx+1)*AUDIO_SIZE_10ms] = cache_buffer + my_enhanced_ifft[0,:,0:AUDIO_SIZE_10ms]
        # updaye cahce: result[480:960]
        cache_buffer = my_enhanced_ifft[0,0,:,AUDIO_SIZE_10ms:].clone()
        # cache_buffer = my_enhanced_ifft[0,:,AUDIO_SIZE_10ms:].clone()

        output_df[:,idx*AUDIO_SIZE_10ms:(idx+1)*AUDIO_SIZE_10ms] = cache_buffer_df + df_enhanced_ifft[0,0,:,0:AUDIO_SIZE_10ms]
        cache_buffer_df = df_enhanced_ifft[0,0,:,AUDIO_SIZE_10ms:].clone()


    

    # plt.figure(1)
    # plt.plot(audio[0,:].detach().numpy())
    # plt.plot(df_enhanced[0,:].detach().numpy())
    # plt.plot(output[0,:].detach().numpy())
    # plt.show()


    save_path = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/my_process_result.wav"
    ta.save(save_path,output,SAMPLERATE)
    save_path2 = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/df_process_result.wav"
    ta.save(save_path2,output_df,SAMPLERATE)


    print("process finished")
    xxx = 1


def test_df_erb_and_erb_function():
    #define variable
    NB_DF = 96
    FFT_SIZE = 960

    # init model
    model, state = init_df()
    model.eval()


    # init my_erb_matrix
    my_erb_calc = ERB(
        samplerate=48000,
        fftsize=960,
        erbsize=32,
        minwidth=1,
        alpha=0.99)
    widths = my_erb_calc.get_band_widths(48000,960,32,1)
    erb_matrix = erb_fb_function(widths,48000)

    # input
    # audio = torch.randn((1,1,1,960),dtype=torch.float32)
    spec = torch.randn((1, 1, 1, 481), dtype=torch.complex64)

    # df erb res
    # df_state = DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32)
    # erb_matrix = erb_fb_function(df_state.erb_widths(), 48000)
    
    df_erb_fb = state.erb_widths()
    # df_erb_fb = df_state.erb_widths()
    df_erb = torch.as_tensor(erb(spec.squeeze(0).numpy(), df_erb_fb))
    df_erb_feat = torch.as_tensor(erb_norm(df_erb.numpy() ,0.99))

    # my erb res
    my_erb_feat = torch.matmul(spec.abs().square(), erb_matrix)

    xxx = 1


def filter_RT_2frame(model, all_model , audio):
    cpu = torch.device('cpu')

    param_sr = 48000
    param_fft_size = 960
    param_hop_size = 480
    param_fft_bins = 481
    param_erb_bins = 32
    param_erb_min_width = 2
    param_deep_filter_bins = 96
    param_norm_alpha = 0.99

    assert getattr(model, 'freq_bins', param_fft_bins) == param_fft_bins
    assert getattr(model, 'erb_bins', param_erb_bins) == param_erb_bins
    assert getattr(model, 'nb_df', getattr(model, 'df_bins', param_deep_filter_bins)) == param_deep_filter_bins
    # assert state.sr() == param_sr
    # assert len(state.erb_widths()) == param_erb_bins

    print(dict(
        sr=param_sr,
        fft_size=param_fft_size,
        hop_size=param_hop_size,
        fft_bins=param_fft_bins,
        erb_bins=param_erb_bins,
        erb_min_width=param_erb_min_width,
        deep_filter_bins=param_deep_filter_bins,
        norm_alpha=param_norm_alpha))

    stft = STFT(
        framesize=param_fft_size,
        hopsize=param_hop_size,
        window='vorbis')

    erb = ERB(
        samplerate=param_sr,
        fftsize=param_fft_size,
        erbsize=param_erb_bins,
        minwidth=param_erb_min_width,
        alpha=param_norm_alpha)

    cpx = CPX(
        cpxsize=param_deep_filter_bins,
        alpha=param_norm_alpha)
    
    # =======================
    all_erb = ERB(
        samplerate=param_sr,
        fftsize=param_fft_size,
        erbsize=param_erb_bins,
        minwidth=param_erb_min_width,
        alpha=param_norm_alpha)

    all_cpx = CPX(
        cpxsize=param_deep_filter_bins,
        alpha=param_norm_alpha)
    # =======================
    
    audio_num = audio.__len__()//480

    input_buffer_1440 = np.zeros(480*3)
    # cache_1440 = np.zeros(480*3)
    cache_480 = np.zeros(480)
    res = np.zeros((audio_num+1)*480)

    # =====================================
    # all audio stft
    pad_0_audio = np.pad(audio,(480,0))
    # xxxx = np.linspace(0,480*3,480*3)
    all_stft = stft.stft(pad_0_audio)
    # xxxxx = stft.istft(all_stft)
    all_real_part = all_stft.real
    all_imaginary_part = all_stft.imag
    all_combined_array = np.stack((all_real_part, all_imaginary_part), axis=-1)
    all_spec_float = torch.as_tensor(all_combined_array,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    x = torch.view_as_complex(all_spec_float).numpy()
    y = all_erb(x)
    all_erb_feat = torch.from_numpy(y.astype(np.float32))

    x = torch.view_as_complex(all_spec_float).numpy()
    y = all_cpx(x)
    y = np.stack((y.real, y.imag), axis=-1)
    all_spec_feat = torch.from_numpy(y.astype(np.float32))

    all_output = all_model(all_spec_float, all_erb_feat, all_spec_feat) # orig: spec.clone()
    all_enhanced = all_output[0].cpu()
    # print('all_enhanced', all_enhanced.shape, all_enhanced.dtype)
    all_enhanced = all_enhanced.squeeze(1)
    # print('all_enhanced squeeze', all_enhanced.shape, all_enhanced.dtype)
    all_enhanced = torch.view_as_complex(all_enhanced) # orig: as_complex
    
    all_cache_480 = np.zeros(480)
    all_res = np.zeros(480 * (all_enhanced.shape[1]+1))
    # for idx in range(all_enhanced.shape[1]):
        # # tmp_ifft_960 = stft.istft(all_enhanced[0,idx,:].detach().numpy())*960
        # # tmp_ifft_960 = np.fft.irfft(all_enhanced[0,idx,:].detach().numpy())*960
        # tmp_ifft_960 = np.fft.irfft(all_enhanced[0,idx,:].detach().numpy(), axis=-1, norm='forward') * stft.get_window()
        # all_res[480*idx : 480*(idx+1)] = all_cache_480 + tmp_ifft_960[:480]
        # all_cache_480 = tmp_ifft_960[480:]

    # =====================================

    write_idx = 0

    # process for half of audio frame num
    for idx in tqdm(range(audio_num//2)):
        # push new data into input_buffer
        input_buffer_1440[:480] = input_buffer_1440[480*2:].copy()
        input_buffer_1440[480:] = audio[ idx*960:(idx+1)*960 ].copy()

        # spec_floar
        tmp_spec = stft.stft(input_buffer_1440)
        real_part = tmp_spec.real
        imaginary_part = tmp_spec.imag
        combined_array = np.stack((real_part, imaginary_part), axis=-1)
        spec_float = torch.as_tensor(combined_array,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # print("diff in stft A: ", (all_spec_float[0,0,idx*2,:,:] - spec_float[0,0,0,:,:])[0:10,:])
        # print("diff in stft B: ", (all_spec_float[0,0,idx*2+1,:,:] - spec_float[0,0,1,:,:])[0:20,:])

        # erb_feat
        x = torch.view_as_complex(spec_float).numpy()
        y = erb(x)
        erb_feat = torch.from_numpy(y.astype(np.float32))
        # print("diff in erb A: ", (all_erb_feat[0,0,idx*2,:] - erb_feat[0,0,0,:])[0:10])
        # print("diff in stft B: ", (all_spec_float[0,0,idx*2+1,:,:] - spec_float[0,0,1,:,:])[0:20,:])

        # spec_feat
        x = torch.view_as_complex(spec_float).numpy()
        y = cpx(x)
        y = np.stack((y.real, y.imag), axis=-1)
        spec_feat = torch.from_numpy(y.astype(np.float32))
        # print("diff in specfeat A: ", (all_spec_feat[0,0,idx*2,:,:] - spec_feat[0,0,0,:,:])[0:10,:])
        
        # model process
        try:
            print("diff in stft A: ", (all_spec_float[0,0,idx*2,:,:] - spec_float[0,0,0,:,:])[0:10,:])
            print("diff in stft B: ", (all_spec_float[0,0,idx*2+1,:,:] - spec_float[0,0,1,:,:])[0:10,:])
            print("diff in erb A: ", (all_erb_feat[0,0,idx*2,:] - erb_feat[0,0,0,:])[0:10])
            print("diff in erb B: ", (all_erb_feat[0,0,idx*2+1,:] - erb_feat[0,0,1,:])[0:10])
            print("diff in specfeat A: ", (all_spec_feat[0,0,idx*2,:,:] - spec_feat[0,0,0,:,:])[0:10,:])
            print("diff in specfeat B: ", (all_spec_feat[0,0,idx*2+1,:,:] - spec_feat[0,0,1,:,:])[0:10,:])
            tmp_output = model(spec_float, erb_feat, spec_feat)
            # tmp_output = model(all_spec_float[:,:,idx*2:idx*2+2,:,:], all_erb_feat[:,:,idx*2:idx*2+2,:], all_spec_feat[:,:,idx*2:idx*2+2,:,:])
            print("output A spec_float diff: ",all_output[0][0,0,2*idx,0:15,:] - tmp_output[0][0,0,0,0:15,:])
            print("output B spec_float diff: ",all_output[0][0,0,2*idx+1,0:15,:] - tmp_output[0][0,0,1,0:15,:])
            # print("tmp_output: ",)
            
            # print("now frame[",idx,"] is done")
        except RuntimeError as e:
            print("Error during model inference:", e)
            print("Checking tensor sizes...")
            raise




        enhanced = tmp_output[0].cpu()
        # print('enhanced', enhanced.shape, enhanced.dtype)
        enhanced = enhanced.squeeze(1)
        # print('enhanced squeeze', enhanced.shape, enhanced.dtype)
        enhanced = torch.view_as_complex(enhanced) # orig: as_complex
        
        # tmp_irfft = stft.istft(enhanced.squeeze(0).detach().numpy())

        #===========
        # idx in all enhanced
        all_irfft_partA_960 = np.fft.irfft(all_enhanced[0,2*idx,:].detach().numpy(), axis=-1, norm='forward') * stft.get_window()
        all_irfft_partB_960 = np.fft.irfft(all_enhanced[0,2*idx+1,:].detach().numpy(), axis=-1, norm='forward') * stft.get_window()
        #===========

        # tmp_irfft_partA_960 = np.fft.irfft(all_enhanced[0,2*idx,:].detach().numpy(), axis=-1, norm='forward') * stft.get_window()
        # tmp_irfft_partB_960 = np.fft.irfft(all_enhanced[0,2*idx+1,:].detach().numpy(), axis=-1, norm='forward') * stft.get_window()
        tmp_irfft_partA_960 = np.fft.irfft(enhanced[0,0,:].detach().numpy(), axis=-1, norm='forward') * stft.get_window()
        tmp_irfft_partB_960 = np.fft.irfft(enhanced[0,1,:].detach().numpy(), axis=-1, norm='forward') * stft.get_window()
        
        print("partA diff:\n ",(all_irfft_partA_960 - tmp_irfft_partA_960)[5:30])
        print("partB diff:\n ",(all_irfft_partB_960 - tmp_irfft_partB_960)[5:30])

        # res[960*idx : 960*idx+480] = cache_480 + tmp_irfft_partA_960[:480]
        # cache_480 = tmp_irfft_partA_960[480:].copy()
        # res[960*idx+480 : 960*(idx+1)] = cache_480 + tmp_irfft_partB_960[:480]
        # cache_480 = tmp_irfft_partB_960[480:].copy()

        
        res[write_idx:write_idx+960] += tmp_irfft_partA_960
        all_res[write_idx:write_idx+960] += all_irfft_partA_960
        write_idx += 480
        
        res[write_idx:write_idx+960] += tmp_irfft_partB_960
        all_res[write_idx:write_idx+960] += all_irfft_partB_960
        write_idx += 480

        # res *= 5
    
    return res,all_res



    xxx = 1


def ut5():
    model, state = init_df()
    model.eval()

    all_model,all_state = init_df()
    all_model.eval()

    audio, sr = read('/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/input_noise_car_talk.wav', state.sr())
    audio = np.squeeze(audio,0)
    # audio = np.linspace(1,48000,48000,dtype=np.float64)
    

    res, all_res = filter_RT_2frame(model, all_model, audio)
    write('/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/input_noise_car_talk_Realtime_res.wav', sr, res)
    write('/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/input_noise_car_talk_all_res.wav', sr, all_res)





if __name__ == "__main__":

    # make_sine()
    # run()
    # ut()
    # ut2()
    # ut_fft_ifft_()
    # test_df_erb_and_erb_function()
    # ut3()
    # ut4()
    ut5()