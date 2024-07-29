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

# from audio import read, write
# from cpx import CPX
from erb import ERB
# from spectrum import spectrogram, erbgram
# from stft import STFT
import numpy as np
# import matplotlib.pyplot as plot
# from erb_self import ERB_self

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
        min_nb_erb_freqs=2,
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


def df_features(audio: Tensor, df: DF, nb_df: int, device=None) -> Tuple[Tensor, Tensor, Tensor]:
    spec = df.analysis(audio.numpy())  # [C, Tf] -> [C, Tf, F]
    a = get_norm_alpha(False)
    erb_fb = df.erb_widths()
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

    for idx in tqdm(range(audio_num)):
        tmp_audio = audio[0,idx*480:idx*480+block_len].unsqueeze(0)
        
        res_enhanced = enhance_diy(model, df_state, tmp_audio)
        
        output_res = (overlap_cache[:,0:480] + res_enhanced[:,0:480])
        res_tensor = torch.cat((res_tensor,output_res),dim=1)
        overlap_cache = overlap_cache[:] + res_enhanced[:,480:]
        # print(res_tensor.shape)

    

    # Denoise the audio
    # enhanced = enhance(model, df_state, audio)
    
    

    # save result wav
    # save_audio("/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/result_noise_car_talk.wav", enhanced, sr)
    save_audio("/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/result_noise_car_talk.wav", res_tensor, sr)

    xxx = 1


def make_sine():
    import numpy as np
    import soundfile as sf
    freq = 1234.0
    res = []
    phase = 0
    for idx in range(480*2):
        res.append(np.sin(2.0*np.pi*freq*idx/48000.0))
    
    sf.write("/Users/donkeyddddd/Documents/Rx_projects/git_projects/DeepFilterNet/assets/sine_20ms_1234hz.wav",res,48000)
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(res)
    plt.show()

    xxx = 1


def vorbis_window(n):
    indices = torch.arange(n, dtype=torch.float32)
    window = torch.sin((torch.pi / 2.0) * torch.pow(torch.sin(indices / float(n) * torch.pi), 2.0))
    return window


def analysis(audio, n_fft, hop):
        stft_norm = 1 / (n_fft ** 2 / (2 * hop))
        spec = torch.stft(
                    audio, n_fft=n_fft, hop_length=hop, window=vorbis_window(n_fft),
                    return_complex=True, normalized=False, center=False
                ).transpose(1, 2) * stft_norm
        return spec


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
    spec_ = analysis(tmp_audio, 960, 480)
    print("==========PART A START==================\n")
    print("origin spec: \n",spec[:,:,0:20])
    print("my spec: \n",spec_[:,:,0:20])
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
    print("origin df spec feat: \n",df_spec_feat[:,:,:,0:96,:])

    my_spec_feat = my_spec_norm(spec)
    print("my spec feat: \n",my_spec_feat[0:96,:])
    print("==========PART C END==================\n")


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



if __name__ == "__main__":

    # make_sine()
    # run()
    # ut()
    ut2()
