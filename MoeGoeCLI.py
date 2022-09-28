import sys, re
from torch import no_grad, LongTensor
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

import commons
from japanese_g2p import japanese_g2p
import json
from pathlib import Path
import utils
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from mel_processing import spectrogram_torch

from scipy.io.wavfile import write
from MoeGoe import print_speakers, get_speaker_id, get_label_value, get_label
import argparse


def cli():
    parser = argparse.ArgumentParser(description='MoeGeoCLI')
    moe = parser.add_subparsers(metavar="MoeGeoCLI")
    tts = moe.add_parser("tts")
    tts.add_argument("-m", '--model', type=str, required=True, help='model path')
    tts.add_argument("-c", '--config', type=str, required=False, help='config path, default model path config.json')
    tts.add_argument("-s", '--speaker', type=str, required=False, help='speaker id')
    tts.add_argument("-t", '--text', type=str, required=True, help='text')
    tts.add_argument("-o", '--output', type=str, required=True, help='wav output path')
    tts.set_defaults(handle=handle_tts)

    vc = moe.add_parser("vc")
    vc.add_argument("-m", '--model', type=str, required=True, help='model path')
    vc.add_argument("-c", '--config', type=str, required=False, help='config path, default model path config.json')
    vc.set_defaults(handle=handle_vc)
    args = parser.parse_args()
    if hasattr(args, 'handle'):
        args.handle(args)
    else:
        parser.print_help()
    pass


def handle_tts(args):
    handle(args, "t")


def handle_vc(args):
    handle(args, "v")


def text_to_sequence(text, symbols=None):
    _symbol_to_id = {s: i for i, s in enumerate(symbols)}
    seq = [_symbol_to_id[c] for c in text if c in symbols]
    return seq


def get_text(text, hps, symbols):
    text_norm = text_to_sequence(text, symbols)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def handle(args, choice):
    model = args.model
    config = (Path(model).parent.joinpath("config.json")) if args.config is None else args.config
    speaker_id = int(args.speaker)
    text = args.text
    out_path = args.output
    do_model(model,config,speaker_id,text, out_path, choice)


def do_model(model, config, speaker_id, text, out_path, choice):
    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    if n_symbols == 0 and Path(model).parent.joinpath("moetts.json").exists():
        with open(str(Path(model).parent.joinpath("moetts.json")), 'rb') as load_f:
            load_dict = json.load(load_f)
            hps_ms.symbols = load_dict['symbols']
            n_symbols = len(hps_ms.symbols)
        pass
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    if speakers == ['0'] and Path(model).parent.joinpath("moetts.json").exists():
        with open(str(Path(model).parent.joinpath("moetts.json")), 'rb') as load_f:
            load_dict = json.load(load_f)
            s = str(load_dict['speakers']).split("\n")
            speakers = [r.split(" ")[0] for r in s]
        pass
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    if n_symbols != 0:
        if choice == 't':
            if text == '[ADVANCED]':
                text = input('Raw text:')
                print('Cleaned text is:')
                print(_clean_text(text, hps_ms.data.text_cleaners))
                return

            length_scale, text = get_label_value(text, 'LENGTH', 1, 'length scale')
            noise_scale, text = get_label_value(text, 'NOISE', 0.667, 'noise scale')
            noise_scale_w, text = get_label_value(text, 'NOISEW', 0.8, 'deviation of noise')
            cleaned, text = get_label(text, 'CLEANED')
            stn_tst = get_text(japanese_g2p.get_romaji_with_space(text), hps_ms, hps_ms.symbols)

            with no_grad():
                x_tst = stn_tst.cpu().unsqueeze(0)
                x_tst_lengths = LongTensor([stn_tst.size(0)]).cpu()
                sid = LongTensor([speaker_id])
                audio = \
                    net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                   noise_scale_w=noise_scale_w,
                                   length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
            write(out_path, hps_ms.data.sampling_rate, audio)

        elif choice == 'v':
            audio_path = input('Path of an audio file to convert:\n')
            print_speakers(speakers)
            audio = utils.load_audio_to_torch(audio_path, hps_ms.data.sampling_rate)

            originnal_id = get_speaker_id('Original speaker ID: ')
            target_id = get_speaker_id('Target speaker ID: ')
            out_path = input('Path to save: ')

            y = audio.unsqueeze(0)

            spec = spectrogram_torch(y, hps_ms.data.filter_length,
                                     hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                                     center=False)
            spec_lengths = LongTensor([spec.size(-1)])
            sid_src = LongTensor([originnal_id])

            with no_grad():
                sid_tgt = LongTensor([target_id])
                audio = net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                    0, 0].data.cpu().float().numpy()
            write(out_path, hps_ms.data.sampling_rate, audio)

    else:
        from hubert_model import hubert_soft
        hubert = hubert_soft(model)

        while True:
            audio_path = input('Path of an audio file to convert:\n')
            print_speakers(speakers)

            import librosa
            if use_f0:
                audio, sampling_rate = librosa.load(audio_path, sr=hps_ms.data.sampling_rate, mono=True)
                audio16000 = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            else:
                audio16000, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)

            target_id = get_speaker_id('Target speaker ID: ')
            out_path = input('Path to save: ')
            length_scale, out_path = get_label_value(out_path, 'LENGTH', 1, 'length scale')
            noise_scale, out_path = get_label_value(out_path, 'NOISE', 0.1, 'noise scale')
            noise_scale_w, out_path = get_label_value(out_path, 'NOISEW', 0.1, 'deviation of noise')

            from torch import inference_mode, FloatTensor
            import numpy as np
            with inference_mode():
                units = hubert.units(FloatTensor(audio16000).unsqueeze(0).unsqueeze(0)).squeeze(0).numpy()
                if use_f0:
                    f0_scale, out_path = get_label_value(out_path, 'F0', 1, 'f0 scale')
                    f0 = librosa.pyin(audio, sr=sampling_rate,
                                      fmin=librosa.note_to_hz('C0'),
                                      fmax=librosa.note_to_hz('C7'),
                                      frame_length=1780)[0]
                    target_length = len(units[:, 0])
                    f0 = np.nan_to_num(np.interp(np.arange(0, len(f0) * target_length, len(f0)) / target_length,
                                                 np.arange(0, len(f0)), f0)) * f0_scale
                    units[:, 0] = f0 / 10

            stn_tst = FloatTensor(units)
            with no_grad():
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = LongTensor([stn_tst.size(0)])
                sid = LongTensor([target_id])
                audio = \
                    net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                   length_scale=length_scale)[0][0, 0].data.float().numpy()
            write(out_path, hps_ms.data.sampling_rate, audio)


if __name__ == '__main__':
    cli()
