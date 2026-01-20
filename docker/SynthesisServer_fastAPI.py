import os.path
import shutil
import time
import zipfile

import uvicorn
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import re
import numpy as np
import torch
from os import path, system
import wave
import logging

MAX_WAV_VALUE = 32768.0
import gc
import yaml
import json

#from load_csv import load_csv
from def_symbols import init_symbols, text_to_sequence
from model import Tacotron2, to_gpu
from py3nvml.py3nvml import *
from pydantic import BaseModel
app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#Gestione dei parametri, conversione di argparse usando Pydantic
class SynthesizeRequest(BaseModel):
    allTextIn: str = ""  #parametro della funzione, input dell'utente
    outputDir: str = "_out"  #directory di output (default)
    config: str = "new_models/tc2_italian.yaml"
    numGpu: int = 0
    noAutoNumbering: bool = False
    prediction: bool = False
    playWav: bool = False
    draw: bool = False
    overWrite: bool = False
    groundTruth: bool = False
    parameterFiles: bool = False
    tacotron: str = "new_models/tacotron2_IT"
    vocoder: str = "new_models/hifigan_IT"
    speaker: str = "LC"
    style: str = "NONE"
    samplingRate: int = 22050
    exe: str = ""
    hparams: str = None
    phoneticOnly: bool = False
    silent: bool = False


class synthesisServer(object):
    hps = []
    exe = "/research/crissp/LPCNet-master/lpcnet_demo_NEB -synthesis"
    spk_imposed = -1
    style_imposed = -1

    def check_gpu(self, msg, num_gpu):
        mo = pow(1024.0, 2)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        if torch.cuda.is_available():
            handle = nvmlDeviceGetHandleByIndex(num_gpu)
            info = nvmlDeviceGetMemoryInfo(handle)
            print(
                "Device {}: {} Free_memory={}/{}MiB".format(
                    num_gpu, nvmlDeviceGetName(handle), info.free >> 20, info.total >> 20
                )
            )

    def check_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                        hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    print("TENSOR> {} [{}]".format(type(obj), obj.size()))
            except:
                pass

    def synthesis(self, nm):
        if self.hps["ext_data"] == "LPCNet":
            cmd = "{} {}.f32 {}.pcm".format(self.exe, nm, nm)
            system(cmd)
            print(cmd)
            cmd = "sox -t raw -b 16 -e signed-integer -r 16000 {}.pcm {}.wav".format(nm, nm)
            system(cmd)
            print(cmd)
            print("{}.wav created".format(nm), flush=True)
            cmd = "/bin/rm -f {}.pcm {}.f32".format(nm, nm)
            system(cmd)
            print(cmd)
        if self.hps["ext_data"] == "WAVERNN":
            os.chdir("WaveRNN/WaveRNN-master")
            nm_sd = nm.split("/", 1)[1]
            cmd = "{} ../../{}.npy".format(self.exe, nm)
            system(cmd)
            print(cmd)
            cmd = "mv model_outputs/ljspeech_mol.wavernn/__{}__797k_steps_gen_batched_target11000_overlap550.wav ../../{}.wav".format(
                nm_sd, nm
            )
            system(cmd)
            print(cmd)
            os.chdir("../../")

#---------------------------------SINTESI-----------------------------------------------------
@app.post("/synthesize")
def synthesize(allTextIn: str = Form(...), modello: str = Form("LC")):
    request = SynthesizeRequest(allTextIn=allTextIn)
    syntesisServer = synthesisServer()
    try:
        numGpu = request.numGpu
        noAutoNumbering = request.noAutoNumbering
        prediction = request.prediction
        playWav = request.playWav
        parameterFiles = request.parameterFiles
        tacotron = request.tacotron
        vocoder = request.vocoder
        if modello.startswith("SPK_"):
            speaker = modello.split("_WAVEGLOW")[0]
            request.config = "tc2_training.yaml"
        else:
            speaker = modello
        samplingRate = request.samplingRate
        hparams = request.hparams
        phoneticOnly = request.phoneticOnly
        silent = request.silent

        print(f"Valore di 'speaker': {modello}")

        hps = yaml.load(open(request.config, "r"), Loader=yaml.FullLoader)
        hps["save_embeddings"] = ""

        if hparams:
            hps.update(yaml.safe_load(hparams))
        init_symbols(hps)
        code_PAR = text_to_sequence("§")[0]  # symbol for text spliting
        code_POINT = text_to_sequence(".")[
            0
        ]  # symbol for end of utterance... to be replaced by end of paragraph at the end of each entry
        code_SILENT = 0

        if playWav:
            import sounddevice as sd

        device = torch.device("cuda:%d" % numGpu if torch.cuda.is_available() else "cpu")
        if device != "cpu":
            nvmlInit()
            torch.cuda.set_device(numGpu)
            syntesisServer.check_gpu("START", numGpu)

        if speaker in hps["speakers"]:
            spk_in = hps["speakers"].index(speaker)
            syntesisServer.spk_imposed = spk_in
            print("SPK [{}] = {}".format(hps["speakers"], spk_in))
        else:
            print(" SPK '{}' not in {}".format(speaker, hps["speakers"]))
            raise HTTPException(status_code=400, detail=f"Speaker '{speaker}' non presente in tc2_italian.yaml")

        if modello.startswith("SPK_"):
            model_path = os.path.join("_out", modello)

            if not os.path.isfile(model_path):
                raise HTTPException(status_code=404, detail=f"Checkpoint non trovato: '{model_path}'")

            tacotron = model_path
            request.tacotron = tacotron

        else:
            tacotron = "new_models/tacotron2_IT"
            request.tacotron = tacotron

        print(f"Percorso completo del modello: {tacotron}")

        if path.exists(vocoder):
            if vocoder.find("waveglow") >= 0:
                sigma = 0.6
                sys.path.append("waveglow")
                waveglow = torch.load(vocoder, map_location=device)["model"]
                waveglow = waveglow.remove_weightnorm(waveglow)
                if torch.cuda.is_available():
                    waveglow.cuda().eval()
            elif vocoder.find("hifigan") >= 0:
                sys.path.append("hifigan")
                from hifigan.env import AttrDict
                from hifigan.models import Generator
                with open("hifigan/config.json", "r") as f:
                    config = json.load(f)
                config = AttrDict(config)
                hgan = Generator(config)
                ckpt = torch.load(vocoder, map_location=torch.device("cpu"))
                hgan.load_state_dict(ckpt["generator"])
                hgan.eval()
                hgan.remove_weight_norm()
                param_size = buffer_size = 0
                for param in hgan.parameters():
                    param.requires_grad = False
                    param_size += param.nelement() * param.element_size()
                for buffer in hgan.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                size_all_mb = (param_size + buffer_size) / 1024 ** 2
                if torch.cuda.is_available():
                    hgan.to(device)
                if not silent:
                    print(
                        "HIFIGAN: MODEL {} loaded: {:.2f} Mo".format(vocoder, size_all_mb),
                        flush=True,
                    )
        else:
            print("VOCODER: MODEL {} not found. No synthesis".format(vocoder), flush=True)
            vocoder = None
        hps["mask_padding"] = False  # only one file at a time!
        model = Tacotron2(hps).to(device)
        if path.exists(tacotron):
            model_dict = torch.load(tacotron, map_location="cpu")["state_dict"]
            if not silent:
                print("List of the checkpoint" "s modules")
                for k, v in model_dict.items():
                    print(k, list(v.shape))
            for key in list(model_dict.keys()):
                if re.search(r"decoderVisual", key):
                    model_dict[key.replace("decoderVisual", "decoder.1")] = model_dict.pop(key)
                if re.search(r"decoder\.[^\d]", key):
                    model_dict[key.replace("decoder", "decoder.0", 1)] = model_dict.pop(key)
                if re.search(r"postnet\.[^\d]", key):
                    model_dict[key.replace("postnet", "postnet.0", 1)] = model_dict.pop(key)
            phonetize = model_dict.get(
                "phonetize.linear_layer.weight"
            )  # change of number of phonetic embeddings
            if phonetize != None:
                nb = len(hps["out_symbols"]) - phonetize.shape[0]
                if nb > 0:
                    phonetize = torch.cat((phonetize, torch.zeros(nb, phonetize.shape[1])))
                    model_dict.update({("phonetize.linear_layer.weight", phonetize)})
                    phonetize = model_dict.get("phonetize.linear_layer.bias")
                    phonetize = torch.cat((phonetize, -1000.0 * torch.ones(nb)))
                    model_dict.update({("phonetize.linear_layer.bias", phonetize)})
                    print("{} phonemes added".format(nb))
            model.load_state_dict(model_dict)
            print('Tacotron2 model "{}" loaded'.format(tacotron))
        else:
            print('Tacotron2 model "{}" not found'.format(tacotron))
            sys.exit()
        syntesisServer.check_gpu("AFTER LOADING MODELS", numGpu)

        data_test, nms_data = [], []

        if not prediction:
            suffix = "syn"
        else:
            suffix = "prd"
        model.eval()
        torch.set_grad_enabled(False)
        if phoneticOnly:
            hps["dim_data"] = []
            model.set_dim_data([])  # only phonetic prediction
        dim_data = hps["dim_data"]
        fe_data = hps["fe_data"]
        nb_out = len(dim_data)

        c_prec = code_PAR
        i_syn = 0
        nm_base = "INPUT"
        i_syn = i_syn + 1
        if re.search("^[\[\.§:?!§;,(§]", allTextIn) is None:
            allTextIn = (
                    "§" + allTextIn
            )  # if no initial punctuation: use of last from previous text
        if re.search("[\[\.§:?!§;,(§]$", allTextIn) is None:
            allTextIn = allTextIn + ".§"  # if no final punctuation: end of chapter
        if syntesisServer.spk_imposed >= 0:
            spk_in = syntesisServer.spk_imposed
        l_tags = re.findall(r"\<([^\>]*)\>\s*", allTextIn)
        # tags in text
        if l_tags:
            allTextIn = re.sub(r"\<([^\>]*)\>\s*", "", allTextIn)
            # remove tags
            for tags in l_tags:
                lt = re.findall("(\w+)=([^;]+)", tags)
                for t in lt:
                    if t[0] == "SPK" and t[1] in hps["speakers"]:
                        spk_in = hps["speakers"].index(t[1])
        allTextIn = np.array(text_to_sequence(allTextIn))
        lg_in = len(allTextIn)
        tensor_spk_in = to_gpu(torch.LongTensor([spk_in])[None, :])
        if hps["nb_styles"]:
            if syntesisServer.style_imposed >= 0:
                style_in = syntesisServer.style_imposed
            tensor_style_in = to_gpu(torch.LongTensor([style_in])[None, :])
        else:
            tensor_style_in = []
            style_in = 0

        wf_syn, out_par, spe_org = nb_out * [None], nb_out * [None], nb_out * [None]
        if not noAutoNumbering:
            nm_base += "_{:04d}".format(i_syn)
        for i_out in range(nb_out):
            nm_syn = "_syn_{}/{}_{}".format(hps["dir_data"][i_out], nm_base, suffix)
            if hps["ext_data"][i_out] == "WAVEGLOW" and vocoder is not None:
                wf_syn[i_out] = wave.open(nm_syn + ".wav", "wb")
                wf_syn[i_out].setparams((1, 2, samplingRate, 0, "NONE", "not compressed"))
                if not silent:
                    print("WAVEGLOW: {}.wav created".format(nm_syn), flush=True)
            out_par[i_out] = np.empty((0, hps["dim_data"][i_out]), dtype=np.float32)
        if not request.prediction:  # synthesis
            if '(~:?!§§§§.#;,[])"'.find(hps["symbols"][allTextIn[0]]) < 0:
                allTextIn = [c_prec] + allTextIn
                lg_in += 1  # prefix first utterance by a chapter onset
            parts = [i for i, val in enumerate(allTextIn) if val == code_PAR]
            nb_parts = len(parts)
            parts = [
                x for x in parts if (x + 1) not in parts
            ]  # keep first § if succession of §§
            if allTextIn[-1] == code_PAR:
                parts[-1] = lg_in
            else:
                parts = parts + [lg_in]
            if parts[0]:
                parts = [0] + parts
        else:
            parts = [0, lg_in]
        # un seul bloc
        c_prec = allTextIn[0]
        d_syn = np.zeros(nb_out, dtype=int)
        for ipart_txt in range(len(parts) - 1):  # splits of text entry
            allTextIn[parts[ipart_txt]] = c_prec
            c_prec = allTextIn[parts[ipart_txt + 1] - 1]
            # text split prefixed by last character (punctuation) of previous split
            text_in = allTextIn[parts[ipart_txt]: parts[ipart_txt + 1]]
            lg_in = len(text_in)
            tensor_text_in = torch.Tensor(text_in)[None, :]
            tensor_text_in = to_gpu(tensor_text_in).long()
            if prediction == False:
                (
                    part_spe_out,
                    part_spe_out_postnet,
                    part_gate_out,
                    pho_out,
                    dur_out,
                    style,
                    part_alignement,
                    part_embeddings,
                ) = model.inference(
                    (tensor_text_in, tensor_spk_in, tensor_style_in), hps["seed"]
                )
            else:
                tensor_out, lg_out = nb_out * [None], nb_out * [None]
                for i_out in range(nb_out):
                    if spe_org[i_out].any():
                        tensor_out[i_out] = torch.Tensor(spe_org[i_out].transpose())[
                                            None, :
                                            ].cuda()
                        lg_out[i_out] = spe_org[i_out].shape[0]
                (
                    part_spe_out,
                    part_spe_out_postnet,
                    part_gate_out,
                    pho_out,
                    dur_out,
                    style,
                    part_alignement,
                    part_embeddings,
                ) = model.forward(
                    (
                        tensor_text_in,
                        [lg_in],
                        tensor_spk_in,
                        tensor_style_in,
                        tensor_out,
                        lg_out,
                        [lg_in],
                    )
                )
                for i_out in range(nb_out):
                    if len(part_alignement[i_out]):
                        part_alignement[i_out] = part_alignement[i_out].transpose(0, 1)
            ch_in = [hps["symbols"][p] for p in text_in]
            if not silent:
                print(
                    "synthesis of chunk {:d} [{:d}-{:d},{:d}]: {}".format(
                        ipart_txt, spk_in, style_in, lg_in, "|".join(ch_in)
                    ),
                    flush=True,
                )
            if len(pho_out) > 0:
                pb = torch.sigmoid(pho_out[0, :, :].cpu()).data.numpy()
                # 			pb=pb/sum(pb)
                # 			ind1=pb.max(axis=0); ind=ind1.indices
                # 			ind2=pb[1:,:].max(axis=0); #second best candidate
                # 			iok=np.where(ind2.values>.45)
                # 			ind[iok]=ind2.indices[iok]+1
                ind = pho_out[0, :, :].argmax(axis=0).cpu().data.numpy()
                ph_prd = "|".join(
                    ["{}".format(hps["out_symbols"][p]) for p in ind]
                )  # predicted aligned phonetic chain
                print("PH_PRD: {}".format(ph_prd))
            # production d'alignement
            if 0:
                ind_aln = np.append(ind, 1)
                ch_in_aln = ch_in + ["x"]
                part_aln = part_alignement[0].cpu().data.numpy()
                to_s = hps["n_frames_per_step"][0] / hps["fe_data"][0]
                d_seq = data_test[i_syn][1]
                id = 0
                for i_in in range(lg_in):
                    if ind[i_in] != code_SILENT:
                        res = (
                            next(
                                x
                                for x, val in enumerate(part_aln[0, i_in, id:])
                                if val > 0.4
                            )
                            if (part_aln[0, i_in, id:].max() > 0.4)
                            else 1
                        )
                        id = id + res
                        # nb de caract§res
                        dd = 0 if hps["out_symbols"][ind[i_in]] == "__" else 1
                        aa = np.where(
                            np.logical_or(
                                ind_aln[i_in + dd:] != code_SILENT,
                                np.array(
                                    [
                                        v in " !?~§'§§§[]{}(),."
                                        for v in ch_in_aln[i_in + dd:]
                                    ]
                                ),
                            )
                        )
                        if len(aa[0]):
                            nb_car = aa[0][0] + dd
                        else:
                            nb_car = 0 if hps["out_symbols"][ind[i_in]] == "__" else 1
                        print(
                            "{:3f} {}|{}".format(
                                d_seq + id * to_s, hps["out_symbols"][ind[i_in]], nb_car
                            ),
                            end="",
                        )
                        if hps["out_symbols"][ind[i_in]] == "__":
                            txt = ""
                            i = i_in
                            while i < lg_in and ch_in[i] in "!?~§'§§§[]{}(),.":
                                txt = txt + ch_in[i]
                                i = i + 1
                            print("|{}".format(txt), end="")
                        if len(txt) == 0:
                            i = i_in
                            while i < lg_in and not ch_in[i] in " !?~§'§§§[]{}(),.":
                                txt = txt + ch_in[i]
                                i = i + 1
                            print("|{}".format(txt), end="")
                        print("")
                    if ch_in[i_in] in " !?~§'§§§[]{}().":
                        txt = ""

            if len(dur_out[0, :]):
                ch_dprd = " ".join(
                    [
                        "{}".format(int(d))
                        for d in 100 * dur_out[0, :].cpu().detach().numpy()[0, :]
                    ]
                )
                if not silent:
                    print("prd_dur: {}".format(ch_dprd))
            for i_out in range(nb_out):
                lg_part_out = part_spe_out[i_out].shape[2]
                if i_out > 0:
                    lg_part_ref = int(
                        part_spe_out_postnet[0].shape[2]
                        * hps["fe_data"][i_out]
                        / hps["fe_data"][0]
                    )
                    if lg_part_out > lg_part_ref:
                        part_spe_out_postnet[i_out] = part_spe_out_postnet[i_out][
                                                      :, :, 0: lg_part_ref - 1
                                                      ]
                        part_spe_out[i_out] = part_spe_out[i_out][:, :, 0: lg_part_ref - 1]
                        lg_part_out = lg_part_ref
                d_syn[i_out] += lg_part_out
                if not silent:
                    print(
                        "{}: {}, {}->{:.2f}\n".format(
                            i_out,
                            d_syn[i_out],
                            lg_part_out,
                            d_syn[i_out] / hps["fe_data"][i_out],
                        )
                    )
                if len(part_alignement[i_out]):
                    part_aln = part_alignement[i_out].cpu().data.numpy()[0]
                    part_gate = torch.sigmoid(part_gate_out[i_out].cpu()).data.numpy()[0, :]
                    ms_from_act = (
                            1000.0
                            * hps["n_frames_per_step"][i_out]
                            * part_aln.sum(axis=1)
                            / hps["fe_data"][i_out]
                    )
                    print(
                        "".join(
                            [
                                "|{:.0f}{}".format(ms_from_act[x], ch_in[x])
                                for x in range(0, len(ch_in))
                            ]
                        )
                    )

                part_out = part_spe_out_postnet[i_out].cpu().data.numpy()
                part_out = part_out[0, :, :].transpose()

                if not silent:
                    print(
                        "prd_{}: {:.2f}s".format(
                            hps["ext_data"][i_out], lg_part_out / hps["fe_data"][i_out]
                        )
                    )

                if ipart_txt:  # insert silence 300ms
                    nt = int(0.3 * hps["fe_data"][i_out])
                    d_syn[i_out] += nt
                    fe_wav = hps["fe_data"][i_out]
                    if wf_syn[i_out]:
                        ne = int(nt * fe_wav / hps["fe_data"][i_out])
                        wf_syn[i_out].writeframes(np.zeros(ne, dtype="int16"))
                    out_par[i_out] = np.concatenate(
                        (out_par[i_out], np.tile(part_out[0, :], (nt, 1)))
                    )
                if wf_syn[i_out]:
                    with torch.no_grad():
                        if vocoder.find("waveglow") >= 0:
                            audio = (
                                    MAX_WAV_VALUE
                                    * waveglow.infer(part_spe_out_postnet[i_out], sigma=sigma)[
                                        0
                                    ]
                            )
                        elif vocoder.find("hifigan") >= 0:
                            audio = MAX_WAV_VALUE * hgan(part_spe_out_postnet[i_out][0])[0]
                    audio = audio.cpu().numpy().astype("int16")
                    wf_syn[i_out].writeframes(audio)
                    if playWav:
                        sd.play(audio, fe_wav)

                out_par[i_out] = np.concatenate((out_par[i_out], part_out))

                # ----------- Display Attention alignments of each chunk ----------------
                if request.draw:
                    aln = (1.0 + np.arange(lg_in)).dot(part_aln) - 1.0

                    if i_out == 0:
                        plt.clf()
                    aa = plt.subplot(2, nb_out, i_out + 1)
                    plt.matshow(
                        part_aln, origin="lower", aspect="auto", fignum=0, vmin=0, vmax=1
                    )
                    axes = plt.gca()
                    axes.get_xaxis().set_visible(False)
                    plt.ylabel("Time")
                    axes.yaxis.set_ticks_position("left")
                    axes.yaxis.set_major_locator(MultipleLocator(1))
                    axes.yaxis.set_major_formatter(FormatStrFormatter("%s"))
                    axes.set_yticks(1 + np.arange(lg_in))
                    axes.yaxis.set_ticklabels(ch_in, fontsize=7, rotation=90)
                    plt.ylabel("Encoder states")
                    nm_syn = "_syn_{}/{}_{}".format(hps["dir_data"][i_out], nm_base, suffix)
                    plt.title('{}: "{}"'.format(nm_syn, "".join(ch_in)), fontsize=9, pad=10)
                    plt.plot(aln, "w-", linewidth=3)
                    plt.draw()
                    plt.subplot(2, nb_out, nb_out + i_out + 1)
                    if wf_syn[i_out]:
                        plt.plot(
                            hps["fe_data"][i_out]
                            * np.arange(len(audio))
                            / fe_wav
                            / hps["n_frames_per_step"][i_out],
                            audio / 10000,
                            "k",
                            linewidth=0.5,
                        )
                        if playWav:
                            sd.play(audio, fe_wav)
                    else:
                        plt.plot(part_out.mean(axis=1))
                        if spe_org[i_out] is not None:
                            plt.plot(spe_org[i_out].mean(axis=1), linestyle="-.")
                    plt.plot(part_gate, "r-", linewidth=0.5)
                    plt.ylim(-2.1, 2.1)
                    plt.xlim(0, len(part_gate))
                    plt.draw()
            if request.draw:
                matfig.show()
                plt.waitforbuttonpress()
            # resynchronize
            if len(out_par):
                lg = [
                    out_par[i_out].shape[0] / hps["fe_data"][i_out]
                    for i_out in range(nb_out)
                ]
                lg_max = max(lg)
                for i_out in range(nb_out):
                    nt = round((lg_max - lg[i_out]) * hps["fe_data"][i_out])
                    if nt:
                        d_syn[i_out] += nt
                        out_par[i_out] = np.concatenate(
                            (out_par[i_out], np.tile(out_par[i_out][-1, :], (nt, 1)))
                        )
                        if wf_syn[i_out]:
                            ne = int(nt * fe_wav / hps["fe_data"][i_out])
                            wf_syn[i_out].writeframes(np.zeros(ne, dtype="int16"))

        for i_out in range(nb_out):
            if not silent:
                print(
                    "dur_syn[{}]={:.3f}s".format(
                        i_out, d_syn[i_out] / hps["fe_data"][i_out]
                    )
                )
            if wf_syn[i_out]:
                wf_syn[i_out].close()
            if parameterFiles:
                nm_syn = "_syn_{}/{}_{}".format(hps["dir_data"][i_out], nm_base, suffix)
                fp = open(nm_syn + "." + hps["ext_data"][i_out], "wb")
                if type(hps["fe_data"][i_out]) is int:
                    num = hps["fe_data"][i_out]
                    den = 1
                else:
                    (num, den) = (hps["fe_data"][i_out]).as_integer_ratio()
                fp.write(np.asarray(out_par[i_out].shape + (num, den), dtype=np.int32))
                fp.write(out_par[i_out].copy(order="C"))
                fp.close()
                print(
                    "{}.{} created [{}, {:d}ms]".format(
                        nm_syn,
                        hps["ext_data"][i_out],
                        out_par[i_out].shape,
                        int(1000.0 * out_par[i_out].shape[0] / hps["fe_data"][i_out]),
                    ),
                    flush=True,
                )
        return FileResponse(
            os.path.abspath(f"{nm_syn}.wav"),
            media_type="audio/wav",
            filename=f"{nm_syn}.wav"
        )
    except Exception as e:
        logger.error("Errore durante il processo", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore durante il processo: {str(e)}")

#---------------------------------Training-----------------------------------------------------
@app.post("/upload/zip")
def upload_zip(file: UploadFile = File(...)):
    logging.debug(f"File ricevuto: {file.filename}")
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Il file caricato non è un file ZIP!")
    zipPath = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        logging.debug(f"Salvataggio del file zip in {zipPath}")
        with open(zipPath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with zipfile.ZipFile(zipPath, 'r') as z:
            z.extractall(UPLOAD_FOLDER)

        logging.debug(f"Contenuti della cartella di upload: {os.listdir(UPLOAD_FOLDER)}")

        curr_folder_path = next(
            (os.path.join(UPLOAD_FOLDER, d) for d in os.listdir(UPLOAD_FOLDER)
             if os.path.isdir(os.path.join(UPLOAD_FOLDER, d)) and not d.startswith('__MACOSX')),
            None
        )

        if curr_folder_path is None:
            raise HTTPException(status_code=400, detail="Cartella estratta non trovata!")

        # cerca e verifica csv
        csv_files = []
        for root, dirs, files in os.walk(curr_folder_path):
            for file_name in files:
                if file_name.endswith('.csv'):
                    csv_files.append(os.path.join(root, file_name))
        if not csv_files:
            shutil.rmtree(curr_folder_path)
            raise HTTPException(status_code=400, detail="File CSV mancante nel file ZIP estratto!")

        csv_path = csv_files[0].replace("\\", "/")
        logging.debug(f"File CSV trovato: {csv_path}")

        from generaSpettrogramma import process_wav_folder

        process_wav_folder(curr_folder_path)

        from VoiceCloning_Train import voiceCloning, VoiceCloningRequest

        yaml_path = "tc2_training.yaml"

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        new_speaker_id = f"SPK_{int(time.time())}"

        if new_speaker_id not in config["speakers"]:
            config["speakers"].append(new_speaker_id)
            config["nb_speakers"] = len(config["speakers"])

        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        dynamic_params = {
            "nm_csv_train": csv_path,
            "speakers": config["speakers"],
            "nb_speakers": config["nb_speakers"]
        }
        hparams_str = yaml.dump(dynamic_params)

        voiceCloning.request = VoiceCloningRequest(
            outputDir="_out",
            config="new_models/tc2_italian.yaml",
            preTrained="new_models/tacotron2_IT",
            hparams=hparams_str,
            modelName=new_speaker_id
        )

        result = voiceCloning.start_train()

        ext_data = config["ext_data"]
        nb_epochs = config["nb_epochs"]

        model_dir_name = f"{result['model_name']}_{'_'.join(ext_data)}_{nb_epochs - 1}"

        return {
            "message": "File zip estratto e addestramento completato",
            "filename": file.filename,
            "extracted_path": UPLOAD_FOLDER,
            "csv_path": csv_path,
            "newModelName": model_dir_name
        }

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante il processo: {str(e)}")
    finally:
        if os.path.exists(zipPath):
            os.remove(zipPath)


#---------------------------------INDEX-----------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <link rel="icon" type="image/png" href="https://cdn-icons-png.flaticon.com/512/1693/1693783.png">
    <title>Grenoble</title>
    <style>
        body{
            font-family:Verdana, Geneva, Tahoma, sans-serif;
            background: url('https://it.vidnoz.com/img/sound-test/sound_test_bg.svg');
        }

        #upload-section .text-center.text-muted.mb-4 b{
            color:#555;
        }

        /* Sidebar */
        .sidebar {
            position: fixed;
            top: 0;
            left: -250px;
            width: 250px;
            height: 100%;
            transition: left 0.3s ease;
            padding-top: 80px;
            border-right: 1px solid #c4cad1;
            z-index: 1;
        }

        .sidebar.open {
            left: 0;
        }

        .sidebar .btn-section {
            width: 100%;
            text-align: left;
            padding: 15px;
            border-radius: 0;
            margin-bottom: 10px;
        }

        .sidebar .btn-section.active {
            background-color: #007bff;
            color: white;
        }

        .hamburger {
            position: fixed;
            top: 10px;
            left: 20px;
            font-size: 30px;
            cursor: pointer;
            z-index: 1000;
            background-color: transparent;
            border: none;
            color: #333;
            padding: 10px;
        }

        .hamburger:focus{
            outline: none;
        }

        .hamburger:hover{
            color: #4a6bff;
        }

        /* Contenuto principale */
        .content {
            margin-left: 0;
            padding: 20px;
            transition: margin-left 0.3s ease;
        }

        .content.shifted {
            margin-left: 250px;
        }

        h1.text-center{
            margin-top: 0;
            color: black;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
            font-style: italic;
        }



        .form-group {
            max-width: 1000px;
            margin: 0 auto;
        }

        .btn-primary{
            margin-top: 20px;
        }

        #allTextIn{
            width: 100%;
            max-width: 1000px;
            margin: 0 auto;
            resize: none;
            overflow: hidden;
            display: block;
            margin-block-end: 2%;
        }

        #allTextIn:hover, #modelSelect:hover,#customSelect:hover{
            border-color: #4a6bff;
            background-color: #f0f5ff;
        }

        #allTextIn,#modelSelect,#customSelect{
            background-color:whitesmoke;
        }

        .mc{
            width: 48%;
            margin-bottom: 10px;
        }

        .left{
            float:left;
        }

        .right{
            float:right;
            text-align: right;
        }

        .form-group::after{
            content:"";
            display: table;
            clear:both;
        }

        #modelSelect{
            max-width: 42%;
        }

        #customSelect{
            max-width: 42%;
            margin-left: auto;
        }

        #loading-synthesis, #error-synthesis, #success-synthesis,
        #loading-upload, #error-upload, #success-upload { 
            display: none; 
        }

        .spin {
            display: inline-block;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Sezione di upload */
        .upload-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px;
            border: 2px dashed #ccc;
            border-radius: 10px;
            background-color: whitesmoke;
            cursor: pointer;
            transition: all 0.3s;
            margin: 20px auto;
            max-width: 600px;
            text-align: center;
        }

        .upload-container:hover {
            border-color: #4a6bff;
            background-color: #f0f5ff;
        }

        .upload-container.dragover {
            border-color: #4a6bff;
            background-color: #e6f0ff;
        }

        .upload-icon {
            font-size: 48px;
            color: #4a6bff;
            margin-bottom: 15px;
        }

        .upload-text {
            font-size: 18px;
            margin-bottom: 10px;
            color: #555;
        }

        .upload-hint {
            font-size: 14px;
            color: #888;
            margin-bottom: 20px;
        }

        .file-info {
            margin-top: 15px;
            font-size: 14px;
            color: #555;
            display: none;
        }

        .btn-upload {
            background-color: #4a6bff;
            color: white;
            padding: 10px 25px;
            border-radius: 5px;
            border: none;
            font-weight: 500;
            cursor: pointer;
            pointer-events: auto;
        }

        #fileInput {
            display: block;
            opacity: 0;
            pointer-events: none;
        }

        .btn-disabled {
            cursor: not-allowed;
            pointer-events: none;
            opacity: 0.5;
        }

        /*responsive*/
        @media (max-width: 480px) {
            .sidebar {
                width: 100%;
                left: -100%;
            }

            .sidebar.open {
                left: 0;
                background-color: rgba(0, 0, 0, 0.6);
            }

            body .content {
                margin-left: 0;
            }

            body .content.shifted {
                margin-left: 0;
            }

            .hamburger {
                top: 15px;
                left: 10px;
                font-size: 28px;
                z-index: 1100;
            }

            h1.text-center {
                font-size: 2em;
                margin-top: 60px;
                margin-bottom: 20px;
            }

            .mc, #modelSelect, #customSelect {
                width: 100%;
                max-width: 100%;
                float: none;
                margin-left: 0;
            }

            .left, .right {
                float: none;
                text-align: left;
            }

            .form-group::after {
                display: none;
            }

            .upload-container {
                padding: 20px;
                margin: 10px;
                max-width: 90%;
            }
        }



    </style>
    <script>
        "use strict";

        const userLang = navigator.language || navigator.userLanguage;
        
        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const content = document.getElementById('content');
            sidebar.classList.toggle('open');
            content.classList.toggle('shifted');
        }
    
        function showSection(section) {
            document.getElementById("synthesis-section").style.display = section === 'synthesis' ? 'block' : 'none';
            document.getElementById("upload-section").style.display = section === 'upload' ? 'block' : 'none';

            const heading = document.querySelector("h1.text-center");
            if(section === 'upload'){
                userLang.startsWith('it') ? heading.textContent = 'Clona la tua Voce' : heading.textContent = 'Clone Your Voice';
                document.getElementById("error-synthesis").style.display = 'none'; 
                document.getElementById("success-synthesis").style.display = 'none';
                document.getElementById("loading-synthesis").style.display = 'none';
            }
            else if(section === 'synthesis'){
                userLang.startsWith('it') ? heading.textContent = 'Sintetizzatore di Testo' : heading.textContent = 'Text Synthesizer';
                document.getElementById("error-upload").style.display = 'none'; 
                document.getElementById("success-upload").style.display = 'none'; 
                document.getElementById("loading-upload").style.display = 'none';
            }
            
            document.getElementById("btn-synthesis").classList.toggle("active", section === 'synthesis');
            document.getElementById("btn-upload").classList.toggle("active", section === 'upload');
        }
    
        function showLoading(type) {
            document.getElementById(`loading-${type}`).style.display = "block";
            document.getElementById(`error-${type}`).style.display = "none";
            document.getElementById(`success-${type}`).style.display = "none";
        }
    
        function showError(type, msg) {
            document.getElementById(`loading-${type}`).style.display = "none";
            document.getElementById(`error-${type}`).innerText = msg;
            document.getElementById(`error-${type}`).style.display = "block";
            document.getElementById(`success-${type}`).style.display = "none";
        }
    
        function showSuccess(type, msg) {
            document.getElementById(`loading-${type}`).style.display = "none";
            document.getElementById(`success-${type}`).innerHTML = msg;
            document.getElementById(`success-${type}`).style.display = "block";
            document.getElementById(`error-${type}`).style.display = "none";
        }
    
        async function uploadZipFile(event) {
            event.preventDefault();
            const fileInput = document.getElementById('fileInput');
            const submitButton = event.target.querySelector("button[type='submit']");
            submitButton.classList.add("btn-disabled");
            showLoading('upload');
    
            const formData = new FormData();
            formData.append('file',fileInput.files[0]);
            
            try {
                const response = await fetch('/upload/zip', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const data = await response.json();
                    userLang.startsWith('it') ? showError('upload', "❌ Errore durante il caricamento: " + (data.detail || "Errore sconosciuto")) : showError('upload', "❌ Error while running: " + (data.detail || "Unknown Error"));
                } else {
                    const data = await response.json();
                    userLang.startsWith('it') ? showSuccess('upload', '<i class="bi bi-check me-2" style="color: green; font-size: 1.5rem; vertical-align: middle;"></i>Caricamento completato con successo!')
                     : showSuccess('upload', '<i class="bi bi-check me-2" style="color: green; font-size: 1.5rem; vertical-align: middle;"></i>Upload completed successfully!');
                    if (data.newModelName) {
                        addModelOption(data.newModelName);
                    }
                    document.getElementById('fileInfo').style.display = 'none';
                    fileInput.value = '';
                }
            } catch (error) {
                userLang.startsWith('it') ? showError('upload', "❌ Errore durante il caricamento del file.") : showError('upload', "❌ Error loading file.");
            } finally {
                submitButton.classList.remove("btn-disabled");
            }
        }
    
        function handleFileSelect(event) {
            const files = event.target.files || (event.dataTransfer && event.dataTransfer.files);
            
            if (files && files.length > 0) {
                const fileInfo = document.getElementById('fileInfo');
                fileInfo.innerHTML = `
                    <i class="fas fa-file-archive"></i> ${files[0].name} 
                `;
                fileInfo.style.display = 'block';
                
                const uploadContainer = document.querySelector('.upload-container');
                uploadContainer.classList.remove('dragover');
                uploadContainer.style.borderColor = '#4a6bff';
                uploadContainer.style.backgroundColor = '#e6f0ff';
            }
        }
    
        function setupDragAndDrop() {
            const uploadContainer = document.querySelector('.upload-container');
            const fileInput = document.getElementById('fileInput');
            const btnUpload = document.querySelector('.btn-upload');
            
            //previene apertura nuova scheda su tutto il documento
            document.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
            
            document.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
    
            btnUpload.addEventListener('click', function(e) {
                e.stopPropagation();
                fileInput.click();
            });
            
            uploadContainer.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadContainer.classList.add('dragover');
            });
            
            ['dragleave', 'dragend'].forEach(type => {
                uploadContainer.addEventListener(type, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    uploadContainer.classList.remove('dragover');
                });
            });
            
            uploadContainer.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                uploadContainer.classList.remove('dragover');
                
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    handleFileSelect(e);
                }
            });
            
            fileInput.addEventListener('change', handleFileSelect);
        }
    
        async function handleSynthesis(event){
            event.preventDefault();
            const submitButton = document.querySelector("button[type='submit']");
            submitButton.classList.add("btn-disabled");
            showLoading('synthesis');

            const modelloInput = document.createElement("input");
            modelloInput.type = "hidden";
            modelloInput.name = "modello";
            const selectedModel = document.getElementById('modelSelect').value || document.getElementById('customSelect').value;
            modelloInput.value = selectedModel;
            event.target.appendChild(modelloInput);
            localStorage.setItem("selectedModel", selectedModel);
    
            const formData = new FormData(event.target);
            const model = formData.get('modello') || 'default';
            
            try {
                const response = await fetch('/synthesize', {
                    method: 'POST',
                    body: formData
                });
    
                if (!response.ok) {
                    throw new Error(userLang.startsWith('it') ? "Errore durante l'esecuzione. Riprova." : "Error while running. Please try again.");
                }
    
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `sintesi_${model}.wav`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
    
                showSuccess('synthesis', userLang.startsWith('it') ? '<i class="bi bi-check me-2" style="color: green; font-size: 1.5rem; vertical-align: middle;"></i>File WAV generato con successo!' 
                : '<i class="bi bi-check me-2" style="color: green; font-size: 1.5rem; vertical-align: middle;"></i>WAV file generated successfully!');
            } catch (error) {
                showError('synthesis', "❌ " + error.message);
            } finally {
                submitButton.classList.remove("btn-disabled");
            }
        }

        function updateSelectPlaceholder(){
            const customSelect = document.getElementById('customSelect');
            const modelSelect = document.getElementById('modelSelect');

            if(customSelect.options.length === 0){
                userLang.startsWith('it') ? customSelect.innerHTML = '<option value="" disabled selected>Nessun modello</option>' : customSelect.innerHTML = '<option value="" disabled selected>No Model</option>';
            }
            else if(modelSelect.value){
                customSelect.value="";
            }
        }

        function handleModelSelection(event){
            const modelSelect = document.getElementById('modelSelect');
            const customSelect = document.getElementById('customSelect');

            if(event.target.value){
                if(event.target === modelSelect){
                    customSelect.value="";
                }else if(event.target === customSelect){
                    modelSelect.value="";
                    localStorage.setItem("selectedModel", event.target.value);
                }
            }
        }

        function addModelOption(modelName) {
            const select = document.getElementById('customSelect');
            if (!Array.from(select.options).some(option => option.value === modelName)) {
                const newOption = document.createElement('option');
                newOption.value = modelName;
                newOption.textContent = modelName;
                select.appendChild(newOption);

                let allModels = JSON.parse(localStorage.getItem("allCustomModels") || "[]");
                if (!allModels.includes(modelName)) {
                    allModels.push(modelName);
                    localStorage.setItem("allCustomModels", JSON.stringify(allModels));
                }
            }
            updateSelectPlaceholder();
        }

        function adaptLanguage(){
            console.log("Lingua del browser rilevata:", userLang);

            if(!userLang.startsWith('it')){
                document.documentElement.lang='en';
                document.querySelector("label[for='allTextIn'] strong").textContent = 'Enter your text:';
                document.querySelector("textarea#allTextIn").placeholder = "Write your text here";
                document.querySelector("label[for='modelSelect'] strong").textContent = 'Default Model:';
                document.querySelector("label[for='customSelect'] strong").textContent = 'Custom Model:';
                document.getElementById("bottone-sintesi").innerHTML='<i class="fas fa-cogs mr-2"></i> Synthesize!';
                document.querySelector(".text-center.text-muted.mb-4 strong").innerHTML = `<b>Upload a ZIP file including:</b> your audio file (.wav) and the corresponding CSV file (.csv)`;
                document.querySelector(".upload-text").textContent = "Drag your file here";
                document.querySelector(".upload-hint").textContent = "or";
                document.querySelector(".btn-upload").textContent = "Choose File";
                document.getElementById("bottone-upload").innerHTML = `<i class="fas fa-upload mr-2"></i> Upload!`;
                document.getElementById("loading-upload").innerHTML = `<i class="fa fa-spinner fa-spin spin me-2"></i> Processing...`;
                document.getElementById("loading-synthesis").innerHTML = `<i class="fa fa-spinner fa-spin spin me-2"></i> Processing...`;
                document.getElementById("noModel").innerHTML = '<option value="" disabled id="noModel">No Model</option>';
            }
        }
    
        //Inizializzazione
        window.onload = function() {
            adaptLanguage();
            showSection('synthesis');
            setupDragAndDrop();
            updateSelectPlaceholder();

            const storedModels = JSON.parse(localStorage.getItem("allCustomModels") || "[]");
            storedModels.forEach(model => addModelOption(model));
            const savedModel = localStorage.getItem("selectedModel");

            document.getElementById('modelSelect').addEventListener('change', handleModelSelection);
            document.getElementById('customSelect').addEventListener('change', handleModelSelection);
        };

    </script>
</head>
<body>
    <!-- Hamburger Menu -->
    <button class="hamburger" onclick="toggleSidebar()">&#9776;</button>

    <!-- Sidebar -->
    <div id="sidebar" class="sidebar">
        <button id="btn-synthesis" class="btn btn-primary btn-section active" onclick="showSection('synthesis')">Text to Speech</button>
        <button id="btn-upload" class="btn btn-secondary btn-section" onclick="showSection('upload')">Voice Cloning</button>
    </div>

    <!-- Contenuto principale -->
    <div id="content" class="content">
        <h1 class="text-center">Sintetizzatore di Testo</h1>
        
        <!-- Sezione Sintesi Vocale -->
        <div id="synthesis-section">
            <form method="POST" action="/synthesize" class="mt-4" onsubmit="handleSynthesis(event)">
                <div class="form-group">
                    <label for="allTextIn"><strong>Inserisci il tuo testo:</strong></label>
                    <textarea id="allTextIn" name="allTextIn" class="form-control" rows="4" placeholder="Scrivi qui il tuo testo" required oninput="this.style.height = ''; this.style.height = this.scrollHeight + 'px';"></textarea>
                    <!-- Dropdown scelta modello-->
                  <div class="mc left">
                    <label for="modelSelect"><strong>Modello di Default:</strong></label>
                    <select id="modelSelect" name="modello" class="form-control">
                        <option value="" disabled id="noModel">Nessun Modello</option>
                        <option value="MT">Mariateresa</option>
                    </select>
                  </div>
                    <div class="mc right">
                    <label for="customSelect"><strong>Modello Personalizzato:</strong></label>
                    <select id ="customSelect" class="form-control"></select>
                    </div>
                </div>

                <div class="d-flex justify-content-center">
                    <button type="submit" id="bottone-sintesi" class="btn btn-primary btn-lg px-4"><i class="fas fa-cogs mr-2"></i> Sintetizza!</button>
                </div>
            </form>
            <div id="loading-synthesis" class="text-center mt-3 text-info">
                <i class="fa fa-spinner fa-spin spin me-2"></i> Elaborazione in corso...
            </div>
            <div id="error-synthesis" class="text-center mt-3 text-danger"></div>
            <div id="success-synthesis" class="text-center mt-3 text-success"></div>
        </div>
        
        <!--Sezione Upload Zip-->
        <div id="upload-section">
            <p class="text-center text-muted mb-4"><strong><b>Carica un file ZIP includendo:</b> il tuo file audio (.wav) e il corrispondente file CSV (.csv)</strong></p>

            <form method="POST" action="/upload/zip" enctype="multipart/form-data" class="mt-3" onsubmit="uploadZipFile(event)">
                <div class="upload-container">
                    <div class="upload-icon">
                        <i class="fas fa-cloud-upload-alt"></i>
                    </div>
                    <div class="upload-text">Trascina il file qui</div>
                    <div class="upload-hint">oppure</div>
                    <button type="button" class="btn-upload">Scegli file</button>
                    <div id="fileInfo" class="file-info"></div>
                    <input type="file" id="fileInput" name="file" accept=".zip" required>
                </div>

                <div class="d-flex justify-content-center mt-4">
                    <button type="submit" id="bottone-upload" class="btn btn-success btn-lg px-4">
                        <i class="fas fa-upload mr-2"></i> Carica!
                    </button>
                </div>
            </form>
        </div>

            <div id="loading-upload" class="text-center mt-3 text-info"><i class="fa fa-spinner fa-spin spin me-2"></i> Elaborazione in corso...</div>
            <div id="error-upload" class="text-center mt-3 text-danger"></div>
            <div id="success-upload" class="text-center mt-3 text-success"></div>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
