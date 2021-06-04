import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from utils import compute_sdr, MAX_INT16, center_trim
from tqdm import tqdm
from pprint import pprint

from asteroid.models.fasnet import FasNetTAC
from asteroid.metrics import get_metrics
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from local.tac_dataset import TACDataset
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from deepbeam import OnlineSimulationDataset,vctk_audio, truncator, ms_snsd, simulation_config_test, test_set

parser = argparse.ArgumentParser()

parser.add_argument(
    "--use_gpu", type=int, default=1, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")
parser.add_argument(
    "--n_save_ex", type=int, default=50, help="Number of audio examples to save, -1 means all"
)

compute_metrics = ["si_sdr"]  # , "sdr", "sir", "sar", "stoi"]


def main(conf):
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = FasNetTAC.from_pretrained(model_path)
    #model = FasNetTAC(**train_conf["net"], sample_rate=train_conf['data']['sample_rate'])
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    #test_set = TACDataset(args.test_json, train=False)


    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    torch.no_grad().__enter__()
    input_sdr_list = []
    output_sdr_list = []
    for idx in tqdm(range(len(test_set))):

        # Forward the network on the mixture.
        input = test_set.__getitem__(idx)
        mix = np.expand_dims(input[0], axis=0)/MAX_INT16  # 1 * channel * length
        ref = input[3]/MAX_INT16
        raw = torch.tensor(mix, dtype=torch.float32, device=model_device)
        ref = torch.tensor(ref, dtype=torch.float32, device=model_device)


        valid_mics = torch.ones((len(mix), 1)).to(dtype=torch.long, device=raw.device) * 6
        spks = model(raw, valid_mics)
        ref = center_trim(ref, spks).transpose(1, 0)
        loss, spks = loss_func(spks, ref, return_est=True)
        spks = spks.data.cpu().numpy().squeeze()
        ref = ref.data.cpu().numpy() * MAX_INT16

        for idx, samps in enumerate(spks):
            samps = samps * MAX_INT16
            input_sdr_list.append(compute_sdr(ref[0, idx], mix[0, 0, :] * MAX_INT16))
            output_sdr_list.append(compute_sdr(ref[0, idx], samps))
    input_sdr_array = np.array(input_sdr_list)
    output_sdr_array = np.array(output_sdr_list)
    result = np.median(output_sdr_array - input_sdr_array)
    print("The SNR: " + str(result))


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic, train_conf=train_conf)