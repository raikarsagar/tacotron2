
import matplotlib
#%matplotlib inline
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import time

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
#from denoiser import Denoiser
from scipy.io.wavfile import read, write
from plotting_utils import *

import math
import cProfile, pstats, io

import torch.autograd.profiler as profiler


# Import Extension
import intel_pytorch_extension as ipex

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("Device : ", device)

taco2_checkpoint_path = "/home/ubuntu/tacotron2/tacotron2_model/tacotron2_statedict.pt"
waveglow_checkpoint_path = "/home/ubuntu/tacotron2/waveglow_pretr/waveglow_256channels_universal_v5_converted.pt"
		

def plot_data(data, figsize=(16, 4)):
	fig, axes = plt.subplots(1, len(data), figsize=figsize)
	for i in range(len(data)):
		axes[i].imshow(data[i], aspect='auto', origin='bottom', interpolation='none')


with profiler.profile(record_shapes=True,use_cuda=use_cuda) as prof:
	with profiler.record_function("model_inference"):
		hparams = create_hparams()
		hparams.sampling_rate = 22050

		model = load_model(hparams)
		model.load_state_dict(torch.load(taco2_checkpoint_path, map_location = device)['state_dict'])
		if use_cuda:
			_ = model.cuda().eval().half()
		else:
			# _ = model.to(device).eval()
			_ = model.to(ipex.DEVICE)

		waveglow = torch.load(waveglow_checkpoint_path,map_location=device)['model']

		if use_cuda:
			waveglow.cuda().eval().half()
		else:
			# waveglow.to(device).eval()
			waveglow.to(ipex.DEVICE)

		for k in waveglow.convinv:
			k.float()
		#denoiser = Denoiser(waveglow)

		text = "This is a test of speech synthesis system"
		sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
		# 
		if use_cuda:
			sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
		else:
			# sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
			sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(ipex.DEVICE).long()

		start_taco2 = time.time()
		mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
		taco2_time = time.time()-start_taco2

		# mel_outputs_postnet = mel_outputs_postnet.to(device)
		mel_outputs_postnet = mel_outputs_postnet.to(ipex.DEVICE)

		start_waveglow = time.time()
		with torch.no_grad():
			audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
		waveglow_time = time.time()-start_waveglow
		

print("-----------Report for inference time profile-------")
# write('synthesized_out.wav',22050,audio.float().cpu().detach().numpy().astype('int16').T)
write('synthesized_en_ipex.wav',22050,audio.float().cpu().detach().numpy().T)
# print(audio.float().cpu().detach()[0])
# pr.disable()

# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# # pr.dump_stats('time_profile.txt')
# # print(s.getvalue())

# with open('time_profile_gpu.txt', 'w+') as f:
# 	f.write(s.getvalue())
# pr.print_stats()

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# prof.export_chrome_trace("trace_gpu.json")

print("Tacotron2 Inference time:", taco2_time)
print("Waveglow Inference time:", waveglow_time)
print("Total Execution time :", taco2_time + waveglow_time)

print("length of audio synthesized :{} sec".format(len(audio.float().cpu().detach()[0])/22050))

