import requests
import json
import time
from random import randint
import urllib3
import argparse
import time
import GPUtil

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
opt = parser.parse_args()

url = 'http://pred.ga:5000'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

class GPU:
	def __init__(self, GPU_ID):
		self.id = GPU_ID
	
	def post(self, postfix: str, req_json: dict):
		res = requests.post(url+postfix, data=json.dumps(req_json), headers=headers, verify=False)
		return res
	
	def sample_post(self, msg:str):
		sample = {'gpuID': self.id, 'message': msg}
		res = self.post(postfix='/log', req_json=sample)
		print(f'msg_sent: {sample}')
		print(f'msg_req : {res}')
#load = GPUs[0].load


gpu_instance = GPU(opt.gpu_id)

while True:
	try:
		# time.sleep(0.1)
		gpu = GPUtil.getGPUs()[0]
		_load = int(gpu.load*100)
		_mem = int(gpu.memoryUsed / gpu.memoryTotal * 100)
		gpu_instance.sample_post(f'{_load:>4}%')
			# gpu_instance.sample_post(f'LOAD: {_load:>3}%  [{"|"*int(_load/2):<50}]\t MEM: {int(_mem):>3}%  [{"|"*int(_mem/2):<50}]')
	except Exception as e:
		print('cannot reach server')
