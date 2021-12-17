import requests
import json
import time
from random import randint
import urllib3
import argparse
import time
import GPUtil
# from io import BytesIO
import io
import base64
from PIL import Image
from src.inference import Tester

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu_id', type=int, help='gpu id')
opt = parser.parse_args()

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

class Checker():
	def __init__(self, gpu_id):
		self.url =' http://pred.ga:8083'
		self.id = gpu_id
		self.tester = Tester()
	
	def get(self, postfix: str, req_json: dict):
		res = requests.get(self.url+postfix, data=json.dumps(req_json), headers=headers, verify=False)
		return res

	def get_image(self):
		req_json = {'gpuID': self.id}
		res = self.get('/image/get', req_json)

		if res.status_code == 200:
			print('\n'*3)
			print(type(res.content))
			print('\n'*3)
			
			img = Image.open(io.BytesIO(base64.b64decode(res.content)))
			img.save('./byte.jpg')
			print(img.size)

			startTime = time.time()
			result = self.tester.get_result(img)
			print(f'result: {result}\n{time.time()-startTime:.2f}\n')
		else:
			print('no Image')
			
			# img = img.convert('RGB')
			
			# buff = BytesIO()
			# img.save(buff, format='JPEG')
			# img.save('./hi.jpg')
			# print('image saved!')
			


checker = Checker(opt.gpu_id)
while True:
	# time.sleep(0.1)
	checker.get_image()
	# ins = input('>>')
	# break