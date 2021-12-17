from flask import Flask, request
from flask_cors import CORS
from io import BytesIO
from PIL import Image
from io import BytesIO
import base64
from src.inference import Tester

app = Flask (__name__)
CORS(app)
tester = Tester()

@app.route('/inference', methods = ['POST'])
def inference():
    data = request.get_json()
    img = Image.open(BytesIO(base64.b64decode(data['image'].split(',')[-1])))
    img = img.convert('RGB')

    buffered = BytesIO()
    img_str = base64.b64encode(buffered.getvalue())
    result = tester.get_result(img)
    print(result, end='\n\n')

    return result


if __name__ == "__main__":
    app.run(host='localhost',port=6006)