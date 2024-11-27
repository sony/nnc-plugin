# Copyright 2024 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import uuid
import json
from datetime import datetime
import webbrowser
import argparse
import subprocess
import csv
import io

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import openai


api_key = os.environ.get('OPENAI_API_KEY_NNC')
log_path = ''
client = None
if "CHATGPT_PORT" in os.environ:
    CHATGPT_PORT = os.environ['CHATGPT_PORT']
else:
    CHATGPT_PORT = 9654


def init(args):
    global log_path
    global client

    if args.output_dir != "":
        log_path = args.output_dir
    else:
        log_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    webbrowser.open(f'http://127.0.0.1:{CHATGPT_PORT}/')

    client = openai.OpenAI(api_key=api_key)


app = Flask(__name__)
CORS(app,
     resources={r"/": {"origins": "*"},
                r"/chatbot": {"origins": "*"},
                r"/history": {"origins": "*"},
                r"/open_log_folder": {"origins": "*"},
                r"/save_system_prompt": {"origins": "*"},
                r"/get_system_prompt": {"origins": "*"},
                r"/edit_system_prompt": {"origins": "*"},

                r"/files": {"origins": "*"},
                r"/get_file": {"origins": "*"},
                r"/delete_file": {"origins": "*"},
                r"/upload_dataset": {"origins": "*"},
                r"/csv_to_jsonl": {"origins": "*"},

                r"/jobs": {"origins": "*"},
                r"/get_job": {"origins": "*"},
                r"/cancel_job": {"origins": "*"},
                r"/create_finetune_job": {"origins": "*"},

                r"/models": {"origins": "*"},

                r"/check": {"origins": "*"}},
     origins=[f"http://localhost:{CHATGPT_PORT}"])


@app.before_request
def limit_remote_addr():
    allowed_ips = ['127.0.0.1']

    if request.remote_addr not in allowed_ips:
        return 'Forbidden', 403


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json
    try:
        response = client.chat.completions.create(
            model=user_message['model'],
            messages=user_message['history'],
            temperature=float(user_message['temperature']),
            top_p=float(user_message['top_p']),
            max_tokens=int(user_message['max_tokens']),
            frequency_penalty=float(user_message['frequency_penalty']),
            presence_penalty=float(user_message['presence_penalty'])
        )

        response_text = response.choices[0].message.content
        user_message['history'].append(
            {'role': 'assistant', 'content': response_text})

        uuid.UUID(user_message['id'])

        now = datetime.now()
        datetime_string = now.strftime("%Y-%m-%d %H:%M:%S")

        user_message['date'] = datetime_string
        global log_path
        with open(os.path.join(log_path, user_message['id'] + '.json'), 'w') as f:
            json.dump(user_message, f)

        return jsonify({'success': True, 'message': response_text, 'usage': f'Usage : Input {response.usage.prompt_tokens}, Output {response.usage.completion_tokens}, Total {response.usage.total_tokens}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e), 'usage': 'An error occurred while calling OpenAI API.'})


@app.route('/history', methods=['GET'])
def history():
    args = request.args
    if 'id' in args:
        try:
            with open(os.path.join(log_path, args['id'] + '.json'), 'r') as f:
                data = json.load(f)
            return data
        except:
            return jsonify({'success': False})
    else:
        file_list = os.listdir(log_path)
        files = []
        for file_name in file_list:
            try:
                uuid.UUID(os.path.splitext(file_name)[0])
                if os.path.splitext(file_name)[1].lower() == '.json':
                    file_path = os.path.join(log_path, file_name)
                    created_time = os.path.getctime(file_path)
                    created_datetime = datetime.fromtimestamp(created_time)
                    created_time_formatted = created_datetime.strftime(
                        "%Y-%m-%d %H:%M:%S")

                    with open(os.path.join(log_path, file_name), "r") as f:
                        data = json.load(f)
                        for item in data['history']:
                            if item['role'] == 'user':
                                text = item['content']
                                break

                    files.append({'date': created_time_formatted, 'id': os.path.splitext(
                        os.path.basename(file_path))[0], 'q': text})
            except:
                pass
        files = sorted(files, key=lambda x: x["date"], reverse=True)
        return jsonify({'success': True, 'history': files})


@app.route('/open_log_folder', methods=['Get'])
def open_log_folder():
    global log_path
    subprocess.run(['explorer', log_path])
    return jsonify({'success': True})


@app.route('/check', methods=['Get'])
def check():
    if openai.api_key != "":
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})


def escape_ret(s):
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    return s


def unescape_ret(s):
    s = s.replace('\\n', '\n')
    s = s.replace('\\r', '\r')
    return str(s)


@app.route('/save_system_prompt', methods=['POST'])
def save_system_prompt():
    with open(os.path.join(log_path, 'system_prompt.txt'), 'a') as f:
        f.write(escape_ret(request.json['system_prompt']) + '\n')
    return jsonify({'success': True})


@app.route('/get_system_prompt', methods=['GET'])
def get_system_prompt():
    system_prompt = []
    try:
        with open(os.path.join(log_path, 'system_prompt.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            system_prompt.append(unescape_ret(line))
    except:
        pass
    return jsonify({'success': True, 'system_prompt': system_prompt})


@app.route('/edit_system_prompt', methods=['Get'])
def edit_system_prompt():
    global log_path
    subprocess.run(['notepad', os.path.join(log_path, 'system_prompt.txt')])
    return jsonify({'success': True})


@app.route('/files', methods=['GET'])
def files():
    response = client.files.list()
    filelist = [{'id': file.id, 'filename': file.filename,
                 'date': datetime.utcfromtimestamp(int(file.created_at)).strftime('%Y-%m-%d %H:%M:%S')}
                for file in response.data]
    filelist = sorted(filelist, key=lambda x: x['date'], reverse=True)
    return jsonify({'success': True, 'filelist': filelist})


@app.route('/get_file', methods=['GET'])
def get_file():
    args = request.args
    if 'id' in args:
        response = client.files.retrieve(args['id'])
        result = {k: v for k, v in response}
        return jsonify({'success': True, 'file': result})
    return jsonify({'success': False})


@app.route('/delete_file', methods=['GET'])
def delete_file():
    try:
        args = request.args
        if 'id' in args:
            client.files.delete(args['id'])
            return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    try:
        file = request.files['file']
        response = client.files.create(
            file=file.stream.read(),
            purpose='fine-tune',
            extra_headers={
                'Content-Disposition': f'form-data; name="file"; filename="{file.filename}"'}
        )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/csv_to_jsonl', methods=['POST'])
def csv_to_jsonl():
    try:
        file = request.files['file']
        text_data = file.stream.read().decode('utf-8-sig')
        text_data_io = io.StringIO(text_data)

        new_filename = os.path.splitext(
            os.path.basename(file.filename))[0] + '.jsonl'
        jsonl = []
        csv_reader = csv.reader(text_data_io)
        lines = [row for row in csv_reader]
        header = lines.pop(0)
        for line in lines:
            messages = []
            for role, content in zip(header, line):
                messages.append({'role': role, 'content': content})
            jsonl.append(json.dumps({'messages': messages}))

        response = Response(
            '\n'.join(jsonl), content_type='text/plain; charset=utf-8')
        response.headers['Content-Disposition'] = 'attachment; filename=' + new_filename

        return response
    except Exception as e:
        error_message = json.dumps({"error": str(e)})
        return Response(error_message, status=400, mimetype='application/json')


@app.route('/jobs', methods=['GET'])
def jobs():
    response = client.fine_tuning.jobs.list()
    joblist = []
    for job in response.data:
        jobitem = {'id': job.id,
                   'model': job.model,
                   'date': datetime.utcfromtimestamp(int(job.created_at)).strftime('%Y-%m-%d %H:%M:%S')}
        joblist.append(jobitem)
    joblist = sorted(joblist, key=lambda x: x['date'], reverse=True)
    return jsonify({'success': True, 'joblist': joblist})


@app.route('/get_job', methods=['GET'])
def get_job():
    args = request.args
    if 'id' in args:
        response = client.fine_tuning.jobs.retrieve(args['id'])
        if response.id is None:
            result = [
                ("id", "N.A."),
                ("error message", "The job with specified id is not found!")
            ]
            return jsonify({'success': True, 'job': result})
        result = [
            ("id", response.id),
            ("model", response.model),
            ("fine_tuned_model", response.fine_tuned_model),
            ("created_at", datetime.utcfromtimestamp(
                int(response.created_at)).strftime('%Y-%m-%d %H:%M:%S')),
            ("finished_at", datetime.utcfromtimestamp(int(response.finished_at)).strftime(
                '%Y-%m-%d %H:%M:%S') if response.finished_at is not None else "N.A."),
            ("organization_id", response.organization_id),
            ("result_files", ",".join(
                [file for file in response.result_files])),
            ("status", response.status),
            ("validation_file", response.validation_file),
            ("training_file", response.training_file),
            ("trained_tokens", response.trained_tokens),
            ("error message", response.error.message if response.error is not None else "")
        ]
        return jsonify({'success': True, 'job': result})
    return jsonify({'success': False})


@app.route('/cancel_job', methods=['GET'])
def cancel_job():
    try:
        args = request.args
        if 'id' in args:
            client.fine_tuning.jobs.cancel(args['id'])
            return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/create_finetune_job', methods=['POST'])
def create_finetune_job():
    user_message = request.json
    try:
        response = client.fine_tuning.jobs.create(
            training_file=user_message['training_file'],
            model=user_message['model']
        )
        return jsonify({'success': True, 'id': response.id})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/models', methods=['GET'])
def models():
    try:
        response = client.models.list()
        modellist_ft = []
        for model in response.data:
            modelitem = {'id': model.id,
                         'date': datetime.utcfromtimestamp(int(model.created)).strftime('%Y-%m-%d %H:%M:%S')}
            modellist_ft.append(modelitem)
        modellist = sorted(modellist_ft, key=lambda x: x['date'], reverse=True)
        return jsonify({'success': True, 'modellist': modellist})
    except Exception as e:
        print(e)
        return jsonify({'success': False, 'message': 'Please set OPENAI_API_KEY at first!\n' + str(e)})


def func(args):
    init(args)
    app.run(port=CHATGPT_PORT)


def main():
    parser = argparse.ArgumentParser(
        description='Chatbot using OpenAI API\n\n' +
        'ChatGPT clone using OpenAI API\n' +
        'https://platform.openai.com/docs/guides/gpt\n\n',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o',
        '--output_dir',
        help='directory to store chat logs',
        default='',
        required=False)
    parser.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
