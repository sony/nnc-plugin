# Copyright 2023 Sony Group Corporation.
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


# python -m pip install fastapi uvicorn

import io
import json
import os
import re
import signal
import sys
import time
import zipfile

from nnabla import logger

pids = []


def signal_handler(signum, frame):
    import psutil
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGINT)
    sys.exit(0)


def server(args, queue):
    try:
        server_func(args, queue)
    except Exception as e:
        logger.log(99, f'Error {type(e)} [{e}]')
        raise
    finally:
        queue.put(None)
    return True


def server_func(args, queue):
    import uvicorn
    import socket

    from nnabla.utils.nnp_graph import NnpNetworkPass, NnpLoader
    import nnabla.utils.load as load
    import numpy as np
    import nnabla as nn

    labels = {}
    if os.path.exists(args.input_val):
        with open(args.input_val) as f:
            ls = f.readline().rstrip().split(',')
            if len(ls) == 2:
                label_names = ls[1].split(':')[1].split(';')
                label_names.pop(0)
                for num, name in enumerate(label_names):
                    labels[num] = name

    image = nn.utils.data_source_loader.load_image(args.image)

    def conv_model(model_string, bytesio):
        new = []
        for l in model_string.splitlines():
            l = l.rstrip().decode()
            l = re.sub(r'([^\\])\\([^\\])', r'\1\\\\\2', l)
            bytesio.write((l + '\n').encode())

    model = args.model
    nnp_info = load.load(model, prepare_data_iterator=False, batch_size=1)
    nnp = NnpLoader(model)

    callback = NnpNetworkPass()
    callback.set_batch_normalization_batch_stat_all(False)

    executor = nnp_info.proto.executor[0]

    input_index = -1
    for n, d in enumerate(executor.data_variable):
        if d.data_name == args.input_name:
            input_index = n
    if input_index < 0:
        logger.critical(
            f'ERROR: Cannot found input variable [{args.input_name}]')
        return

    output_index = -1
    for n, o in enumerate(executor.output_variable):
        if o.data_name == args.output_name:
            output_index = n
    if output_index < 0:
        logger.critical(
            f'ERROR: Cannot found output variable [{args.output_name}]')
        return
    net = nnp.get_network(executor.network_name, 1, callback)

    if args.attention_map_variable_name not in net.variables:
        logger.critical(
            f'ERROR: {args.attention_map_variable_name} not in {args.model}')
        return

    att_out = net.variables[args.attention_map_variable_name]
    inputs = list(net.inputs.values())[input_index]
    outputs = list(net.outputs.values())[output_index]
    inputs.d = image

    maps = {}
    if os.path.exists(args.map):
        with np.load(args.map, allow_pickle=True) as npz:
            for key in npz.keys():
                maps[key] = npz[key].tolist()

    sock = socket.socket()
    sock.bind(("", 0))
    host, port = sock.getsockname()
    # if args.host is not None:
    #     host = args.host
    # if args.port != 0:
    #     port = args.port
    host = 'localhost'
    sock.close()

    import fastapi
    from fastapi.middleware.cors import CORSMiddleware
    app = fastapi.FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    @app.get("/Version", tags=['nnabla'])
    async def api_version():
        return {"message": "1.0.0"}

    @app.get("/ImagePath", tags=['nnabla'])
    async def api_imagepath():
        return {"path": os.path.abspath(args.image)}

    @app.get("/MapInfo", tags=['nnabla'])
    async def api_mapinfo():
        return {'attention_map_shape': att_out.d[0, 0, :, :].shape}

    @app.get("/Infer", tags=['nnabla'])
    async def api_infer(map):
        input_map = np.array(json.loads(map), dtype=np.float32)
        if np.all(input_map == 0):
            outputs.forward()
        else:
            att_out_unlinked = att_out.get_unlinked_variable(need_grad=False)
            att_out_unlinked.rewire_on(att_out)
            att_out_unlinked.d = input_map[np.newaxis, np.newaxis, ...]
            outputs.forward()
            att_out.rewire_on(att_out_unlinked)
        att_map = att_out.d[0, 0, :, :].tolist()
        maps[args.image] = att_map

        np.savez(args.map, **maps)

        return {'result': outputs.d.tolist(),
                'att_map': att_map}

    @app.get("/Labels", tags=['nnabla'])
    async def api_labels():
        return {"labels": labels}

    @app.get("/Quit", tags=['nnabla'])
    async def api_quit():
        os.kill(os.getpid(), signal.SIGINT)
        return {}

    queue.put({'host': host,
               'port': port})
    uvicorn.run(app, host=host, port=port)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Attention Editor (single image)\n' +
        '\n' +
        'You can use this editor to change the attention map in an attention branch network to see how the inference results change.\n' +
        '\n' +
        'Attention Branch Network: Learning of Attention Mechanism for Visual Explanation\n' +
        'Hiroshi Fukui, Tsubasa Hirakawa, Takayoshi Yamashita, Hironobu Fujiyoshi\n' +
        'https://arxiv.org/abs/1812.10025\n' +
        '', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-v',
        '--input-val',
        help='path to validation dataset csv file (csv)',
        default="")

    parser.add_argument(
        '-i',
        '--image',
        type=str,
        help='path to image file (image) default=None',
        default=None)

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='path to model nnp file (model) default=results.nnp',
        default='results.nnp')

    parser.add_argument(
        '-M',
        '--map',
        type=str,
        help='path to attention map file (map) default=abn.npz',
        default='abn.npz')

    parser.add_argument(
        '-A',
        '--attention-map-variable-name',
        type=str,
        help='Name of the attention map variable default=Attention_Map',
        default='Attention_Map')

    parser.add_argument(
        '-I',
        '--input-name',
        type=str,
        help='Name of input variable default=x',
        default='x')

    parser.add_argument(
        '-O',
        '--output-name',
        type=str,
        help='Name of output variable default=y\'',
        default='y\'')

    # parser.add_argument(
    #    '--host',
    #    type=str,
    #    help='Specify host name',
    #    default='localhost')

    # parser.add_argument(
    #     '-p',
    #     '--port',
    #     type=int,
    #     help='Specify port number',
    #     default=0)

    parser.add_argument(
        '-e',
        '--editor',
        type=str,
        help='Path to editor executable.',
        default=None)

    args = parser.parse_args()

    import multiprocessing
    import subprocess
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=server, args=(args, q,))
    p.start()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        settings = q.get(timeout=120)
    except:
        settings = None
    if settings is not None:
        if args.editor is not None and os.path.isfile(args.editor):
            editor = args.editor
        else:
            editor = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'attention_editor_binary',
                                  'attention-editor.exe')

        os.environ['AE_SERVICE_ADDR'] = f"http://{settings['host']}:{settings['port']}/"
        proc = subprocess.run([editor])

    os.kill(p.pid, signal.SIGINT)
    p.join()

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    logger.log(99, 'Attention Editor completed successfully.')
    return True


if __name__ == '__main__':
    main()
