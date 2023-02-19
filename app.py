# coding=utf-8
import time

from scipy.io.wavfile import write

import utils
import commons
from models import SynthesizerTrn
from text import text_to_sequence
from torch import no_grad, LongTensor
import openai
from flask import Flask, request

app = Flask(__name__)
outdir = r'./cache'

hps_ms = utils.get_hparams_from_file(r'./model/config.json')
net_g_ms = SynthesizerTrn(
    len(hps_ms.symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
_ = net_g_ms.eval()
speakers = hps_ms.speakers
model, optimizer, learning_rate, epochs = utils.load_checkpoint(r'./model/G_953000.pth', net_g_ms, None)


def gpt3_chat(prompt) -> str:
    openai.api_key = "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"
    model_engine = "text-davinci-003"
    # Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
        timeout=5,
    )
    response = completion.choices[0].text
    return response


def get_text(ss, hps, language):
    text = gpt3_chat(ss).strip()
    if language == 0:
        text = f"[ZH]{text}[ZH]"
    elif language == 1:
        text = f"[JA]{text}[JA]"
    else:
        text = f"{text}"
    with open(outdir + "/temp.txt", "w", encoding="utf-8") as f:
        f.write(text)
    text_norm, clean_text = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text


def vits(text, language, speaker_id, noise_scale, noise_scale_w, length_scale):
    start = time.perf_counter()
    if not len(text):
        return "输入文本不能为空！", None, None
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
    if len(text) > 100:
        return f"输入文字过长！{len(text)}>100", None, None
    stn_tst, clean_text = get_text(text, hps_ms, language)
    print(f"stn_tst:{stn_tst} ")
    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        speaker_id = LongTensor([speaker_id])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                               length_scale=length_scale)[0][0, 0].data.float().numpy()
        write(outdir+"/output.wav", hps_ms.data.sampling_rate, audio)
    return "生成成功!", (22050, audio), f"生成耗时 {round(time.perf_counter()-start, 2)} s"


@app.route('/gpt')
def text_api():
    text = request.args.get('text', '')
    language = 0
    vits(text, language, 228, 0.6, 0.668, 1.2)
    with open(outdir+"/output.wav", 'rb') as bit:
        wav_bytes = bit.read()
    return wav_bytes, 200, {'Content-Type': 'audio/ogg'}


@app.route('/word')
def show():
    with open(outdir + "/temp.txt", "r", encoding="utf-8") as f1:
        text = f1.read()
        return text.replace('[JA]', '').replace('[ZH]', '')


if __name__ == '__main__':
    app.run("0.0.0.0", 8080)