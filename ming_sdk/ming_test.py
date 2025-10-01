import torch
import torchaudio
from ming_sdk.ming import Ming

# text generate
def test_text_generate():
    
    res = ming.generate(text="介绍一下杭州")
    print(res)
    assert res is not None


# text stream generate
def test_text_generate_stream():
    all_text = ""
    request_id = ""
    for text, request_id in ming.generate_stream(
        text="介绍一下杭州", max_new_tokens=128
    ):
        all_text += text
    print(f"request_id:{request_id},text={all_text}")
    assert text is not None


# generate audio
def test_audio_generate():
    output_audio_path = "test.wav"
    waveform = ming.generate(
        text="介绍一下杭州", output_type="speech", max_new_tokens=128
    )
    torchaudio.save(output_audio_path, waveform, 24000)
    assert os.path.exists(output_audio_path)


# steam generate audio
def test_audio_generate_stream():
    all_wavs = []
    all_text = ""
    request_id = ""
    output_audio_path = "test_stream.wav"
    for tts_speech, text, request_id in ming.generate_stream(
        text="介绍一下杭州", output_type="speech", max_new_tokens=128
    ):
        all_text += text
        all_wavs.append(tts_speech)
    waveform = torch.cat(all_wavs, dim=-1)
    torchaudio.save(output_audio_path, waveform, 24000)
    print(f"request_id:{request_id},audio:{output_audio_path},text={all_text}")
    assert os.path.exists(output_audio_path)


# steam generate audio interrupte
def test_audio_generate_stream_interrupte():
    all_wavs = []
    all_text = ""
    request_id = ""
    output_audio_path = "test_stream.wav"
    for tts_speech, text, request_id in ming.generate_stream(
        text="介绍一下杭州", output_type="speech", max_new_tokens=128
    ):
        all_text += text
        all_wavs.append(tts_speech)
        if len(all_text) > 20:
            ming.generate_interrupte(request_id)
    waveform = torch.cat(all_wavs, dim=-1)
    torchaudio.save(output_audio_path, waveform, 24000)
    print(f"request_id:{request_id},audio:{output_audio_path},text={all_text}")
    assert os.path.exists(output_audio_path)


def test_image_generate():

    image_t2i = ming.generate(text="生成一张图 美女在沙滩上奔跑", output_type="image")
    pic_path = "auto_t2i.jpg"
    image_t2i.save(pic_path)
    assert os.path.exists(pic_path)


def test_image_qa():

    text = ming.generate(image="auto_t2i.jpg", output_type="text")
    # image_t2i.save("auto_t2i.jpg")
    print("test_image_qa", text)


def test_image_edit():
    image_t2i = "auto_t2i.jpg"
    image_edit = ming.generate(
        text="给人物戴上眼镜", image=[image_t2i], output_type="image"
    )
    image_edit_path = "auto_edit.jpg"
    image_edit.save(image_edit_path)
    assert os.path.exists(image_edit_path)


def test_audio_task():
    audio_url =  "https://raylet.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/ci/media-ai/test/xwlb-20s.wav"
    asr_res = ming.generate(
        text="Please recognize the language of this speech and transcribe it. Format: oral.",
        audio=audio_url,
    )
    print(asr_res)
    assert asr_res is not None

def test_image_task():
    pic_res = ming.generate(
        text="描述下这个图",
        image=["auto_t2i.jpg"],
    )
    print(pic_res)
    assert pic_res is not None


def test_tts():
    output_audio_path = "test_tts.wav"
    waveform = ming.tts(text="我爱北京故宫")
    torchaudio.save(output_audio_path, waveform, 24000)
    assert os.path.exists(output_audio_path)


def test_video():
    video = 'test.mp4'
    text = ming.generate(text='详细描述一下这段视频', video=video, output_type="text")
    print("test_video", text)

if __name__ == "__main__":
    model_path = "YOUR_MODEL_PATH"
    
    ming = Ming(
      model_path=model_path, 
      driver="0", 
      talker_with_vllm=True,
      gpu_memory_utilization = {"moe": 0.55,"talker": 0.17}
    )

    print("------TTS----------")
    test_tts()

    test_image_generate()
    print("----------------")
    print("done")

    print("----------------")
    test_image_generate()

    print("------TTS----------")
    test_tts()

    print("----------------")
    test_audio_generate_stream()
    
    print("----------------")
    test_text_generate()
    print("----------------")
    test_text_generate_stream()
    
    test_audio_generate_stream_interrupte()
    print("----------------")
    test_image_generate()
    print("----------------")
    test_image_edit()
    print("----------------")
    test_audio_task()
    print("----------------")
    test_image_task()
    print("------TTS----------")
    test_tts()

    print("-----video----")
    test_video()