import src.constants as c
from src.pre_process import extract_features
from src.test_model import batch_cosine_similarity
from scipy.io.wavfile import read
import numpy as np
import base64
from pydub import AudioSegment
from src import silence_detector
import librosa

def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames:
        clipped_x = x[0: num_frames]
    else:
        clipped_x = x

    return clipped_x

def convertBase64CafToWav(cafBase64):
    cafFile = base64.b64decode(cafBase64);
    with open("audio.caf", "wb") as fh:
        fh.write(cafFile)
    flac_audio = AudioSegment.from_file("audio.caf", "caf")
    flac_audio.export("sampleWave.wav", format="wav")
    with open("sampleWave.wav", "rb") as voice_file:
        encoded_string = base64.b64encode(voice_file.read()).decode('ascii')
        return encoded_string;
    return '';

def VAD(audio):
    chunk_size = int(c.SAMPLE_RATE*0.05) # 50ms
    index = 0
    sil_detector = silence_detector.SilenceDetector(15)
    nonsil_audio=[]
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index+chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)

def read_audio(filename, sample_rate=c.SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = VAD(audio.flatten())
    start_sec, end_sec = c.TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * c.SAMPLE_RATE)
    end_frame = int(end_sec * c.SAMPLE_RATE)

    if len(audio) < (end_frame - start_frame):
        # au = [0] * (end_frame - start_frame)
        # for i in range(len(audio)):
        #     au[i] = audio[i]
        # audio = np.array(au)
        k = int(((end_frame - start_frame)/len(audio))+1)
        audio = np.tile(audio,k)
    return audio

def compareTwoVoice(model, embeddingInput, sampleItem):
    feat2 = None
    try:
        dir = "sample-npy/"+sampleItem["_id"]+".npy";
        feat2 = np.load(dir)
    except:
        feat2 = None
    if feat2 is None:
        sampleBase64Wav = sampleItem["voice_wav"]
        cafSampleFile = base64.b64decode(sampleBase64Wav);
        with open("wav_sample.wav", "wb") as fh:
            fh.write(cafSampleFile)
        utt2 = read_audio('wav_sample.wav')
        # utt2 = utt2 / (2**15 - 1)
        feat2 = extract_features(utt2)
        feat2 = clipped_audio(feat2)
        feat2 = feat2[np.newaxis, ...]
        np.save("sample-npy/"+sampleItem["_id"], feat2)
    emb2 = model.predict(feat2)
    #print(emb1)
    # similarity
    # mul = np.multiply(emb1, emb2)
    # s = np.sum(mul, axis=1)
    # print(s)
    similarity=batch_cosine_similarity(embeddingInput,emb2)
    # print(cdist(emb1,  emb2, metric="cosine"))
    print(sampleItem["name"]+": "+str(similarity))
    return similarity;

def predictEmbedding(model, inputBase64Wav):
    cafInputFile = base64.b64decode(inputBase64Wav);
    with open("wav_input.wav", "wb") as fh:
        fh.write(cafInputFile)

    utt1 = read_audio('wav_input.wav')
    # utt1 = utt1 / (2**15 - 1)
    feat1 = extract_features(utt1)
    feat1 = clipped_audio(feat1)
    feat1 = feat1[np.newaxis, ...]
    emb1 = model.predict(feat1)
    return emb1;