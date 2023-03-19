import argparse
import os
from whisper import load_model
import whisperx
import torch
import librosa
import soundfile
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import wget
from omegaconf import OmegaConf
import json
import shutil
import openai
APIKEY = os.environ["OPENAI_API_KEY"]
openai.api_key = APIKEY

""" Auto transcribe and diarize a batch of audio files, titling the most-frequent speaker as the 'instructor'

This is a work in progress to maintain consistent speaker numbers matching voice embeddings throughout the entire batch

Thanks to MahmoudAshraf97 for the command line integration of WhisperX and NeMo, adapted below for API key usage
    """
    
def batch_diarize_audio(input_audios, model_name="API", stemming=False):
    all_results = []
    for input_audio in input_audios:
        wsm, ssm = diarize_audio(input_audio, model_name, stemming)
        # Extract the instructor's embeddings from the first file, if not already extracted
        instructor_speaker_number=identify_most_frequent_speaker_number(ssm)
        instructor_embeddings = extract_instructor_embeddings(wsm, instructor_speaker_number)
        ssm = update_speaker_numbers(ssm, instructor_speaker_number)
        all_results.append((input_audio,wsm,ssm,instructor_speaker_number,instructor_embeddings))

        # Write the updated speaker-wise sentence mappings to the output files
        with open(f"{input_audio[:-4]}.txt", "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f)
        with open(f"{input_audio[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt)
    return all_results

def update_speaker_numbers(ssm, instructor_speaker_number):
    # 1. Iterate through ssm
    # 2. replace instructor_speaker_number with "instructor"
    # 3. Return the updated sentence speaker list

    new_ssm = []
    for segment in ssm:
        if instructor_speaker_number == segment['speaker']:
            segment['speaker'] = "Instructor" # Replace the instructor speaker number with "Instructor"
            snt = {
                "speaker": f"Instructor",
                "start_time": segment['start_time'],
                "end_time": segment['end_time'],
                "text": segment['text'],
            }
            new_ssm.append(snt)
        else:
            new_ssm.append(segment)

    return new_ssm

def identify_most_frequent_speaker_number(ssm):
    # Extract the speaker number assigned to the instructor
    # We can assume that the instructor is the most frequent speaker in the sample
    speaker_counts = {}
    for segment in ssm:
        speaker = segment['speaker']
        if speaker in speaker_counts:
            speaker_counts[speaker] += 1
        else:
            speaker_counts[speaker] = 1
    # Find the speaker with the highest count
    instructor_speaker_number = max(speaker_counts, key=speaker_counts.get)

    return instructor_speaker_number

    
# Implement the `extract_instructor_embeddings` function to extract the instructor's embeddings from the speaker-wise word mappings and sentence mappings.
def extract_instructor_embeddings(wsm, instructor_speaker_number):
    # 1. Initialize an empty list to store the instructor's embeddings
    # 2. Iterate through the speaker-wise word mappings (wsm) and extract the embeddings for the instructor's speaker number
    # 3. Return the instructor's embeddings

    instructor_embeddings = []

    for word_dict in wsm:
        if word_dict['speaker'] == instructor_speaker_number:
            instructor_embeddings.append(word_dict['embedding'])
    return instructor_embeddings

def diarize_audio(input_audio, model_name="medium.en", stemming=False):


    punct_model_langs = [
     "en",
     "fr",
     "de",
     "es",
     "it",
     "nl",
     "pt",
     "bg",
     "pl",
     "cs",
     "sk",
     "sl",
     ]
    wav2vec2_langs = [
     "en",
     "fr",
     "de",
     "es",
     "it",
     "nl",
     "pt",
     "ja",
     "zh",
     "uk",
     "pt",
     "ar",
     "ru",
     "pl",
     "hu",
     "fi",
     "fa",
     "el",
     "tr",
     ]

    if stemming:
        # Isolate vocals from the rest of the audio/music, default to false for vocals only interviews
        return_code = os.system(
            f'python3 -m demucs.separate -n htdemucs_ft --two-stems=vocals "{input_audio}" -o "temp_outputs"'
        )
        if return_code != 0:
            print(
                "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
            )
            vocal_target = input_audio
        else:
            vocal_target = f"temp_outputs/htdemucs_ft/{input_audio[:-4]}/vocals.wav"
    else:
        vocal_target = input_audio

    # Large models result in considerably better and more aligned (words, timestamps) mapping.
    if model_name=="API":
        whisper_results = openai.Audio.transcribe("whisper-1", vocal_target, beam_size=None, verbose=False)

    else:
        whisper_model = load_model(model_name)
        whisper_results = whisper_model.transcribe(vocal_target, beam_size=None, verbose=False)

    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()

    device = "cuda"
    alignment_model, metadata = whisperx.load_align_model(
        language_code=whisper_results["language"], device=device
    )
    result_aligned = whisperx.align(
        whisper_results["segments"], alignment_model, metadata, vocal_target, device
    )

    # clear gpu vram
    del alignment_model
    torch.cuda.empty_cache()

    # convert audio to mono for NeMo combatibility
    signal, sample_rate = librosa.load(vocal_target, sr=None)
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    os.chdir(temp_path)
    soundfile.write("mono_file.wav", signal, sample_rate, "PCM_24")

    # Initialize NeMo MSDD diarization model
    msdd_model = NeuralDiarizer(cfg=create_config())
    msdd_model.diarize()

    del msdd_model
    torch.cuda.empty_cache()

    # Reading timestamps <> Speaker Labels mapping

    output_dir = "nemo_outputs"

    speaker_ts = []
    with open(f"{output_dir}/pred_rttms/mono_file.rttm", "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(result_aligned["word_segments"], speaker_ts, "start")

    if whisper_results["language"] in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        punct_model = PunctuationModel(model="kredor/punctuate-all")

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    else:
        print(
            f'Punctuation restoration is not available for {whisper_results["language"]} language.'
        )

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    os.chdir(ROOT)  # back to parent dir
    with open(f"{input_audio[:-4]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{input_audio[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    cleanup(temp_path)
    #return the speaker-wise word mappings and sentences mappings, 
    # so that we can use this information to maintain consistent speaker numbering across all files in batch
    return (wsm,ssm)

def create_config():
    data_dir = "./"
    DOMAIN_TYPE = "telephonic"  # Can be meeting or telephonic based on domain type of the audio file
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = os.path.join(data_dir, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG):
        MODEL_CONFIG = wget.download(CONFIG_URL, data_dir)
    config = OmegaConf.load(MODEL_CONFIG)
    ROOT = os.getcwd()
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    meta = {
        "audio_filepath": "mono_file.wav",
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open("data/input_manifest.json", "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")
    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"
    config.num_workers = 1  # Workaround for multiprocessing hanging with ipython issue
    output_dir = "nemo_outputs"  # os.path.join(ROOT, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    config.diarizer.manifest_filepath = "data/input_manifest.json"
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VAD provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = False
    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )
    return config
def get_word_ts_anchor(s, e, option="start"):
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s
def get_words_speaker_mapping(wrd_ts, spk_ts, word_anchor_option="start"):
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    for wrd_dict in wrd_ts:
        ws, we, wrd = (int(wrd_dict["start"] * 1000),int(wrd_dict["end"] * 1000),wrd_dict["text"], )
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")
        wrd_spk_mapping.append(
            {"word": wrd, "start_time": ws, "end_time": we, "speaker": sp} )
    return wrd_spk_mapping
sentence_ending_punctuations = ".?!"
def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list, max_words):
    is_word_sentence_end = (lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations    )
    left_idx = word_idx
    while (left_idx > 0 and word_idx - left_idx < max_words and speaker_list[left_idx - 1] == speaker_list[left_idx] and not is_word_sentence_end(left_idx - 1)): left_idx -= 1
    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1
def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    is_word_sentence_end = ( lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations    )
    right_idx = word_idx
    while (right_idx < len(word_list) and right_idx - word_idx < max_words and not is_word_sentence_end(right_idx)):
        right_idx += 1
    return (right_idx if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx) else -1 )
def get_realigned_ws_mapping_with_punctuation(word_speaker_mapping, max_words_in_sentence=50):
    is_word_sentence_end = (lambda x: x >= 0 and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations)
    wsp_len = len(word_speaker_mapping)
    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)
    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        if ( k < wsp_len - 1 and speaker_list[k] != speaker_list[k + 1] and not is_word_sentence_end(k)):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (get_last_word_idx_of_sentence( k, words_list, max_words_in_sentence - k + left_idx - 1 ) if left_idx > -1   else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue
            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue
            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (right_idx - left_idx + 1)
            k = right_idx
        k += 1
    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy()
        line_dict["speaker"] = speaker_list[k]
        realigned_list.append(line_dict)
        k += 1
    return realigned_list
def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    s, e, spk = spk_ts[0]
    prev_spk = spk
    snts = []
    snt = {"speaker": f"Speaker {spk}", "start_time": s, "end_time": e, "text": ""}
    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"], wrd_dict["end_time"]
        if spk != prev_spk:
            snts.append(snt)
            snt = {
                "speaker": f"Speaker {spk}",
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e
        snt["text"] += wrd + " "
        prev_spk = spk
    snts.append(snt)
    return snts
def get_speaker_aware_transcript(sentences_speaker_mapping, f):
    for sentence_dict in sentences_speaker_mapping:
        sp = sentence_dict["speaker"]
        text = sentence_dict["text"]
        f.write(f"\n\n{sp}: {text}")
def format_timestamp(
    milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert milliseconds >= 0, "non-negative timestamp expected"
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}")
def write_srt(transcript, file):
    """ Write a transcript to a file in SRT format."""
    for i, segment in enumerate(transcript, start=1):
        # write srt lines
        print(f"{i}\n"
            f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,)
def cleanup(path: str):
    """path could either be relative or absolute."""
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path)
    else:
        raise ValueError("Path {} is not a file or dir.".format(path))
