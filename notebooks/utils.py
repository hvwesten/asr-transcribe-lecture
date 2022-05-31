from datasets import Dataset, Audio
import re


chars_to_ignore_regex = "[\,\?\.\!\-\;\:\"½+-0123456789&%$()=><…—–\n]"

replace_dict = {
    'à': 'a',
    'â': 'a',
    'é': 'e',
    'ï': 'i',
    '”': '"',
    '“': '"',
    '‘': "'",
    '’': "'",
}

# chars_to_ignore_regex = '[\,\?\.\!\-\;\:"]'
# chars_to_ignore_regex = '[\,\?\.\!\-\;\:\½"]'

# ignore_list = ['½', 'à', 'â', 'é', 'ï', '–', '—', '‘', '’', '“', '”', '…<', '=', '>',
#                '$', '%', '&', '(', ')', '+', '/', '0', '1', '2', '3', '4', '5', '6',
#                '7', '8', '9']
# '%': 'percent',
# '$': 'dollar',
# '+': 'plus',
# '-': 'minus',
# '½': 'half',

def retrieve_text(batch):
    # load the contents of the file as a string
    txt_file = batch["txt"]
    with open(txt_file, 'r') as f:
        text = f.read()

    for k, v in replace_dict.items():
        text = text.replace(k, v)

    # text = re.sub('[\n]', ' ', text)

    # text = re.sub(chars_to_replace_1, '"', text)

    # do some processing
    batch["txt"] = re.sub(chars_to_ignore_regex, ' ', text).lower()
    return batch


def load_custom_dataset(audio_dir, transcripts_dir, split=True):
    mp3_files = [str(audio_file) for audio_file in audio_dir.glob('*.mp3')]
    # mp3_files = sorted(mp3_files)[:7]

    txt_files = [str(text_file) for text_file in transcripts_dir.glob('*.txt')]
    # txt_files = sorted(txt_files)[:7]

    data_dict = {
        'mp3': mp3_files,
        'txt': txt_files,
    }

    dataset = Dataset.from_dict(data_dict, split="all")

    if split:
        dataset = dataset.train_test_split(test_size=0.2)
    dataset = dataset.cast_column("mp3", Audio(sampling_rate=16_000))

    dataset = dataset.map(retrieve_text)

    return dataset


def fix_arpa_file(in_file, out_file):
    with open(in_file, "r") as read_file, open(out_file, "w") as write_file:
        has_added_eos = False
        for line in read_file:
            if not has_added_eos and "ngram 1=" in line:
                count=line.strip().split("=")[-1]
                write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
            elif not has_added_eos and "<s>" in line:
                write_file.write(line)
                write_file.write(line.replace("<s>", "</s>"))
                has_added_eos = True
            else:
                write_file.write(line)
