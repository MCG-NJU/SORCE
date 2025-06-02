import os
import torch
from tqdm import tqdm
from accelerate import Accelerator

from PIL import Image
from datasets import load_from_disk, load_dataset

import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from peft import PeftModel

try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
except ImportError:
    print("Qwen2VL not found in current transformers version")
try:
    import clip
except ImportError:
    print("Package clip not found. Use pip install git+https://github.com/openai/CLIP.git")


DEBUG = False
MODEL_TYPE = 'llava'
EXTRA_PROMPTS = False
emb_data_func = None
PATCH_SPLIT = None

accelerator = Accelerator()

llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
qwen2_template = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [b[key] for b in batch]
    return collated_batch


def emb_data(model, transform, dataset, device,
             emb_type='text', prompt=None, bsz=4,
             text_column='caption', img_column='img'):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=3 * bsz if emb_type == 'text' else bsz,
        shuffle=False, num_workers=1,
        collate_fn=custom_collate_fn
    )
    dataloader = accelerator.prepare(dataloader)
    embs = []
    bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        if emb_type == 'text':
            if type(batch[text_column][0]) is list:
                input_texts = [prompt.replace('<sent>', text) for text in sum(batch[text_column], start=[])]
            else:
                input_texts = [prompt.replace('<sent>', text) for text in batch[text_column]]

            inputs = transform(text=input_texts,
                               return_tensors="pt", padding=True)
            for key in inputs:
                if inputs[key] is not None:
                    inputs[key] = inputs[key].to(device)
        else:
            input_texts = [prompt] * len(batch[img_column])
            if isinstance(batch[img_column][0], str):
                batch[img_column] = [Image.open(img).convert('RGB') for img in batch[img_column]]
            inputs = transform(text=input_texts,
                               images=batch[img_column], return_tensors="pt", padding=True).to(device)
            if EXTRA_PROMPTS:
                input_texts_main = [prompt.replace('above image', 'the main component in the above image')] * len(
                    batch[img_column])
                input_texts_background = [prompt.replace('above image', 'the background in the above image')] * len(
                    batch[img_column])
                input_texts_detail = [prompt.replace('above image', 'the detail in the above image')] * len(
                    batch[img_column])
                inputs_main = transform(text=input_texts_main,
                                        images=batch[img_column], return_tensors="pt", padding=True,
                                        max_length=1800).to(device)
                inputs_background = transform(text=input_texts_background,
                                              images=batch[img_column], return_tensors="pt", padding=True,
                                              max_length=1800).to(device)
                inputs_detail = transform(text=input_texts_detail,
                                          images=batch[img_column], return_tensors="pt", padding=True,
                                          max_length=1800).to(device)
        with torch.no_grad():
            emb = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            if emb_type == 'image' and EXTRA_PROMPTS:
                assert inputs_detail is not None and inputs_background is not None and inputs_main is not None
                emb_main = model(**inputs_main, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                emb_background = model(**inputs_background, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                emb_detail = model(**inputs_detail, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                emb = torch.cat([emb, emb_main, emb_background, emb_detail], dim=0)
            emb = F.normalize(emb, dim=-1)
        emb = accelerator.gather(emb)
        embs.append(emb.cpu().float())
        bar.update(1)
    embs = torch.cat(embs)
    total = 0
    for i in dataset:
        if emb_type == 'text' and type(i[text_column]) is list:
            total += len(i[text_column])
        else:
            total += 1
            if emb_type == 'image' and EXTRA_PROMPTS:
                total += 3
    bar.close()
    return embs[:total]


def emb_data_position(model, transform, dataset, device,
                      emb_type='text', prompt=None, bsz=4,
                      text_column='caption', img_column='img'):
    print("Using positional prompts")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=3 * bsz if emb_type == 'text' else bsz,
        shuffle=False, num_workers=1,
        collate_fn=custom_collate_fn
    )
    dataloader = accelerator.prepare(dataloader)
    embs = []
    bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        if emb_type == 'text':
            if type(batch[text_column][0]) is list:
                input_texts = [prompt.replace('<sent>', text) for text in sum(batch[text_column], start=[])]
            else:
                input_texts = [prompt.replace('<sent>', text) for text in batch[text_column]]

            inputs = transform(text=input_texts,
                               return_tensors="pt", padding=True)
            for key in inputs:
                if inputs[key] is not None:
                    inputs[key] = inputs[key].to(device)
        else:
            input_texts = [prompt] * len(batch[img_column])
            if isinstance(batch[img_column][0], str):
                batch[img_column] = [Image.open(img).convert('RGB') for img in batch[img_column]]
            inputs = transform(text=input_texts, images=batch[img_column], return_tensors="pt", padding=True).to(device)

            if EXTRA_PROMPTS:
                input_texts_lu = [prompt.replace('above image', 'the left upper corner in the above image')] * len(
                    batch[img_column])
                input_texts_lb = [prompt.replace('above image', 'the left lower corner in the above image')] * len(
                    batch[img_column])
                input_texts_ru = [prompt.replace('above image', 'the right upper corner in the above image')] * len(
                    batch[img_column])
                input_texts_rb = [prompt.replace('above image', 'the right lower corner in the above image')] * len(
                    batch[img_column])

                inputs_lu = transform(text=input_texts_lu,
                                      images=batch[img_column], return_tensors="pt", padding=True).to(device)
                inputs_lb = transform(text=input_texts_lb,
                                      images=batch[img_column], return_tensors="pt", padding=True).to(device)
                inputs_ru = transform(text=input_texts_ru,
                                      images=batch[img_column], return_tensors="pt", padding=True).to(device)
                inputs_rb = transform(text=input_texts_rb,
                                      images=batch[img_column], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            emb = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            if emb_type == 'image' and EXTRA_PROMPTS:
                emb_lu = model(**inputs_lu, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                emb_lb = model(**inputs_lb, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                emb_ru = model(**inputs_ru, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                emb_rb = model(**inputs_rb, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                emb = torch.cat([emb, emb_lu, emb_lb, emb_ru, emb_rb], dim=0)
            emb = F.normalize(emb, dim=-1)
        emb = accelerator.gather(emb)
        embs.append(emb.cpu().float())
        bar.update(1)
    embs = torch.cat(embs)
    total = 0
    for i in dataset:
        if emb_type == 'text' and type(i[text_column]) is list:
            total += len(i[text_column])
        else:
            total += 1
            if emb_type == 'image' and EXTRA_PROMPTS:
                total += 4
    bar.close()
    return embs[:total]


def emb_data_clip(model, transform, dataset, device,
                  emb_type='text', prompt=None, bsz=4,
                  text_column='caption', img_column='img'):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=3 * bsz if emb_type == 'text' else bsz,
        shuffle=False, num_workers=1,
        collate_fn=custom_collate_fn
    )
    dataloader = accelerator.prepare(dataloader)
    embs = []
    bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        for batch in dataloader:
            if emb_type == 'text':
                input_texts = torch.cat([clip.tokenize(t, truncate=True).to(device) for t in batch[text_column]])
                emb = model.encode_text(input_texts)
            else:
                if isinstance(batch[img_column][0], str):
                    batch[img_column] = [Image.open(img).convert('RGB') for img in batch[img_column]]
                input_images = torch.cat([transform(img).unsqueeze(0).to(device) for img in batch[img_column]])
                emb = model.encode_image(input_images)
            emb = F.normalize(emb, dim=-1)
            emb = accelerator.gather(emb)
            embs.append(emb.cpu().float())
            bar.update(1)
    embs = torch.cat(embs)
    total = 0
    for i in dataset:
        if emb_type == 'text' and type(i[text_column]) is list:
            total += len(i[text_column])
        else:
            total += 1
    bar.close()
    return embs[:total]


def log_to_file(data, metrics, checkpoint_name, difficulty=['easy', 'medium', 'hard']):
    if data == 'flickr30k' or data == 'coco':
        output = f"{data}: image R@1 {metrics['image_retrieval_recall@1']:.4f} text R@1 {metrics['text_retrieval_recall@1']:.4f} \n"
        output += f"{data}: image R@5 {metrics['image_retrieval_recall@5']:.4f} text R@5 {metrics['text_retrieval_recall@5']:.4f} \n"
        output += f"{data}: image R@10 {metrics['image_retrieval_recall@10']:.4f} text R@10 {metrics['text_retrieval_recall@10']:.4f} \n"
    else:
        output = ''
        if 'sorce' in data:
            for level in difficulty:
                output += f"{data}: {level} R@1 {metrics[f'[{level}] image_retrieval_recall@1']:.4f} \n"
                output += f"{data}: {level} R@5 {metrics[f'[{level}] image_retrieval_recall@5']:.4f} \n"
                output += f"{data}: {level} R@10 {metrics[f'[{level}] image_retrieval_recall@10']:.4f} \n"
        else:
            output += f"{data}: image R@1 {metrics['image_retrieval_recall@1']:.4f} \n"
            output += f"{data}: image R@5 {metrics['image_retrieval_recall@5']:.4f} \n"
            output += f"{data}: image R@10 {metrics['image_retrieval_recall@10']:.4f} \n"
        output += f"{data}: text R@1 {metrics['text_retrieval_recall@1']:.4f} \n"
        output += f"{data}: text R@5 {metrics['text_retrieval_recall@5']:.4f} \n"
        output += f"{data}: text R@10 {metrics['text_retrieval_recall@10']:.4f} \n"

    if checkpoint_name is not None:
        with open(checkpoint_name, 'a') as f:
            print(output, file=f)
    return output


def init_model_and_transform(lora_path, bf16, fp32, use_e5v=False, use_e5v_rep=False):
    dtype = torch.bfloat16 if bf16 else torch.float16
    if fp32:
        dtype = torch.float32

    if MODEL_TYPE == 'clip':
        model, preprocess = clip.load('./pretrained/ViT-B-16.pt')

        return model, preprocess

    elif 'qwen2vl' in MODEL_TYPE:
        MODEL_CLASS = Qwen2VLForConditionalGeneration
        min_pixels = 256 * 28 * 28
        max_pixels = 1800 * 28 * 28
        if '2B' in MODEL_TYPE:
            print("Using Qwen2VL 2B model")
            transform = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels,
                                                         max_pixels=max_pixels)
        else:
            print("Using Qwen2VL 7B model")
            transform = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels,
                                                         max_pixels=max_pixels)
        transform.tokenizer.padding_side = "left"
        transform.tokenizer.padding = True


    else:
        MODEL_CLASS = LlavaNextForConditionalGeneration
        transform = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        if MODEL_TYPE == 'llava_llama3':
            tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
            transform.tokenizer = tokenizer
            transform.tokenizer.add_tokens('<image>')
            transform.tokenizer.pad_token_id = transform.tokenizer.eos_token_id
        transform.tokenizer.padding_side = "left"
        transform.tokenizer.padding = True

    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"

    if MODEL_TYPE == 'llava_llama3':
        model_name = "./pretrained/llava-llama-3-8b"
    elif 'qwen2vl' in MODEL_TYPE:
        model_name = "./pretrained/Qwen2-VL-2B-Instruct" if '2B' in MODEL_TYPE else "./pretrained/Qwen2-VL-7B-Instruct"

    if use_e5v:
        model_name = 'royokong/e5-v'
        transform = LlavaNextProcessor.from_pretrained('royokong/e5-v')

    if use_e5v_rep:
        model_name = 'lcxrocks/e5-v-ReP'
        transform = LlavaNextProcessor.from_pretrained("lcxrocks/e5-v-ReP")

    if lora_path is not None:
        if '/' in lora_path:
            lora_subfolder, lora_filename = lora_path.split('/')[:-1], lora_path.split('/')[-1]
            merge_path = os.path.join(*lora_subfolder, 'merged-' + lora_filename.replace('/', '-').replace('.', ''))
        else:
            merge_path = 'merged-' + lora_path.replace('/', '-').replace('.', '')
        with accelerator.main_process_first():
            if not os.path.exists(merge_path):
                model = MODEL_CLASS.from_pretrained(model_name,
                                                    device_map='cpu')
                model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
                model.save_pretrained(merge_path)
                del model  # to free up memory
                torch.cuda.empty_cache()
        model_name = merge_path

    model = MODEL_CLASS.from_pretrained(model_name,
                                            torch_dtype=dtype, low_cpu_mem_usage=True)
    if MODEL_TYPE == 'llava_llama3':
        model.config.image_token_index = 128256
        if 'e5-v' not in model_name:
            model.resize_token_embeddings(len(transform.tokenizer))  # for transformers 4.48.0

        # BEGIN: transformers 4.48+ fix
        transform.patch_size = model.config.vision_config.patch_size
        transform.vision_feature_select_strategy = model.config.vision_feature_select_strategy
        transform.num_additional_image_tokens = 1  # for transformers 4.48
        # END: transformers 4.48+ fix

    return model, transform


def sorce_recall_at_k_difficulty(scores, positive_pairs, k, difficulty='easy',
                                 total_difficulties=['easy', 'medium', 'hard']):
    """
    Compute the recall at k for each sample
    :param scores: compatibility score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    assert difficulty in ['easy', 'medium',
                          'hard'] and difficulty in total_difficulties, 'difficulty should be one of easy, medium, hard'

    mod1, mod2 = scores.shape
    tmp_scores = scores.clone()
    tmp_positive_pairs = positive_pairs.clone()

    if mod1 < mod2:  # text to image retrieval
        for i in range(mod1):
            indices = torch.where(tmp_positive_pairs[i] == True)[0].chunk(len(total_difficulties))
            # NOTE:
            #  To expand the candidate pool, we combine images of three difficulty levels into one unified candidate set.
            #  However, for evaluating a ground-truth (GT) pair of a specific difficulty level,
            #  only GTs of the same difficulty are considered as successful retrievals.
            if difficulty == 'easy':
                if 'medium' in total_difficulties:
                    medium_idx = indices[1]
                    tmp_scores[i, medium_idx] = 0
                    tmp_positive_pairs[i, medium_idx] = False
                if 'hard' in total_difficulties:
                    hard_idx = indices[2] if 'medium' in total_difficulties else indices[1]
                    tmp_scores[i, hard_idx] = 0
                    tmp_positive_pairs[i, hard_idx] = False
            elif difficulty == 'medium':
                if 'easy' in total_difficulties:
                    easy_idx = indices[0]
                    tmp_scores[i, easy_idx] = 0
                    tmp_positive_pairs[i, easy_idx] = False
                if 'hard' in total_difficulties:
                    hard_idx = indices[2] if 'easy' in total_difficulties else indices[1]
                    tmp_scores[i, hard_idx] = 0
                    tmp_positive_pairs[i, hard_idx] = False
            elif difficulty == 'hard':
                if 'easy' in total_difficulties:
                    easy_idx = indices[0]
                    tmp_scores[i, easy_idx] = 0
                    tmp_positive_pairs[i, easy_idx] = False
                if 'medium' in total_difficulties:
                    medium_idx = indices[1] if 'easy' in total_difficulties else indices[0]
                    tmp_scores[i, medium_idx] = 0
                    tmp_positive_pairs[i, medium_idx] = False

    topk_indices = torch.topk(tmp_scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = tmp_positive_pairs.sum(dim=1)
    # mod2, k, mod1
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=mod2)
    # compute number of true positives
    positive_pairs_reshaped = tmp_positive_pairs.view(mod1, 1, mod2)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def sorce_ir(model, transform,
             img_prompt, text_prompt,
             data, device,
             batch_size=None, difficulty=['easy', 'medium', 'hard']):
    from datasets import concatenate_datasets
    dataset = load_dataset("json", data_files=data)['train']  # only one split

    dataset = dataset.rename_column('image', 'img')

    dirname = os.path.dirname(data)
    easy_dir = os.path.join(dirname, 'zoom_3x')
    medium_dir = os.path.join(dirname, 'zoom_2x')
    hard_dir = os.path.join(dirname, 'full_res')

    dataset_collection = []
    if 'easy' in difficulty:
        easy_dataset = dataset.map(lambda x: {'img': os.path.join(easy_dir, x['img'])}, num_proc=4)
        dataset_collection.append(easy_dataset)
    if 'medium' in difficulty:
        medium_dataset = dataset.map(lambda x: {'img': os.path.join(medium_dir, x['img'])}, num_proc=4)
        dataset_collection.append(medium_dataset)
    if 'hard' in difficulty:
        hard_dataset = dataset.map(lambda x: {'img': os.path.join(hard_dir, x['img'])}, num_proc=4)
        dataset_collection.append(hard_dataset)

    all_dataset = concatenate_datasets(dataset_collection)

    bsz = 4
    if batch_size is not None:
        bsz = batch_size

    with torch.no_grad():  # NOTE: different dataset length for image and text
        img_embs = emb_data_func(model, transform, all_dataset, device, emb_type='image', prompt=img_prompt, bsz=bsz)
        text_embs = emb_data_func(model, transform, dataset, device, emb_type='text', prompt=text_prompt, bsz=bsz)

    dataset_multiples = len(difficulty)
    image_text_index = [i for i in range(text_embs.shape[0])] * dataset_multiples

    assert text_embs.isnan().sum().item() == 0, 'nan in retrieve emb'
    assert img_embs.isnan().sum().item() == 0, 'nan in images emb'

    # get the score for each text and image pair
    scores = img_embs @ text_embs.t()
    if EXTRA_PROMPTS:
        extra_feat_num = img_embs.shape[0] // len(all_dataset)
        scores = scores.view(-1, extra_feat_num, text_embs.shape[0])
        scores = scores.max(dim=1)[0]

    print(f'Image embs shape: {img_embs.shape}')
    print(f'Text embs shape: {text_embs.shape}')
    print(f'Scores shape: {scores.shape}')

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), image_text_index] = True

    metrics = {}
    recall_k_list = [1, 5, 10]
    batch_size = 64
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        for level in difficulty:
            metrics[f"[{level}] image_retrieval_recall@{recall_k}"] = (
                    batchify(sorce_recall_at_k_difficulty, scores.T, positive_pairs.T, batch_size, device,
                             k=recall_k, difficulty=level, total_difficulties=difficulty) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (
                batchify(recall_at_k, scores, positive_pairs, batch_size, device,
                         k=recall_k) > 0).float().mean().item()

    return metrics


def ir(model, transform,
       img_prompt, text_prompt,
       data, device,
       batch_size=None):
    dataset = load_dataset(f'royokong/{data}_test', split='test')

    dataset = dataset.rename_column('text', 'caption')
    dataset = dataset.rename_column('image', 'img')
    if data == 'coco':
        dataset = dataset.map(lambda x: {'caption': x['caption'][:5]}, num_proc=4)

    bsz = 4
    if batch_size is not None:
        bsz = batch_size

    img_embs = emb_data_func(model, transform, dataset, device, emb_type='image', prompt=img_prompt, bsz=bsz)
    text_embs = emb_data_func(model, transform, dataset, device, emb_type='text', prompt=text_prompt, bsz=bsz)

    if EXTRA_PROMPTS:
        extra_feat_num = img_embs.shape[0] // (text_embs.shape[0] // 5)
        texts_image_index = [i // 5 for i in range(img_embs.shape[0] // extra_feat_num * 5)]
    else:
        texts_image_index = [i // 5 for i in range(img_embs.shape[0] * 5)]
    # assert len(texts_image_index) == len(text_embs)

    assert text_embs.isnan().sum().item() == 0, 'nan in retrieve emb'
    assert img_embs.isnan().sum().item() == 0, 'nan in images emb'

    # get the score for each text and image pair
    scores = text_embs @ img_embs.t()
    if EXTRA_PROMPTS:
        scores = scores.view(text_embs.shape[0], -1, extra_feat_num)
        scores = scores.max(dim=2)[0]

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    recall_k_list = [1, 5, 10]
    batch_size = 64
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        metrics[f"image_retrieval_recall@{recall_k}"] = (
                batchify(recall_at_k, scores, positive_pairs, batch_size, device,
                         k=recall_k) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = (
                batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device,
                         k=recall_k) > 0).float().mean().item()

    return metrics


def main(
        llava: bool = False,
        llava_llama3: bool = False,
        lora_path: str = None,
        name: str = None,
        batch_size: int = 1,
        bf16: bool = False,
        fp32: bool = False,
        data: str = None,
        debug: bool = False,
        use_e5v: bool = False,
        qwen2vl: str = None,
        clip: bool = False,
        extra_prompts: bool = False,
        position_prompts: bool = False,
        difficulty: str = None,
        use_e5v_rep: bool = False,
):
    global DEBUG, MODEL_TYPE, EXTRA_PROMPTS, PATCH_SPLIT, emb_data_func
    DEBUG = debug
    emb_data_func = emb_data
    EXTRA_PROMPTS = extra_prompts
    if position_prompts:
        emb_data_func = emb_data_position
        EXTRA_PROMPTS = True
        print("Using position prompts, set EXTRA_PROMPTS to True")

    if llava_llama3:
        MODEL_TYPE = 'llava_llama3'
    elif llava:
        MODEL_TYPE = 'llava'
    elif use_e5v or use_e5v_rep:
        llava_llama3 = True
        MODEL_TYPE = 'llava_llama3'
    elif qwen2vl is not None:
        MODEL_TYPE = f'qwen2vl-{qwen2vl}'
    elif clip:
        MODEL_TYPE = 'clip'
        emb_data_func = emb_data_clip

    assert MODEL_TYPE in ['llava', 'llava_llama3', 'qwen2vl-2B', 'qwen2vl-7B', 'clip'], f"Model type {MODEL_TYPE} not supported"

    # set NCCL_DEBUG
    if os.environ.get("NCCL_DEBUG", None) is None:
        os.environ["NCCL_DEBUG"] = "ERROR"

    device = accelerator.device

    model, transform = init_model_and_transform(lora_path, bf16, fp32, use_e5v=use_e5v, use_e5v_rep=use_e5v_rep)
    model.to(device)

    from datasets import disable_caching
    disable_caching()
    datasets = ["./datasets/sorce-1k/dataset.jsonl"]  #, 'flickr30k', 'coco'
    difficulty = ['easy', 'medium', 'hard']  # NOTE: we use all difficulties to enlarge candidate pool, see L391

    if data:
        datasets = data.split(',')

    all_results = []
    for data in datasets:
        if 'flickr30k' in data or 'coco' in data or 'sorce' in data:
            if llava_llama3:
                img_prompt = llama3_template.format('<image>\nSummarize above image in one word: ')
                text_prompt = llama3_template.format('<sent>\nSummarize above sentence in one word: ')
            elif qwen2vl:
                img_prompt = qwen2_template.format(
                    '<|vision_start|><|image_pad|><|vision_end|>\nSummarize above image in one word:')
                text_prompt = qwen2_template.format('<sent>\nSummarize above sentence in one word:')
            else:
                img_prompt = "[INST] <image>\nSummarize above image in one word: [/INST]"
                text_prompt = "[INST] <sent>\nSummarize above sentence in one word: [/INST]"

            if accelerator.is_main_process:
                print(img_prompt)
                print(text_prompt)
            if "sorce" in data.lower():
                metrics = sorce_ir(model, transform, img_prompt, text_prompt, data, device,
                                   batch_size, difficulty=difficulty)
            else:
                metrics = ir(model, transform, img_prompt, text_prompt,
                             data, device, batch_size)
        else:
            raise ValueError(f"Dataset {data} not supported. Please use 'flickr30k', 'coco' or 'sorce-1k'.")

        if accelerator.is_main_process:
            print(metrics)
            os.mkdir("results") if not os.path.exists("results") else None
            if lora_path is not None or name is not None:
                if name is not None:
                    checkpoint_name = name + '.txt'
                    checkpoint_name = os.path.join("results", checkpoint_name)
                else:
                    if '/' in lora_path:
                        lora_subfolder, lora_filename = lora_path.split('/')[:-1], lora_path.split('/')[-1]
                        checkpoint_name = os.path.join("results", lora_filename + '.txt')
                    else:
                        checkpoint_name = lora_path + '.txt'
                        checkpoint_name = os.path.join("results", checkpoint_name)
            else:
                checkpoint_name = 'temp_file'
                checkpoint_name = os.path.join("results", checkpoint_name)
            all_results.append(log_to_file(data, metrics, checkpoint_name, difficulty=difficulty))

    if accelerator.is_main_process:
        print('\n'.join(all_results))


if __name__ == '__main__':
    from fire import Fire

    Fire(main)
