# import detect
import railib.liability.augmentation.automold as am

from pathlib import Path
import PIL
from PIL import Image, ImageFont, ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
from collections import Counter
from pathlib import Path
import json
import random
import subprocess
import cv2
from google.colab.patches import cv2_imshow

LABELS = ['Prohibitory', 'Priority', 'Danger', 'Mandatory']

BASE_PATH = '/content'

DETECT_PATH = BASE_PATH + '/yolo5/detect.py'
DETECT_EXP_PATH = BASE_PATH + '/yolo5/runs/detect'

GT_PATH = BASE_PATH + '/data/data/GT.csv'

TEST_TXT_PATH = BASE_PATH + '/data/data/test.txt'
TRAIN_TXT_PATH = BASE_PATH + '/data/data/train.txt'

TEST_LABELS_PATH = BASE_PATH + '/data/exp/exp32/labels'
TRAIN_LABELS_PATH = BASE_PATH + '/data/exp/exp36/labels'

TEST_JSON_LABELS_PATH = BASE_PATH + '/data/exp/exp32/labels_data.json'
TRAIN_JSON_LABELS_PATH = BASE_PATH + '/data/exp/exp36/labels_data.json'

def map_labels(label):
  label = int(label)
  new_label = 0
  # prohibitory - 0 
  if (label in range(0, 10)) or (label in range(15, 17)):
      new_label = 0
  # priority - 1 
  elif (label in range(11, 14)):
      new_label = 1
  # danger 2
  elif label in range(18, 32):
      new_label = 2
  # mandatory - 3
  elif label in range(33, 42):
      new_label = 3
  return new_label


def convert_pred_image_id_to_GT_format(image):
  new_image = ((5 - len(str(image))) * '0') + str(image) + '.txt'
  return new_image


def get_images_list(path):
  images_list = os.listdir(path.split('/labels')[0])
  for i in images_list:
      if not i.endswith('.jpg'):
          images_list.remove(i)
  if 'labels_data.json' in images_list:
      images_list.remove('labels_data.json')
  if 'labels' in images_list:
      images_list.remove('labels')
  return [int(i.replace('.jpg', '').replace('.png', '')) for i in images_list]


def evaluation(pred_path, GT_path=GT_PATH, thresh=0.25, num_of_classes=4, print_results=True, categories_reminder=False):
  GT_df = pd.read_csv(GT_path)
  GT_df['new_label'] = GT_df.apply(lambda row: map_labels(row.label), axis=1)

  pred_df = pd.read_json(pred_path)

  image_list = set(get_images_list(pred_path))
  image_list_GT_format = [convert_pred_image_id_to_GT_format(i) for i in image_list]

  GT_df_test = GT_df.loc[GT_df['id'].isin(image_list_GT_format)]

  TP = [0] * num_of_classes
  FP = [0] * num_of_classes
  TN = [0] * num_of_classes
  FN = [0] * num_of_classes
  results_dict = {}

  for image in image_list:
      pred_labels = Counter(list(pred_df[(pred_df['image_id'] == image) & (pred_df['score'] > thresh)].category_id))
      pred_labels_list = pred_labels.keys()

      id = ((5 - len(str(image))) * '0') + str(image) + '.txt'
      GT_labels = Counter(list(GT_df[(GT_df['id'] == id)].new_label))
      GT_labels_list = GT_labels.keys()

      pred_arr = [0] * num_of_classes
      GT_arr = [0] * num_of_classes

      for i in range(num_of_classes):
          if (i in pred_labels_list) and (i in GT_labels_list):
              TP[i] += 1
              pred_arr[i] = 1
              GT_arr[i] = 1
          elif (i in pred_labels_list) and (i not in GT_labels_list):
              FP[i] += 1
              pred_arr[i] = 1
          elif (i not in pred_labels_list) and (i not in GT_labels_list):
              TN[i] += 1
          else:
              FN[i] += 1
              GT_arr[i] = 1

      results_dict[image] = [GT_arr, list(GT_labels_list), pred_arr, list(pred_labels_list)]

  results_df = pd.DataFrame.from_dict(results_dict, orient='index', 
                                      columns=['GT', 'GT_indexes', 'pred', 'pred_indexes'])
  accuracy = round((sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN)), 2)
  
  conf_matrix = pd.DataFrame(columns=LABELS)
  conf_matrix.loc['TP'] = TP
  conf_matrix.loc['FP'] = FP
  conf_matrix.loc['TN'] = TN
  conf_matrix.loc['FN'] = FN

  if categories_reminder:
      present_image('./data/explanation_images/categories_reminder.png')
      
  if print_results:
      print(f'-------------model accuracy: {accuracy}--------------')
      print(f'number of images in test set: {len(image_list)}')
      print(conf_matrix)
      print('-----------------------------------------------')

  return accuracy


def convert_label(label):
    return LABELS[int(label)]


def get_GT_labels(im):
  GT_df = pd.read_csv(GT_PATH)
  GT_df['new_label'] = GT_df.apply(lambda row: map_labels(row.label), axis=1)
  id = ((5 - len(str(im))) * '0') + str(im)
  if not id.endswith('.txt'):
      id = id.replace('.jpg', '.txt')
  GT_labels = Counter(list(GT_df[(GT_df['id'] == id)].new_label))
  GT_labels_list = GT_labels.keys()
  return list(GT_labels_list)


def get_im_labels(im, labels_root_path, GT=True):
  filename = f'{labels_root_path}/{im.replace(".jpg", ".txt")}'
  GT_text = f'GT: {[LABELS[i] for i in get_GT_labels(im)]} \n'

  if not os.path.isfile(filename):
      text = 'No traffic signs were detected \n'
      if GT:
          return text + GT_text
      return text

  image_labels_df = pd.read_table(filename, sep=' ', names=['label', '1', '2', '3', '4', 'confidence'])
  image_labels_df['label_name'] = image_labels_df.apply(lambda row: convert_label(row.label), axis=1)
  labels_list = list(image_labels_df['label_name'].unique())
  text = ''
  for label in labels_list:
      conf = image_labels_df[image_labels_df['label_name'] == label].confidence.max()
      text += f'{label}: {round(conf, 2)}\n'

  if GT:
      text += GT_text

  return text


# write over an image
def im_write(im_path, labels_path):
  im_id = im_path.split('/')[-1]
  img = Image.open(im_path)

  draw = ImageDraw.Draw(img)
  font = ImageFont.truetype('./src/arial.ttf', 40)

  labels = get_im_labels(im_id, labels_path, False)

  x, y = 0, 0
  w, h = font.getsize(labels)
  h *= sum('\n' in item for item in labels)
  # w = sum('\n' in item for item in labels)
  draw.rectangle((x, y, x + w, y + h), fill='black')

  draw.text((0, 0), labels, (209, 239, 8), font=font)

  plt.figure(figsize=(200, 100))
#   plt.axis('off')

  im_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
  cv2_imshow(im_bgr)
  # imshow(np.asarray(img))


def present_image(filename):
  img = Image.open(filename)
#   plt.figure(figsize=(200, 100))
#   plt.axis('off')
  im_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
  cv2_imshow(im_bgr)
  # imshow(np.asarray(img))

def present_matrix_of_images(filename='./data/data/test.txt', x_num_of_im=5, y_num_of_im=5, 
                            labels_path='', sample=True, with_prediction=False):
  paths_file = Path(filename)
  img_paths = paths_file.read_text().splitlines()

  fig = plt.figure(figsize=(100, 60))

  num_of_images_to_present = x_num_of_im * y_num_of_im
  if num_of_images_to_present > len(img_paths):
      num_of_images_to_present = len(img_paths)

  if sample:
      random.shuffle(img_paths)

  for i in range(num_of_images_to_present):
      image = Path(img_paths[i]).stem + '.jpg'

      ax = fig.add_subplot(x_num_of_im, y_num_of_im, i + 1, xticks=[], yticks=[])
      img = Image.open(img_paths[i])
      draw = ImageDraw.Draw(img)

      if with_prediction:
          # font = ImageFont.truetype(<font-file>, <font-size>)
          font = ImageFont.truetype('./src/arial-bold.ttf', 20)
          x, y = 0, 0
          labels = get_im_labels(image, labels_path)

          w, h = font.getsize(labels)
          h *= sum('\n' in item for item in labels)
          # w = sum('\n' in item for item in labels)
          draw.rectangle((x, y, x + w, y + h), fill='black')

          # draw.text((x, y), "Sample Text", (r, g, b))
          draw.text((0, 0), labels, (209, 239, 8), font=font)
      # plt.imshow(img)
      im_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
      cv2_imshow(im_bgr)
      # ax.set_title(f'{labels}')


def get_detection_labels_folder(detect_output):
  return detect_output[-2].split(' ')[-1]


def evaluation_plot_bar(D):
  plt.bar(range(len(D)), list(D.values()), align='center')
  plt.xticks(range(len(D)), list(D.keys()))
  xlocs, xlabs = plt.xticks()

  # Add title and axis names
  # plt.title('title')
  plt.xlabel('Augmentation')
  plt.ylabel('Accuracy')

  for i, v in enumerate(list(D.values())):
      plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
  plt.show()


def horizontal_evaluation_plot_bar(D):
  plt.rcdefaults()
  fig, ax = plt.subplots()

  aug = list(D.keys())
  y_pos = np.arange(len(aug))
  accuracy = list(D.values())

  ylocs, ylabs = plt.yticks()

  ax.barh(y_pos, accuracy, align='center')
  ax.set_yticks(y_pos)
  ax.set_yticklabels(aug)
  ax.invert_yaxis()  # labels read top-to-bottom
  ax.set_xlabel('Accuracy')
  ax.set_ylabel('Augmentation')

  for i, v in enumerate(list(D.values())):
      plt.text(v, i, str(round(v, 2)), color='steelblue', va="center")

  #plt.show()
  return ax


def save_to_json(labels_path):
  json_file = []
  for subdir, dirs, files in os.walk(labels_path):
      for file in files:
          if file.endswith('.txt'):
              file_path = Path(os.path.join(subdir, file))
              image_labels = file_path.read_text().splitlines()
              image_id = file.replace('.txt', '')
              for line in image_labels:
                  category_id = line.split(' ')[0]
                  score = line.split(' ')[-1]
                  line_to_dict = {'image_id': image_id, 'category_id': category_id, 'score': score}
                  json_file.append(line_to_dict)

  save_path = f'{labels_path}_data.json'
  with open(save_path, 'w', encoding='utf-8') as f:
      json.dump(json_file, f, ensure_ascii=False, indent=4)

  return save_path


def plot_all_aug_eval_bar_plot():
  # 4 classes
  light_rain_results_path = './data/exp/exp56/labels_data.json'
  heavy_rain_results_path = './data/exp/exp33/labels_data.json'
  torrential_rain = './data/exp/exp53/labels_data.json'
  dust_results_path = './data/exp/exp35/labels_data.json'
  light_rain_and_dust_results_path = './data/exp/exp55/labels_data.json'
  heavy_rain_and_dust_results_path = './data/exp/exp34/labels_data.json'
  torrential_rain_and_dust_results_path = './data/exp/exp54/labels_data.json'
  no_aug_path = './data/exp/exp32/labels_data.json'


  eval_dict = {'drizzle_rain': evaluation(light_rain_results_path, print_results=False), 
                'heavy_rain': evaluation(heavy_rain_results_path, print_results=False), 
                'torrential_rain': evaluation(torrential_rain, print_results=False), 
                'dust': evaluation(dust_results_path, print_results=False), 
                'light_rain_AND_dust': evaluation(light_rain_and_dust_results_path, print_results=False), 
                'heavy_rain_AND_dust': evaluation(heavy_rain_and_dust_results_path, print_results=False), 
                'torrential_rain_AND_dust': evaluation(torrential_rain_and_dust_results_path, print_results=False), 
                'no_aug': evaluation(no_aug_path, print_results=False)}
  
  return horizontal_evaluation_plot_bar(eval_dict)


def run_detect(images_folder, model='./models/last.pt'):
  r = subprocess.run(f"python {DETECT_PATH} --weights {model} --source {images_folder} --save-txt --save-conf".split(),capture_output=True)
  if r.returncode == 0:
    # labels_path = str(r.stdout).split('\\n')[-3].split()[-1]
    detect_exp_num = str(r.stderr).split('Results saved to ')[1].split('exp')[1].split('\\')[0]
    labels_path = f'{DETECT_EXP_PATH}/exp{detect_exp_num}/labels'
    return labels_path
  return 'ERROR: ' + str(r)

def apply_augmentation(filename, aug_func, **aug_params):
    
  paths_file = Path(filename)

  func_name = aug_func.__name__
  func_params_desc = '-'.join(map(str, aug_params.values())).replace('.', '_')

  if func_params_desc:
    func_params_desc = '_' + func_params_desc
  idetifier = f'{paths_file.stem}--aug-{func_name}{func_params_desc}'

  output_path = paths_file.parent  / idetifier
  output_path.mkdir(exist_ok=True)

  img_paths = paths_file.read_text().splitlines()


  output_img_paths = []

  for img_path in img_paths:

    img = np.array(PIL.Image.open(img_path))

    img = aug_func(img, **aug_params)

    filename = Path(img_path).stem + '.jpg'
    output_img_path = str(output_path / filename)
    output_img_paths.append(output_img_path)
    
    PIL.Image.fromarray(img).save(output_img_path)

  output_paths_file = paths_file.parent / idetifier
  output_paths_file = output_paths_file.with_suffix('.txt')
  output_paths_file.write_text('\n'.join(output_img_paths))

  return str(output_paths_file)
