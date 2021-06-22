
# DATASET_DIR = '/content/voxmini/'
#DATASET_DIR = '/content/standard_huanluyen_test/standard-train/'
DATASET_DIR = 'data/vox1/training_set/'
TEST_DIR = 'data/vox1/testing_set/'
# DATASET_DIR = '/content/data-train-npy/data-train-npy/'
#TEST_DIR = '/content/standard_huanluyen_test/standard-test/'
TEST_DIR_WAV = 'data/vox1/test_wav/'
WAV_DIR = 'data/vox1/wav'
KALDI_DIR = ''

BATCH_SIZE = 128*1    #must be even
TRIPLET_PER_BATCH = 3

SAVE_PER_EPOCHS = 200
TEST_PER_EPOCHS = 200
CANDIDATES_PER_BATCH = 640      # 18s per batch
TEST_NEGATIVE_No = 99


NUM_FRAMES = 160   # 299 - 16*2
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
ALPHA = 0.2
HIST_TABLE_SIZE = 10
NUM_SPEAKERS = 251
DATA_STACK_SIZE = 10
CHECKPOINT_FOLDER = 'src/check_point'
BEST_CHECKPOINT_FOLDER = 'bestcheckpoint_inception_resnet_v2'
# BEST_CHECKPOINT_FOLDER = 'bestcheckpoint_mobilenet'
# BEST_CHECKPOINT_FOLDER = 'BestCheckPointMobileNetV1'
# CHECKPOINT_FOLDER = 'checkpointvgg'
# CHECKPOINT_FOLDER = 'checkpoint2'
# CHECKPOINT_FOLDER = 'checkpoint_mobilenet'
# CHECKPOINT_FOLDER = 'MobileNetV1'
# CHECKPOINT_FOLDER = 'checkpoint_mobilenetv2'
# CHECKPOINT_FOLDER = 'checkpoint_mobilenetv3'
# CHECKPOINT_FOLDER = 'checkpoint_efficient'
# CHECKPOINT_FOLDER = '/content/gdrive/My Drive/Deep_Speaker-speaker_recognition_system-master/bestcheckpoint_incep_resnetv2'
# BEST_CHECKPOINT_FOLDER = 'bestcheckpoint_mobilev2'
# BEST_CHECKPOINT_FOLDER = 'bestcheckpoint2'
# BEST_CHECKPOINT_FOLDER = 'bestcheckpoint_efficient'
# BEST_CHECKPOINT_FOLDER = '/content/gdrive/My Drive/Deep_Speaker-speaker_recognition_system-master/bestcheckpoint_incep_resnetv2'
PRE_CHECKPOINT_FOLDER = 'pretraining_checkpoints'
GRU_CHECKPOINT_FOLDER = 'gru_checkpoints'

LOSS_LOG= CHECKPOINT_FOLDER + '/losses.txt'
TEST_LOG= CHECKPOINT_FOLDER + '/acc_eer.txt'
TEST_EER= CHECKPOINT_FOLDER + '/test_eer.txt'
TEST_NEW= CHECKPOINT_FOLDER + '/test_new.txt'
PRE_TRAIN = False

COMBINE_MODEL = False
RESULT_FILE = "res/results.csv"
SUBMIT_FILE = "res/submit.csv"

THRESHOLD = 0.601