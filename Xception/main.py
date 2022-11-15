import cv2
import numpy as np
import pandas as pd
# %matplotlib inline
from matplotlib import pyplot as plt

df = pd.read_csv('CheXpert-v1.0-small/train.csv') # Earlier, df_train
df_val = pd.read_csv('CheXpert-v1.0-small/valid.csv')

def clean_df(df):
  # Focusing only on 5 classes:
  df = df[[
    'Path', 
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion'
  ]]

  # Handling the NaN values
  df = df.fillna(0)

  # Handling the uncertain values
  ## Different policy for each feature:
  u_ones = ['Atelectasis', 'Edema']
  u_zeros = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']
  df[u_ones]  = df[u_ones].replace(-1, 1)
  df[u_zeros] = df[u_zeros].replace(-1, 0)
  print(df.shape)
  return df

df = clean_df(df)
df_val = clean_df(df_val)

BATCH_SIZE = 32
IMAGE_SIZE = 224
CLASSES = [ 
  'Atelectasis',
  'Cardiomegaly',
  'Consolidation',
  'Edema',
  'Pleural Effusion'
]

# Constants
FRAC = 1 # Fraction of total data to be taken as sample
SHAPE = (320, 390, 3) # Common shape for featurewise centering & normalization

sample_paths = df['Path'].sample(frac=FRAC).to_numpy()

print(sample_paths.shape)
X_temp = np.array([np.array(cv2.imread(path, 1), dtype=float) for path in sample_paths])
X_sample = np.array([x for x in X_temp if x.shape == SHAPE])

# print(X_temp)

from keras.preprocessing.image import ImageDataGenerator as IDG

datagen = IDG(
    rescale=1./255, 
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=0.1,
    zoom_range = 0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split = 0.1,
    fill_mode = 'nearest'
)
# print(X_sample.shape)
datagen.fit(X_sample)

test_datagen = IDG(rescale=1./255)


def get_gen():
  train_gen = datagen.flow_from_dataframe(
      dataframe = df,
      #directory = '/content/CheXpert-v1.0-small/train',
      # directory = "/content/Chexpert-master/Chexpert/dataset/",
      x_col = 'Path',
      y_col = CLASSES, #'classes',
      class_mode='raw',
      #validate_filenames = False,
      seed=42,
      shuffle=True,
      target_size=(IMAGE_SIZE, IMAGE_SIZE), 
      batch_size=BATCH_SIZE, 
      subset = 'training'
  )

  val_gen = datagen.flow_from_dataframe(
      dataframe = df,
      #directory = '/content/CheXpert-v1.0-small/train',
      # directory = "/content/Chexpert-master/Chexpert/dataset/",
      x_col = 'Path',
      y_col = CLASSES, #'classes',
      class_mode='raw',
      #validate_filenames = False,
      seed=42,
      shuffle=True,
      target_size=(IMAGE_SIZE, IMAGE_SIZE), 
      batch_size=BATCH_SIZE, 
      #classes = columns,
      subset = 'validation'
  )

  return train_gen, val_gen

from keras.callbacks import *

class CyclicLR(Callback):
    
    def __init__(self, base_lr=0.001, max_lr=0.01, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())




# Building on top of the base:
from keras.applications import Xception
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout

def build_model():
  # The convolutional base:
  model_base = Xception(
      weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
      )
  #model_base.trainable = False
  # Unfreezing all the layers:
  for layer in model_base.layers:
      layer.trainable = True

  model = Sequential()
  model.add(model_base) # Adding the base as a layer
  model.add(GlobalAveragePooling2D())
  model.add(Dense(1024, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.3))
  #model.add(Flatten())
  #model.add(Dense(1024, activation='relu'))
  #model.add(Dropout(0.25))
  model.add(Dense(5, activation='sigmoid'))
  
  return model

from keras.metrics import AUC, categorical_accuracy as catacc
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

auc = AUC()
adam = Adam(learning_rate=0.00005) # 0.05 of default

es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=2)
mc = ModelCheckpoint(
    filepath='xception-keras-2.h5', verbose=1 #, save_best_only=True
)

clr = CyclicLR(base_lr=0.00005, max_lr=0.0001)
cb_list = [es, mc] # Will add clr later, as we'll have to tune it's hyperparameters

model = build_model()
model.compile(
    loss='binary_crossentropy',
    optimizer=adam,
    metrics=[auc, catacc] # Earlier, 'acc' 
)

train_gen, val_gen = get_gen()

TRAIN_STEPS = train_gen.n//BATCH_SIZE
VAL_STEPS   = val_gen.n//BATCH_SIZE
N_EPOCHS = 5

history = model.fit_generator(
    train_gen,
    steps_per_epoch=TRAIN_STEPS,
    epochs=N_EPOCHS,
    validation_data=val_gen,
    validation_steps=VAL_STEPS,
    callbacks = cb_list
)

cat_acc = history.history['categorical_accuracy']
val_cat_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(cat_acc) + 1)

plt.plot(epochs, cat_acc, 'b', label='Training accuracy')
plt.plot(epochs, val_cat_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_gen = test_datagen.flow_from_dataframe(
    dataframe = df_val,
    #directory = '/content/CheXpert-v1.0-small/valid',
    x_col = 'Path',
    y_col = CLASSES, #'classes',
    class_mode='raw',
    #validate_filenames = False,
    target_size=(IMAGE_SIZE, IMAGE_SIZE), 
    batch_size=1, 
    shuffle = False,
    #classes = columns,
)  

y_labels = df_val[CLASSES].to_numpy()
y_pred = model.predict_generator(test_gen, steps=test_gen.n) 


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

for i in range(len(CLASSES)):
   fpr, tpr, thresholds = roc_curve(y_labels[:, i], y_pred[:, i])
   individual_auc = auc(fpr, tpr)
   plt.plot(fpr, tpr, label= (CLASSES[i] + '(area = {})'.format(individual_auc)))

    
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


results = model.evaluate_generator(test_gen)

print(results)
print(y_labels)
print(y_pred)
pd.DataFrame(y_labels).to_csv("ylabels.csv")
pd.DataFrame(y_pred).to_csv("ypred.csv")

