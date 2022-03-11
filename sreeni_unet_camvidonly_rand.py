import itertools
import random
import numpy as np
import tensorflow
import time
import cv2
import glob as glob
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from matplotlib import pyplot as plt
from tensorflow.keras.losses import MeanSquaredError
from unet import model, model2#, combined
from load_numpy_data_camvid import generator, generator2, generator3

#""""
color_map = {
        0: (64, 128, 64),	#Animal
        1: (192, 0, 128),	#Archway
        2: (0, 128, 192),	#Bicyclist
        3: (0, 128, 64),	#Bridge
        4: (128, 0, 0),		#Building
        5: (64, 0, 128),	#Car
        6: (64, 0, 192),	#CartLuggagePram
        7: (192, 128, 64),	#Child
        8: (192, 192, 128),	#Column_Pole
        9: (64, 64, 128),	#Fence
        10: (128, 0, 192),	#LaneMkgsDriv
        11: (192, 0, 64),	#LaneMkgsNonDriv
        12: (128, 128, 64),	#Misc_Text
        13: (192, 0, 192),	#MotorcycleScooter
        14: (128, 64, 64),	#OtherMoving
        15: (64, 192, 128),	#ParkingBlock
        16: (64, 64, 0),	#Pedestrian
        17: (128, 64, 128),	#Road
        18: (128, 128, 192),	#RoadShoulder
        19: (0, 0, 192),		#Sidewalk
        20: (192, 128, 128),	#SignSymbol
        21: (128, 128, 128),	#Sky
        22: (64, 128, 192),	#SUVPickupTruck
        23: (0, 0, 64),		#TrafficCone
        24: (0, 64, 64),		#TrafficLight
        25: (192, 64, 128),	#Train
        26: (128, 128, 0),	#Tree
        27: (192, 128, 192),	#Truck_Bus
        28: (64, 0, 64),		#Tunnel
        29: (192, 192, 0),	#VegetationMisc
        30: (0, 0, 0),		#Void
        31: (64, 192, 0)	#Wall
}
#"""
""""  Cityscape
color_map = {
        0: (  0,  0,  0),	#unlabeled
        1: (111, 74,  0),	#dynamic
        2: ( 81,  0, 81),	#ground
        3: (128, 64,128),	#road
        4: (244, 35,232),		#sidewalk
        5: (250,170,160),	#parking
        6: (230,150,140),	#rail track
        7: ( 70, 70, 70),	#building
        8: (102,102,156),	#wall
        9: (190,153,153),	#fence
        10: (180,165,180),	#guard rail
        11: (150,100,100),	#bridge
        12: (150,120, 90),	#tunnel
        13: (153,153,153),	#pole
        14: (250,170, 30),	#traffic light
        15: (220,220,  0),	#traffic sign
        16: (107,142, 35),	#vegetation
        17: (152,251,152),	#terrain
        18: ( 70,130,180),		#sky
        19: (220, 20, 60),	#person
        20: (255,  0,  0),	#rider
        21: (  0,  0,142),	#car
        22: (  0,  0, 70),		#truck
        23: (  0, 60,100),		#bus
        24: (  0,  0, 90),	#caravan
        25: (  0,  0,110),	#trailer
        26: (  0, 80,100),	#train
        27: (  0,  0,230),		#motorcycle
        28: (119, 11, 32),	#bicycle
        29:  (  0,  0,142),		#license plate
        30: (1, 1, 1),  # extra
        31: (2, 2, 2)  # extra
}
"""
def rgb_to_mask(img, color_map):
    #    Converts a RGB image mask of shape [batch_size, h, w, 3] to Binary Mask of shape [batch_size, classes, h, w]
    # Parameters:img: A RGB img mask
    # color_map: Dictionary representing color mappings
    # returns:out: A Binary Mask of shape [batch_size, classes, h, w]
    num_classes = len(color_map)
    shape = img.shape[:2] + (num_classes,)
    out = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(color_map):
        out[:, :, i] = np.all(img.reshape((-1, 3)) == color_map[i], axis=1).reshape(shape[:2])
    return out

def generate_fake_samples(datasetx,datasety,batch):
    idx = np.random.randint(0,datasetx.shape[0],batch)
    x = datasetx[idx]
    y = datasety[idx]
    return x,y

samples = sorted(glob.glob("camvid/train_images/*.png"))
samples2 = sorted(glob.glob("camvid/train_masks/*.png"))
samples3 = sorted(glob.glob("camvid/test_images/*.png"))
samples4 = sorted(glob.glob("camvid/test_masks/*.png"))

#train samples should me less than real samples

# samples = sorted(glob.glob("cityscape/train_images/*.png"))
# samples2 = sorted(glob.glob("cityscape/train_GT/*.png"))
# samples3 = sorted(glob.glob("cityscape/test_images/*.png"))
# samples4 = sorted(glob.glob("cityscape/test_GT/*.png"))
# samples5 = sorted(glob.glob("camvid/train_images/*.png"))
# samples6 = sorted(glob.glob("camvid/train_masks/*.png"))
# #train samples should me less than real samples

print("\n# of trainx= %d" % len(samples))
print("# of trainy= %d" % len(samples2))
print("# of valx= %d" % len(samples3))
print("# of valy= %d" % len(samples4))

IMG_SIZE = 128 # for resize

train_x = []
for i in samples:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    img = np.float32(img) / 255
    train_x.append(img)
    #
train_y = []
for i in samples2:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    img = rgb_to_mask(img, color_map)  # 256,256,32 (already in one hot form)
    train_y.append(img)
    #
val_x = []
for i in samples3:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    img = np.float32(img) / 255
    val_x.append(img)
    #
val_y = []
for i in samples4:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    img = rgb_to_mask(img, color_map)  # 256,256,32 (already in one hot form)
    val_y.append(img)
    #

train_x = np.array(train_x)
train_y = np.array(train_y)
val_x = np.array(val_x)
val_y = np.array(val_y)

print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)

# num = random.randint(0, len(train_x)-1)
# num2 = random.randint(0, len(val_x)-1)
# num3 = random.randint(0, len(real_x)-1)
# plt.imshow(train_x[num])
# plt.show()
# plt.imshow(train_y[num])
# plt.show()
# plt.imshow(val_x[num2])
# plt.show()
# plt.imshow(val_y[num2])
# plt.show()
# plt.imshow(real_x[num3])
# plt.show()
# plt.imshow(real_y[num3])
# plt.show()

batch_size = 16

model = model(input_shape=(128,128,3))
model.compile(optimizer=tensorflow.keras.optimizers.Adam(1e-3), loss="CategoricalCrossentropy", metrics=["accuracy"])

BEST_VAL_G_LOSS = 10.0
start = time.time()
NUM_EPOCHS = 500
batch_per_epoch = int(train_x.shape[0]/batch_size)
for epoch in tqdm(range(NUM_EPOCHS)):
    print(f"===== Epoch {epoch + 1}/{NUM_EPOCHS} started! =====")
    g_losses = []
    g_acc = []
    g_val_losses = []
    g_val_acc = []
    #for train_x,train_y,target_x,target_y in combined_geonerator:
    #for [train_x, train_y], realx in tqdm(final_gen):
    for j in tqdm(range(batch_per_epoch)):
        train_x2, train_y2 = generate_fake_samples(train_x, train_y, batch_size)
        val_x2, val_y2 = generate_fake_samples(val_x, val_y, batch_size)
        seg_out = model.predict_on_batch(train_x2)
        #gen_loss,gen_acc,adv_loss,adv_acc,_ = gan_model.train_on_batch([train_x, seg_out], [train_y, real_label])
        gen_loss, gen_acc = model.train_on_batch(train_x2, train_y2)
        #print(model.metrics_names)
        #['loss', 'accuracy']
        #The loss is the weighted sum of the individual losses provided for various outputs of the model. If no class_weights are provided, the loss is simply the sum of my_loss_1, my_loss_2

        val_loss, val_acc = model.test_on_batch(val_x2, val_y2)
        #Runs a single gradient update on a single batch of data.
        g_losses.append(gen_loss)
        g_acc.append(gen_acc)
        g_val_losses.append(val_loss)
        g_val_acc.append(val_acc)
        #break
    # cannot use this method in adv learning, because wwee want to change the loss, here we cant
    # because train on batch loss calculate and grad find and optimizer update is internal
    # Convert the list of losses to an array to make it easy to average
    g_losses = np.array(g_losses)
    g_acc = np.array(g_acc)
    g_val_losses = np.array(g_val_losses)
    g_val_acc = np.array(g_val_acc)

    # Calculate the average losses for generator and discriminator
    g_loss_f = np.sum(g_losses, axis=0) / len(g_losses)
    g_acc_f = np.sum(g_acc, axis=0) / len(g_acc)
    g_val_loss_f = np.sum(g_val_losses, axis=0) / len(g_val_losses)
    g_val_acc_f = np.sum(g_val_acc, axis=0) / len(g_val_acc)

    # Report the progress during training.
    print("epoch:", epoch + 1, "g_loss:", g_loss_f,"g_acc:", g_acc_f,"g_val_loss:", g_val_loss_f,"g_val_acc:", g_val_acc_f)
    # if (epoch + 1) % 10 == 0:  # Change the frequency for model saving, if needed
    #     # Save the generator after every n epochs (Usually 10 epochs)
    #     model.save("gen_e_" + str(epoch + 1) + ".h5")

    if (epoch + 1) % 10 == 0:  # Change the frequency for model saving, if needed
        if g_val_loss_f < BEST_VAL_G_LOSS:
            BEST_VAL_G_LOSS = g_loss_f
            model.save("gen_e_camvid_rand" + str(epoch + 1) + ".h5")
        else: None